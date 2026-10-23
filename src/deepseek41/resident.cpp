#include "strata/deepseek41/resident.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <fstream>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace strata::deepseek41 {
namespace {
constexpr size_t huge = 2ull<<20;
// Linux long-term pinning can allocate a page-pointer array proportional to
// the registration size. Use smaller regions on fragmented hosts, while the
// transfer helper below honors exactly the same registration boundaries.
size_t registration_chunk() {
    static const size_t bytes = [] {
        const char* value = std::getenv("STRATA_DS41_REGISTER_64M");
        return value && std::strcmp(value,"1") == 0 ? 64ull<<20 : 1ull<<30;
    }();
    return bytes;
}
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
}
struct ResidentWeights::Impl {
    struct Range {
        uint8_t* address = nullptr;
        size_t bytes = 0, allocated = 0;
        std::vector<size_t> registered;
        pid_t helper = -1;
    };
    std::map<std::string,Range> ranges;
    uint64_t total = 0;
    ~Impl() {
        for (auto& entry : ranges) {
            auto& r = entry.second;
            for (size_t off : r.registered) cudaHostUnregister(r.address+off);
            if (r.allocated) munmap(r.address,r.allocated);
            if (r.helper > 0) { kill(r.helper,SIGTERM); while (waitpid(r.helper,nullptr,0) < 0 && errno == EINTR) {} }
        }
    }
    void own(const std::string& name,TensorRef ref) {
        auto& r = ranges[name]; r.bytes = ref.bytes(); r.allocated = (r.bytes+huge-1)&~(huge-1);
        void* allocation = mmap(nullptr,r.allocated+huge,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (allocation == MAP_FAILED) { r.allocated = 0; throw std::runtime_error("cannot allocate resident expert "+name); }
        const uintptr_t base = reinterpret_cast<uintptr_t>(allocation), aligned = (base+huge-1)&~(huge-1);
        if (aligned > base) munmap(allocation,aligned-base);
        if (base+huge > aligned) munmap(reinterpret_cast<void*>(aligned+r.allocated),base+huge-aligned);
        r.address = reinterpret_cast<uint8_t*>(aligned);
        madvise(r.address,r.allocated,MADV_HUGEPAGE);
        std::atomic<size_t> next{0}; std::atomic<bool> equal{true};
        std::vector<std::thread> copy;
        for (int i = 0; i < 16; ++i) copy.emplace_back([&] {
            for (;;) {
                const size_t off = next.fetch_add(huge); if (off >= r.bytes) break;
                const size_t n = std::min(huge,r.bytes-off);
                std::memcpy(r.address+off,ref.data()+off,n);
                if (std::memcmp(r.address+off,ref.data()+off,n)) equal = false;
            }
        });
        for (auto& t : copy) t.join();
        if (!equal) throw std::runtime_error("resident raw-byte verification failed: "+name);
        const size_t chunk = registration_chunk();
        for (size_t off = 0; off < r.allocated; off += chunk) {
            const size_t bytes = std::min(chunk,r.allocated-off);
            const auto status = cudaHostRegister(r.address+off,bytes,cudaHostRegisterPortable);
            if (status != cudaSuccess)
                throw std::runtime_error("CUDA host registration failed: "+name+
                    " offset="+std::to_string(off)+" bytes="+std::to_string(bytes)+
                    ": "+cudaGetErrorString(status));
            r.registered.push_back(off);
        }
        // Release only this tensor's original file pages, after exact copying
        // and locking. Every inference access uses the owned pointer above.
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        const uintptr_t begin = (reinterpret_cast<uintptr_t>(ref.data())+page-1)&~(page-1);
        const uintptr_t end = (reinterpret_cast<uintptr_t>(ref.data())+r.bytes)&~(page-1);
        if (end > begin) madvise(reinterpret_cast<void*>(begin),end-begin,MADV_DONTNEED);
        const int fd = open(ref.file->path().c_str(),O_RDONLY|O_CLOEXEC);
        if (fd >= 0) { posix_fadvise(fd,ref.file->data_start()+ref.tensor->offset,r.bytes,POSIX_FADV_DONTNEED); close(fd); }
        total += r.bytes;
    }
    void lock_table(const std::string& name,TensorRef ref) {
        auto& r = ranges[name]; r.address = const_cast<uint8_t*>(ref.data()); r.bytes = ref.bytes();
        char exe[4096]; const auto n = readlink("/proc/self/exe",exe,sizeof(exe)-1);
        if (n <= 0) throw std::runtime_error("cannot locate RAM lock helper");
        std::string helper(exe,size_t(n)); helper = helper.substr(0,helper.rfind('/')+1)+"strata-deepseek41-ram-lock";
        std::vector<std::string> args = {helper,ref.file->path(),std::to_string(ref.file->data_start()+ref.tensor->offset),std::to_string(r.bytes),std::to_string(getpid())};
        std::vector<char*> argv; for (auto& a : args) argv.push_back(a.data()); argv.push_back(nullptr);
        int pipefd[2]; if (pipe(pipefd)) throw std::runtime_error("RAM lock readiness pipe failed");
        posix_spawn_file_actions_t actions; posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions,pipefd[1],STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions,pipefd[0]); posix_spawn_file_actions_addclose(&actions,pipefd[1]);
        const int result = posix_spawn(&r.helper,helper.c_str(),&actions,nullptr,argv.data(),environ);
        posix_spawn_file_actions_destroy(&actions); close(pipefd[1]);
        if (result) { close(pipefd[0]); r.helper = -1; throw std::runtime_error("RAM lock helper spawn failed"); }
        std::string reply; char b[128];
        for (;;) { const auto got = read(pipefd[0],b,sizeof(b)); if (got > 0) reply.append(b,size_t(got)); else if (!got || errno != EINTR) break; }
        close(pipefd[0]); if (reply != "RAM_LOCK_READY\n") throw std::runtime_error("Engram table RAM locking failed");
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        const uintptr_t first = reinterpret_cast<uintptr_t>(r.address)&~(page-1);
        const uintptr_t last = (reinterpret_cast<uintptr_t>(r.address)+r.bytes+page-1)&~(page-1);
        if (madvise(reinterpret_cast<void*>(first),last-first,MADV_POPULATE_READ)) throw std::runtime_error("cannot populate locked Engram mapping");
        total += r.bytes;
    }
};
ResidentWeights::ResidentWeights() : p_(std::make_unique<Impl>()) {}
void copy_registered_slice(void* destination,const uint8_t* base,size_t offset,size_t bytes,void* stream) {
    if (!stream || offset > SIZE_MAX-bytes) throw std::invalid_argument("invalid registered model transfer");
    auto* dst = static_cast<uint8_t*>(destination);
    const size_t chunk = registration_chunk();
    while (bytes) {
        const size_t n = std::min(bytes,chunk-offset%chunk);
        ck(cudaMemcpyAsync(dst,base+offset,n,cudaMemcpyHostToDevice,static_cast<cudaStream_t>(stream)));
        dst += n; offset += n; bytes -= n;
    }
}
ResidentWeights::~ResidentWeights() = default;
const uint8_t* ResidentWeights::data(const std::string& name) const { return p_->ranges.at(name).address; }
uint64_t ResidentWeights::bytes() const { return p_->total; }
void ResidentWeights::load(const Model& m) {
    if (!p_->ranges.empty()) throw std::runtime_error("resident weights already loaded");
    std::ifstream meminfo("/proc/meminfo"); std::string field; uint64_t kib = 0, available = 0;
    while (meminfo >> field) {
        if (field == "MemAvailable:") { meminfo >> kib; available = kib*1024; break; }
        std::string rest; std::getline(meminfo,rest);
    }
    if (!available || available < m.routed_bytes()+m.engram_bytes()+(8ull<<30))
        throw std::runtime_error("not enough available RAM for the entire model plus 8 GiB reserve");
    ck(cudaSetDevice(0));
    std::fprintf(stderr,"strata-deepseek41: host registration region=%zu MiB\n",registration_chunk()>>20);
    for (int l = 0; l < m.config().layers; ++l) {
        const std::string prefix = "blk."+std::to_string(l)+".";
        for (const char* matrix : {"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) p_->own(prefix+matrix,m.tensor(prefix+matrix));
        std::fprintf(stderr,"strata-deepseek41: resident layer=%d copied/byte-verified/pinned %.3f GiB\n",l,double(p_->total)/(1ull<<30));
    }
    for (uint64_t l : m.config().engram.layers) {
        const std::string name = "blk."+std::to_string(l)+".engram_embd.weight"; p_->lock_table(name,m.tensor(name));
        std::fprintf(stderr,"strata-deepseek41: Engram layer=%llu locked %.3f GiB\n",(unsigned long long)l,double(p_->ranges.at(name).bytes)/(1ull<<30));
    }
    verify();
}
void ResidentWeights::verify() const {
    const size_t page = size_t(sysconf(_SC_PAGESIZE)), chunk = registration_chunk(); std::vector<unsigned char> resident(chunk/page);
    uint64_t pages = 0;
    for (const auto& item : p_->ranges) {
        const auto& r = item.second;
        if (r.helper > 0 && kill(r.helper,0)) throw std::runtime_error("Engram RAM locker exited");
        const uintptr_t first = reinterpret_cast<uintptr_t>(r.address)&~(page-1), last = (reinterpret_cast<uintptr_t>(r.address)+r.bytes+page-1)&~(page-1);
        for (uintptr_t start = first; start < last; start += chunk) {
            const size_t n = std::min(size_t(last-start),chunk);
            if (mincore(reinterpret_cast<void*>(start),n,resident.data())) throw std::runtime_error("RAM residency query failed");
            for (size_t i = 0; i < n/page; ++i) { if (!(resident[i]&1)) throw std::runtime_error("nonresident model page: "+item.first); ++pages; }
        }
    }
    std::fprintf(stderr,"strata-deepseek41: RAM residency PASS %.3f GiB, %llu pages; paging fallback disabled\n",double(p_->total)/(1ull<<30),(unsigned long long)pages);
}
}  // namespace strata::deepseek41
