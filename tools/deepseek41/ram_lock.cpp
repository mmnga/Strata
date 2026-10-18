#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::runtime_error("expected file, offset, bytes and parent PID");
        const auto offset = std::stoull(argv[2]), bytes = std::stoull(argv[3]);
        const auto parent = std::stol(argv[4]);
        if (prctl(PR_SET_PDEATHSIG, SIGTERM) || getppid() != parent) return 2;
        const int fd = open(argv[1], O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error(std::strerror(errno));
        struct stat st;
        if (fstat(fd, &st) || !bytes || offset > uint64_t(st.st_size) || bytes > uint64_t(st.st_size) - offset)
            throw std::runtime_error("invalid model lock range");
        const size_t page = sysconf(_SC_PAGESIZE), first = offset & ~(page - 1);
        const size_t size = (offset - first + bytes + page - 1) & ~(page - 1);
        void * address = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, first);
        close(fd);
        if (address == MAP_FAILED || mlock(address, size)) throw std::runtime_error(std::strerror(errno));
        std::puts("RAM_LOCK_READY");
        std::fflush(stdout);
        close(STDOUT_FILENO);
        for (;;) pause();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "strata model RAM locker: %s\n", e.what());
        return 1;
    }
}
