#include "strata/deepseek41/resident.hpp"
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
int main() {
    constexpr size_t chunk = 1ull<<30, tail = 2ull<<20;
    uint8_t* src = static_cast<uint8_t*>(mmap(nullptr,chunk+tail,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if (src == MAP_FAILED) return 1;
    try {
        ck(cudaHostRegister(src,chunk,cudaHostRegisterPortable)); ck(cudaHostRegister(src+chunk,tail,cudaHostRegisterPortable));
        for (int device = 0; device < 2; ++device) {
            ck(cudaSetDevice(device)); cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            uint8_t* dst; ck(cudaMalloc(&dst,tail));
            for (size_t i = 0; i < tail; ++i) src[chunk-64+i] = uint8_t(i*19+7);
            const auto unsplit = cudaMemcpyAsync(dst,src+chunk-64,128,cudaMemcpyHostToDevice,stream);
            std::cout << "device=" << device << " unsplit crossing registration: " << cudaGetErrorString(unsplit) << '\n';
            cudaGetLastError();
            strata::deepseek41::copy_registered_slice(dst,src,chunk-64,tail,stream);
            ck(cudaStreamSynchronize(stream));
            uint8_t* readback = new uint8_t[tail]; ck(cudaMemcpy(readback,dst,tail,cudaMemcpyDeviceToHost));
            if (std::memcmp(readback,src+chunk-64,tail)) throw std::runtime_error("cross-boundary raw bytes changed");
            delete[] readback; cudaFree(dst); cudaStreamDestroy(stream);
        }
        cudaHostUnregister(src); cudaHostUnregister(src+chunk); munmap(src,chunk+tail);
        std::cout << "Registered model transfer boundary PASS\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; cudaHostUnregister(src); cudaHostUnregister(src+chunk); munmap(src,chunk+tail); return 1;
    }
}
