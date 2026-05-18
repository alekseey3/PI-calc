#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

struct CpuResult {
    double pi;
    double time_ms;
};

struct GpuResult {
    double pi;
    float time_ms;
};

__global__ void setupCurandKernel(curandState_t* states,
                                  unsigned long long seed,
                                  int total_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_threads) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

__global__ void piMonteCarloKernel(curandState_t* states,
                                   unsigned long long* block_counts,
                                   unsigned long long N) {
    extern __shared__ unsigned int shared_counts[];

    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    curandState_t local_state = states[global_tid];
    unsigned int local_count = 0;

    for (unsigned long long i = static_cast<unsigned long long>(global_tid); i < N; i += stride) {
        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);
        if (x * x + y * y < 1.0f) {
            local_count++;
        }
    }

    states[global_tid] = local_state;
    shared_counts[local_tid] = local_count;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_tid < s) {
            shared_counts[local_tid] += shared_counts[local_tid + s];
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        block_counts[blockIdx.x] = static_cast<unsigned long long>(shared_counts[0]);
    }
}

__global__ void reduceKernel(const unsigned long long* input,
                             unsigned long long* output,
                             int n) {
    extern __shared__ unsigned long long sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    unsigned long long sum = 0ULL;
    if (i < static_cast<unsigned int>(n)) {
        sum += input[i];
    }
    if (i + blockDim.x < static_cast<unsigned int>(n)) {
        sum += input[i + blockDim.x];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

CpuResult computePiCpu(unsigned long long N, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();

    unsigned long long inside = 0ULL;
    for (unsigned long long i = 0; i < N; ++i) {
        double x = dist(rng);
        double y = dist(rng);
        if (x * x + y * y < 1.0) {
            inside++;
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(finish - start).count();
    double pi = 4.0 * static_cast<double>(inside) / static_cast<double>(N);

    return {pi, time_ms};
}

GpuResult computePiGpu(unsigned long long N, unsigned long long seed) {
    const int block_size = 256;
    const int max_blocks = 1024;

    int blocks = static_cast<int>((N + block_size - 1ULL) / block_size);
    if (blocks < 1) blocks = 1;
    if (blocks > max_blocks) blocks = max_blocks;

    int total_threads = blocks * block_size;

    curandState_t* d_states = nullptr;
    unsigned long long* d_block_counts = nullptr;
    unsigned long long* d_reduce_temp = nullptr;

    CHECK_CUDA(cudaMalloc(&d_states, static_cast<size_t>(total_threads) * sizeof(curandState_t)));
    CHECK_CUDA(cudaMalloc(&d_block_counts, static_cast<size_t>(blocks) * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_reduce_temp, static_cast<size_t>(blocks) * sizeof(unsigned long long)));

    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));

    CHECK_CUDA(cudaEventRecord(start_event));

    setupCurandKernel<<<blocks, block_size>>>(d_states, seed, total_threads);
    CHECK_CUDA(cudaGetLastError());

    piMonteCarloKernel<<<blocks, block_size, block_size * sizeof(unsigned int)>>>(
        d_states, d_block_counts, N
    );
    CHECK_CUDA(cudaGetLastError());

    int current_n = blocks;
    unsigned long long* d_in = d_block_counts;
    unsigned long long* d_out = d_reduce_temp;

    while (current_n > 1) {
        int reduce_blocks = (current_n + block_size * 2 - 1) / (block_size * 2);
        reduceKernel<<<reduce_blocks, block_size, block_size * sizeof(unsigned long long)>>>(
            d_in, d_out, current_n
        );
        CHECK_CUDA(cudaGetLastError());

        current_n = reduce_blocks;
        unsigned long long* tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    CHECK_CUDA(cudaEventRecord(stop_event));
    CHECK_CUDA(cudaEventSynchronize(stop_event));

    float time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start_event, stop_event));

    unsigned long long inside = 0ULL;
    CHECK_CUDA(cudaMemcpy(&inside, d_in, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    double pi = 4.0 * static_cast<double>(inside) / static_cast<double>(N);

    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaFree(d_states));
    CHECK_CUDA(cudaFree(d_block_counts));
    CHECK_CUDA(cudaFree(d_reduce_temp));

    return {pi, time_ms};
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N> [--csv] [--seed <seed>]" << std::endl;
        return EXIT_FAILURE;
    }

    unsigned long long N = std::stoull(argv[1]);
    unsigned long long seed = 12345ULL;
    bool csv = false;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            csv = true;
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        }
    }

    CpuResult cpu = computePiCpu(N, seed);
    GpuResult gpu = computePiGpu(N, seed);

    const double pi_ref = 3.14159265358979323846;
    double cpu_error = std::abs(cpu.pi - pi_ref);
    double gpu_error = std::abs(gpu.pi - pi_ref);
    double speedup = cpu.time_ms / static_cast<double>(gpu.time_ms);

    if (csv) {
        std::cout << std::setprecision(12)
                  << N << ","
                  << cpu.pi << ","
                  << gpu.pi << ","
                  << cpu.time_ms << ","
                  << gpu.time_ms << ","
                  << speedup << ","
                  << cpu_error << ","
                  << gpu_error << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "N = " << N << std::endl;
    std::cout << "CPU pi: " << cpu.pi << std::endl;
    std::cout << "GPU pi: " << gpu.pi << std::endl;
    std::cout << std::setprecision(6);
    std::cout << "CPU time, ms: " << cpu.time_ms << std::endl;
    std::cout << "GPU time, ms: " << gpu.time_ms << std::endl;
    std::cout << "Speedup: " << speedup << std::endl;
    std::cout << "CPU error: " << cpu_error << std::endl;
    std::cout << "GPU error: " << gpu_error << std::endl;

    return EXIT_SUCCESS;
}
