
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status, const char* expression,
                       const char* file, int line)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n",
                     file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

static void check_cublas(cublasStatus_t status, const char* expression,
                         const char* file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d (%s): status %d\n",
                     file, line, expression, (int)status);
        std::exit(EXIT_FAILURE);
    }
}

#define CUBLAS_CHECK(expression) \
    check_cublas((expression), #expression, __FILE__, __LINE__)

////////////////////////////////////////////////////////////////////////////////
// Custom batched GEMM
//
// C[b] = A[b] * B[b]
//
// Matrices are N x N, column-major.
////////////////////////////////////////////////////////////////////////////////
__global__ void batch_gemm_kernel(int n, int batch_size,
                                    const float* A,
                                    const float* B,
                                    float* C)
{
    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;

    // One thread computes one C(row,col). Flattening is column-major.
    for (size_t index = first; index < total; index += stride) {
        const size_t batch = index / matrix_size;
        const size_t local = index - batch * matrix_size;
        const int row = (int)(local % n);
        const int col = (int)(local / n);
        const size_t base = batch * matrix_size;

        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[base + (size_t)k * n + row] *
                   B[base + (size_t)col * n + k];
        }
        C[index] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation
void batched_gemm_cpu(int n, int batch_size,
                      const float* A,
                      const float* B,
                      float* C)
{
    const size_t matrix_size = (size_t)n * n;

    for (int b = 0; b < batch_size; ++b) {

        const size_t base = (size_t)b * matrix_size;

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {

                float sum = 0.0f;

                for (int k = 0; k < n; ++k) {
                    sum += A[base + (size_t)k * n + row] *
                           B[base + (size_t)col * n + k];
                }

                C[base + (size_t)col * n + row] = sum;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Forward error for one matrix
//
// ||Cgpu-Cref||_F / (N ||Cref||_F)
// ------------------------------------------------------------
double forward_error(int n,
                     const float* result,
                     const float* reference)
{
    double diff2 = 0.0;
    double ref2  = 0.0;

    const size_t size = (size_t)n * n;

    for (size_t i = 0; i < size; ++i) {
        const double diff =
            (double)result[i] - (double)reference[i];

        diff2 += diff * diff;
        ref2  += (double)reference[i] * reference[i];
    }

    const double diff_norm = std::sqrt(diff2);
    const double ref_norm  = std::sqrt(ref2);

    if (ref_norm > 0.0)
        return diff_norm / (n * ref_norm);

    return (diff_norm == 0.0) ? 0.0 : INFINITY;
}

////////////////////////////////////////////////////////////////////////////////
// Maximum forward error over the batch
double max_batch_forward_error(int n,
                               int batch_size,
                               const float* result,
                               const float* reference)
{
    const size_t matrix_size = (size_t)n * n;

    double max_error = 0.0;

    for (int b = 0; b < batch_size; ++b) {

        const size_t base = (size_t)b * matrix_size;

        const double e =
            forward_error(n,
                          result + base,
                          reference + base);

        max_error = std::max(max_error, e);
    }

    return max_error;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int n          = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int repeats = 100;

    if (n <= 0 || n > 32 || batch_size <= 0) {

        std::fprintf(stderr, "Usage: %s matrix_size(1..32) batch_size\n", argv[0]);

        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t bytes = total * sizeof(float);

    // --------------------------------------------------------
    // Initialize input matrices
    // --------------------------------------------------------
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_ref(total);
    std::vector<float> h_custom(total);
    std::vector<float> h_cublas(total);

    for (size_t i = 0; i < total; ++i) {
        h_A[i] = std::sin(0.013f * (float)(i + 1));
        h_B[i] = std::cos(0.017f * (float)(i + 1));
    }

    // ref. CPU GEMM
    batched_gemm_cpu(n, batch_size, h_A.data(), h_B.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    // --------------------------------------------------------
    // Copy A and B to GPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // --------------------------------------------------------
    // GPU information
    // --------------------------------------------------------
    int device = 0;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("GPU: %s (compute capability %d.%d)\n",
                prop.name, prop.major, prop.minor);

    std::printf("Matrix size: %d x %d, batch size: %d\n",
                n, n, batch_size);

    std::printf("FP32, column-major matrices, repeats: %d\n\n",
                repeats);

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // Custom kernel configuration
    // --------------------------------------------------------
    const int threads_per_block = 256;
    const size_t required_blocks =
        (total + (size_t)threads_per_block - 1) / threads_per_block;
    const size_t grid_limit = (size_t)prop.maxGridSize[0];
    const unsigned int blocks = (unsigned int)std::min(required_blocks, grid_limit);
    dim3 block(threads_per_block, 1, 1);
    dim3 grid(blocks, 1, 1);

    // --------------------------------------------------------
    // Warmup run
    // --------------------------------------------------------
    batch_gemm_kernel<<<grid, block>>>(n, batch_size, d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < repeats; ++r) {
        batch_gemm_kernel<<<grid, block>>>(n, batch_size, d_A, d_B, d_C);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&custom_ms, start, stop));

    const double custom_us = custom_ms * 1000.0 / repeats;


    CUDA_CHECK(cudaMemcpy(
        h_custom.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // cuBLAS strided batched GEMM
    // --------------------------------------------------------

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const long long int stride = (long long int)matrix_size;

    // Each batch member occupies n*n contiguous elements.
    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_A, n, stride,
        d_B, n, stride,
        &beta,
        d_C, n, stride,
        batch_size));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n,
            &alpha,
            d_A, n, stride,
            d_B, n, stride,
            &beta,
            d_C, n, stride,
            batch_size));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, start, stop));

    const double cublas_us = cublas_ms * 1000.0 / repeats;

    CUDA_CHECK(cudaMemcpy(
        h_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // --------------------------------------------------------
    // Performance and forward error
    // --------------------------------------------------------
    const double flops = 2.0 * n * n * n * batch_size;

    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = max_batch_forward_error(
                                    n, batch_size,
                                    h_custom.data(), h_ref.data());

    const double cublas_error = max_batch_forward_error(
                                    n, batch_size,
                                    h_cublas.data(), h_ref.data());

    std::printf("%-24s %12s %12s %20s\n",
                "Implementation",
                "Time (us)",
                "GFLOP/s",
                "Max batch fwd error");

    std::printf("-----------------------------------------------------------------------\n");

    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "Custom kernel",
                custom_us,
                custom_gflops,
                custom_error);

    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "cuBLAS Strided Batched",
                cublas_us,
                cublas_gflops,
                cublas_error);

    const double tol = 1e-6;
    const bool passed = custom_error <= tol && cublas_error <= tol;
    if (passed) {
        std::printf("\nTEST PASS\n");
        std::printf("Speedup (cuBLAS time / custom time): %.3fx\n", cublas_us / custom_us);
    }
    else {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
