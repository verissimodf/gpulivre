// BF16 GEMM microbenchmark for validating the gpu_unlock tensor-core rate.
//
// Measures a dense 8192x8192x8192 matrix multiplication with BF16 inputs and
// FP32 accumulation through cuBLAS -- the exact math path NVIDIA caps to half
// rate on GeForce cards and that gpu_unlock restores.
//
// Build: nvcc -O3 -arch=native -o benchmark_bf16 benchmark_bf16.cu -lcublas
// Run:   ./benchmark_bf16

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CK(x)                                                              \
  do {                                                                     \
    cudaError_t err = (x);                                                 \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(err));              \
      return 1;                                                            \
    }                                                                      \
  } while (0)

#define CB(x)                                                              \
  do {                                                                     \
    cublasStatus_t status = (x);                                           \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
      fprintf(stderr, "cuBLAS status %d\n", (int)status);                  \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main(void) {
  const int n = 8192;
  const size_t elems = (size_t)n * n;

  __nv_bfloat16 *a, *b;
  float *c;
  CK(cudaMalloc(&a, elems * sizeof(*a)));
  CK(cudaMalloc(&b, elems * sizeof(*b)));
  CK(cudaMalloc(&c, elems * sizeof(*c)));
  CK(cudaMemset(a, 0x3f, elems * sizeof(*a)));
  CK(cudaMemset(b, 0x3f, elems * sizeof(*b)));

  cublasHandle_t handle;
  CB(cublasCreate(&handle));
  CB(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const float alpha = 1.0f, beta = 0.0f;
  const int warmup = 10, iters = 30;

  for (int i = 0; i < warmup; i++)
    CB(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a,
                    CUDA_R_16BF, n, b, CUDA_R_16BF, n, &beta, c, CUDA_R_32F,
                    n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CK(cudaEventCreate(&start));
  CK(cudaEventCreate(&stop));
  CK(cudaEventRecord(start));
  for (int i = 0; i < iters; i++)
    CB(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a,
                    CUDA_R_16BF, n, b, CUDA_R_16BF, n, &beta, c, CUDA_R_32F,
                    n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CK(cudaEventRecord(stop));
  CK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  double tflops =
      2.0 * (double)n * (double)n * (double)n * iters / ((double)elapsed_ms * 1e9);
  printf("gemm n=%d iters=%d avg=%.2f ms -> %.1f TFLOPS (BF16 in / FP32 acc)\n",
         n, iters, elapsed_ms / iters, tflops);

  cublasDestroy(handle);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}
