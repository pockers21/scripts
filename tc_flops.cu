#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <random>
#include <algorithm>
using namespace std;


// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

#define MATRIX_M 4096
#define MATRIX_N 4096
#define MATRIX_K 4096

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__global__ void simple_gemm(const half *A, const half *B, float *C, int M, int N, int K, float alpha, float beta) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is necessary to make things work under tile quantization
    if (x <  M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
        tmp += (__half2float(A[x * K + i]) * __half2float(B[i * N + y]));
        }
        C[x * N + y] = alpha *tmp + beta*C[x * N + y];
  }
}


__global__ void wmma_example(const half *a, const half *b, float *c, int M, int N, int K, float alpha, float beta) {

   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);


   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }


   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}


void generateUniformHalf(half* array, size_t size) {
    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f); // [0.0, 1.0) 均匀分布

    for (size_t i = 0; i < size; ++i) {
    float fvalue = distribution(generator);
    half value = __float2half(fvalue);
        array[i] = value;
    }
}



void run_tc(const half *a_fp16, const half *b_fp16, float *c_wmma, float alpha, float beta){
   printf("Running with wmma...\n");
   //tensor core
   dim3 tc_gridDim;
   dim3 tc_blockDim;

   tc_blockDim.x = 128;
   tc_blockDim.y = 4;

   tc_gridDim.x = (MATRIX_M + (WMMA_M * tc_blockDim.x / 32 - 1)) / (WMMA_M * tc_blockDim.x / 32);
   tc_gridDim.y = (MATRIX_N + WMMA_N * tc_blockDim.y - 1) / (WMMA_N * tc_blockDim.y);
   wmma_example <<< tc_gridDim, tc_blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

}

void run_nontc(const half *a_fp16, const half *b_fp16, float *c_wmma, float alpha, float beta){
   printf("Running without wmma...\n");
   //simple gemm
   dim3 nontc_gridDim;
   dim3 nontc_blockDim;
   nontc_blockDim.x = 128;
   nontc_blockDim.y = 4;

   nontc_gridDim.x = (MATRIX_M + (nontc_blockDim.x-1))/nontc_blockDim.x;
   nontc_gridDim.y = (MATRIX_N + (nontc_blockDim.y-1))/nontc_blockDim.y;
   simple_gemm<<< nontc_gridDim, nontc_blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
}

int main(int argc, char* argv[]) {
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;


   curandGenerator_t gen;
   cublasHandle_t cublasHandle;


   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;


   cudaErrCheck(cudaEventCreate(&startcublas));
   cudaErrCheck(cudaEventCreate(&stopcublas));


   cublasErrCheck(cublasCreate(&cublasHandle));

   // Use tensor cores
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

   cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
   half* h_a_fp16 = (half *)malloc(MATRIX_M * MATRIX_K * sizeof(half));
   half* h_b_fp16 = (half *)malloc(MATRIX_N * MATRIX_K * sizeof(half));

   cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));


   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));


   generateUniformHalf(h_a_fp16, MATRIX_M * MATRIX_K );
   generateUniformHalf(h_b_fp16, MATRIX_K * MATRIX_N );
   printf("print matrix value");


   cudaMemcpy(a_fp16, h_a_fp16, MATRIX_M * MATRIX_K * sizeof(half) , cudaMemcpyHostToDevice);
   cudaMemcpy(b_fp16, h_b_fp16, MATRIX_N * MATRIX_K * sizeof(half) , cudaMemcpyHostToDevice);


   float alpha = 1.0f;
   float beta = 0.0f;

   bool if_run_tc = false;
   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // warm up cuda kernel

   if(if_run_tc){
      run_tc(a_fp16, b_fp16, c_wmma, alpha, beta);
   } else {
      run_nontc(a_fp16, b_fp16, c_wmma, alpha, beta);
   }


   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   // Warm up cuBLAS run starts
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   // Warm up cuBLAS run ends

   // reset the c_cublas buffer
   cudaErrCheck(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

   cudaErrCheck(cudaEventRecord(startcublas));
   cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
   cudaErrCheck(cudaEventRecord(stopcublas));
   cudaErrCheck(cudaEventSynchronize(stopcublas));



   cudaEvent_t beg, end;
   cudaEventCreate(&beg);
   cudaEventCreate(&end);
   float cublasTime;
   cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
   float elapsed_time;
   int repeat_times = 1;
   cudaEventRecord(beg);
   for (int j = 0; j < repeat_times; j++) {
      if(if_run_tc){
         run_tc(a_fp16, b_fp16, c_wmma, alpha, beta);
      } else {
         run_nontc(a_fp16, b_fp16, c_wmma, alpha, beta);
      }
   }
   cudaEventRecord(end);
   cudaEventSynchronize(beg);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&elapsed_time, beg, end);

   printf("1e-3:%f\n",1e-3);
   uint64_t flops = 2 * 1e-9 * MATRIX_M * MATRIX_N * MATRIX_K;
   printf("flops: %d\n",flops);
   elapsed_time = elapsed_time/1000.;
   float GFLOPS = (repeat_times * flops ) / elapsed_time;

   printf("gemm Time:%f \n", elapsed_time);
   printf("gemm GFLOPS:%f\n", GFLOPS);



   cudaEventRecord(beg, 0);
   cudaEventRecord(end, 0);
   cudaEventRecord(beg);
   for (int j = 0; j < repeat_times; j++) {
      cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
   }
   cudaEventRecord(end);
   cudaEventSynchronize(beg);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&elapsed_time, beg, end);
   elapsed_time = elapsed_time/1000;
   printf("cublas Time:%f \n", elapsed_time );
   GFLOPS = (repeat_times * flops ) / elapsed_time;
   printf("cublas GFLOPS:%f\n", GFLOPS);



   cudaErrCheck(cudaEventDestroy(startcublas));
   cudaErrCheck(cudaEventDestroy(stopcublas));

   cudaErrCheck(cudaFree(a_fp16));
   cudaErrCheck(cudaFree(b_fp16));

   cudaErrCheck(cudaFree(c));
   cudaErrCheck(cudaFree(c_cublas));
   cudaErrCheck(cudaFree(c_wmma));

   cudaErrCheck(cudaDeviceReset());
   free(h_a_fp16);
   free(h_b_fp16);
   return 0;
}
