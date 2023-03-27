#include<mma.h>
#include<cuda/pipeline>
#include<cooperative_groups.h>
#include<stdio.h>
using namespace nvcuda;

#define CUDA_CHECK_RETURN(X) X
#define NUM_ITERS 10

// Disables `pipeline_shared_state` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

// Define some error checking macros.
#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int M = 327660;
const int N = 1536;
const int K = 512;

const int num_threads = 16;
const int smem = K * 16;

__global__ void wmma_kernel(half* a, half* b, float* c){

   __shared__ half SMEM[2*smem];
   int warp_id = threadIdx.x/32;
   int work_per_warp = N/(WMMA_N*32);

   wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> frag_a;
   wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> frag_b;
   wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> frag_c;

   auto group = cooperative_groups::this_thread_block();
   constexpr auto scope = cuda::thread_scope_block;
   constexpr auto stages_count = 2;
   __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
   auto pipeline = cuda::make_pipeline(group, &shared_state);

   pipeline.producer_acquire();
   cuda::memcpy_async(group,&SMEM[0],&a[0],sizeof(half)*K*16,pipeline);
   pipeline.producer_commit();

   for(int it=0; it<NUM_ITERS; it++){
    for(int m=1; m<(M/16); m++){

     pipeline.producer_acquire();
     cuda::memcpy_async(group,&SMEM[(m%2)?smem:0],&a[m*K*16],sizeof(half)*K*16,pipeline);
     pipeline.producer_commit();

     pipeline.consumer_wait();
     for(int i=0 ; i<1; i++){
      for(int j=0; j<work_per_warp; j++){
       wmma::fill_fragment(frag_c,0.0f);
       for(int k=0; k<(K/WMMA_K); k++){
        wmma::load_matrix_sync(frag_a,&SMEM[((i*K*WMMA_M) + (k*WMMA_K)) + (m%2)?0:smem],K);
        wmma::load_matrix_sync(frag_b,&b[(j*WMMA_N) + work_per_warp*warp_id*WMMA_N + (k*WMMA_K*N)],N);
  
        wmma::mma_sync(frag_c,frag_a,frag_b,frag_c);
       }
      wmma::store_matrix_sync(&c[(i*WMMA_M*N)+(m*16*N)+((j+(warp_id*work_per_warp))*WMMA_N)],frag_c,N,wmma::mem_row_major);
      }
     }
     pipeline.consumer_release();
    }
   }
}

int main(){

 half *d_a, *h_a, *d_b, *h_b;
 float *d_c, *h_c;
 h_c = new float[M*N];
 h_b = new half[K*N];
 h_a = new half[M*K];
 cudaMalloc(&d_a, M*K*sizeof(half));
 cudaMalloc(&d_b, K*N*sizeof(half));
 cudaMalloc(&d_c, M*N*sizeof(float));
 for (int i = 0; i < M*K; i++)
   h_a[i] = 1.0f;
 for (int i = 0; i < N*K; i++)
   h_b[i] = 1.0f;
 cudaMemcpy(d_a, h_a, M*K*sizeof(half), cudaMemcpyHostToDevice);
 cudaMemcpy(d_b, h_b, K*N*sizeof(half), cudaMemcpyHostToDevice);
 
 cudaEvent_t start, stop;
 CUDA_CHECK_RETURN(cudaEventCreate(&start));
 CUDA_CHECK_RETURN(cudaEventCreate(&stop));

 CUDA_CHECK_RETURN(cudaEventRecord(start));
 wmma_kernel<<<1,num_threads>>>(d_a, d_b, d_c);
 cudaErrCheck(cudaGetLastError());
 CUDA_CHECK_RETURN(cudaEventRecord(stop));

 cudaMemcpy(h_c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

 float elapsedTime;
 cudaEventElapsedTime(&elapsedTime, start, stop);

 printf("Elapsed Time : %f\n",elapsedTime);

 return 0;
}
