#include<mma.h>
#include<stdio.h>
using namespace nvcuda;

#define CUDA_CHECK_RETURN(X) X

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_kernel(half* a, half* b, float* c){

   wmma::fragment<wmma::matrix_a,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> frag_a;
   wmma::fragment<wmma::matrix_b,WMMA_M,WMMA_N,WMMA_K,half,wmma::row_major> frag_b;
   wmma::fragment<wmma::accumulator,WMMA_M,WMMA_N,WMMA_K,float> frag_c;

   wmma::fill_fragment(frag_c,0.0f);
   wmma::fill_fragment(frag_a,1.0f);
   wmma::fill_fragment(frag_b,1.0f);

   //wmma::load_matrix_sync(frag_a,a,WMMA_M);
   //wmma::load_matrix_sync(frag_b,b,WMMA_K);

   for(int i=0; i<1000000000; i++)
    wmma::mma_sync(frag_c,frag_a,frag_b,frag_c);

   wmma::store_matrix_sync(c,frag_c,WMMA_M,wmma::mem_row_major);
}

int main(){

 half *d_a, *h_a, *d_b, *h_b;
 float *d_c, *h_c;
 h_c = new float[16*16];
 h_b = new half[16*16];
 h_a = new half[16*16];
 cudaMalloc(&d_a, 16*16*sizeof(half));
 cudaMalloc(&d_b, 16*16*sizeof(half));
 cudaMalloc(&d_c, 16*16*sizeof(float));
 for (int i = 0; i < 16*16; i++) {
   h_a[i] = 1.0f;
   h_b[i] = 1.0f;}
 cudaMemcpy(d_a, h_a, 16*16*sizeof(half), cudaMemcpyHostToDevice);
 cudaMemcpy(d_b, h_b, 16*16*sizeof(half), cudaMemcpyHostToDevice);
 
 cudaEvent_t start, stop;
 CUDA_CHECK_RETURN(cudaEventCreate(&start));
 CUDA_CHECK_RETURN(cudaEventCreate(&stop));

 CUDA_CHECK_RETURN(cudaEventRecord(start));
 wmma_kernel<<<1,1024>>>(d_a, d_b, d_c);
 CUDA_CHECK_RETURN(cudaEventRecord(stop));

 cudaMemcpy(h_c, d_c, 16*16*sizeof(float), cudaMemcpyDeviceToHost);

 float elapsedTime;
 cudaEventElapsedTime(&elapsedTime, start, stop);

 printf("Elapsed Time : %f\n",elapsedTime);

 return 0;
}
