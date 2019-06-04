#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations. */


__global__ void jacobi_iteration_kernel_naive (const float *A, float *new_naive_cuda_x, float *x, const float *B, unsigned int num_rows, unsigned int num_cols)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	double sum;
	int i, j;


        for (i = 0; i < num_rows; i++){
              sum = -A[i * num_cols + i] * x[i];
              for (j = 0; j < num_cols; j++)
                 sum += A[i * num_cols + j] * x[j];
        } 
        new_naive_cuda_x[i] = (B[i] - sum)/A[i * num_cols + i];
//Dr.K doesn't have the above line inside the for loop, not sure if that's intentional.        
           
	return;
}

__global__ void jacobi_iteration_kernel_optimized ()
{
    return;
}

