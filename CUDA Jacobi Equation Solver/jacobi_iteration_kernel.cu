#include "jacobi_iteration.h"
texture<float, 2> A_on_tex;
//texture<float, 2> out_on_tex;

/* FIXME: Write the device kernels to solve the Jacobi iterations. */


__global__ void jacobi_iteration_kernel_naive (const float *A, float* X, float *new_naive_cuda_x, const float *B, unsigned int num_rows, unsigned int num_cols)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	double sum = 0.0;
	int j;

             	for (j = 0; j < num_cols; j++)
		{
			if(row != j)
                 		sum += A[row * num_cols + j] * X[j];
		}
	new_naive_cuda_x[row] = (B[row] - sum)/A[row * num_cols + row];
	return;
}

__global__ void jacobi_iteration_kernel_optimized (float* A, float* X, float *new_naive_cuda_x, unsigned int num_rows, unsigned int num_cols, int iter, double* ssd)
{
	__shared__ double shared_ssd[2048];
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	double sum = 0.0;
	float value;
	int j;

             	for (j = 0; j < num_cols; j++)
		{
			if(row != j)
			{
				value = tex2D (A_on_tex, j, row);
                 		sum += value * X[j];
			}
		}
	value = tex2D (A_on_tex, row, row);
       new_naive_cuda_x[row] = (B_c[row] - sum) / value;
	shared_ssd[row] = (new_naive_cuda_x[row] - X[row]) * (new_naive_cuda_x[row] - X[row]);
	__syncthreads();
	
	int k = (num_rows)/2;
	while(k != 0)
	{
		if (row < k)
	      		shared_ssd[row] += shared_ssd[row+k];
		k /= 2;
	}
	if(row == 0)
		*ssd = shared_ssd[0];
           
	return;
}

