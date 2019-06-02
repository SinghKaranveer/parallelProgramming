#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations. */


__global__ void jacobi_iteration_kernel_naive (const matrix_t A, matrix_t new_naive_cuda_x, matrix_t x, const matrix_t B)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	double sum;
	int i, j;
	
	unsigned int num_rows = A.num_rows;
	unsigned int num_cols = A.num_columns;


        for (i = row; i < num_rows; i++){
              sum = -A.elements[i * num_cols + i] * x.elements[i];
              for (j = col; j < num_cols; j++)
                 sum += A.elements[i * num_cols + j] * x.elements[j];
        
        
        new_naive_cuda_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];
//Dr.K doesn't have the above line inside the for loop, not sure if that's intentional.        
	}

	return;
}

__global__ void jacobi_iteration_kernel_optimized ()
{
    return;
}

