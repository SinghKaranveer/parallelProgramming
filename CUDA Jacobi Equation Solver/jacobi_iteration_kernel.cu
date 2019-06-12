#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations. */

__device__ 
void lock (int *mutex) 
{	  
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}

__device__ 
void unlock (int *mutex) 
{
    atomicExch (mutex, 0);
    return;
}


__global__ void jacobi_iteration_kernel_naive (const float *A, float* X, float *new_naive_cuda_x, const float *B, unsigned int num_rows, unsigned int num_cols, double *ssd, int *mutex)
{
	__shared__ double ssd_per_thread[THREAD_BLOCK_SIZE]; //Should be thread_block_size in here, but we're using 1 thread 
	
	//double ssd = 0.0f;

	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	//ssd = 0;
//	printf("Hello\n");
	double sum = 0.0;
//	printf("%f\n", A[row*num_cols+col]);
	int i, j;

//	printf("\nFor Loop I starts\n");
//        for (i = 0; i < num_rows; i++){
//             double sum = -A[i * num_cols + i] * x[i];
//	printf("\nFor Loop J starts\n");
             	for (j = 0; j < num_cols; j++)
		{
			if(row != j)
                 		sum += A[row * num_cols + j] * X[j];
		}
//        }
//       printf("sum= %f\n", sum);
       new_naive_cuda_x[row] = (B[row] - sum)/A[row * num_cols + row];
	//__syncthreads();
	ssd_per_thread[threadX] = (new_naive_cuda_x[row] - X[row]) * (new_naive_cuda_x[row] - X[row]);
	X[row] = new_naive_cuda_x[row];
	__syncthreads();
	//printf("%d\n", ssd);

	if (threadX == 0) {
		lock (mutex);
		*ssd += ssd_per_thread[0];
		unlock (mutex);
	}

           
	return;
}

__global__ void jacobi_iteration_kernel_optimized ()
{
    return;
}

