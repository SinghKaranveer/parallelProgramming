/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	int i, j, num_neighbors, pixelIndex, curr_row, curr_col;
	float blur_value;
	pixelIndex = row * size + col;
	blur_value = 0.0;
	num_neighbors = 0;
	for(i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++)
	{
		for(j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++)
		{
			curr_row = row + i;
			curr_col = col + j;
			if((curr_row > -1) && (curr_row < size) &&\
				(curr_col > -1) && (curr_col < size))
			{
				blur_value += in[curr_row * size + curr_col];
				num_neighbors += 1;
			}
		}
	}
	out[pixelIndex] = blur_value/num_neighbors;
		
    	return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
