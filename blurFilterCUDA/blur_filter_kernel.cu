/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{
	int tid = threadIdx.x;
	int pix, i, j, num_neighbors, pixelIndex, curr_row, curr_col;
	float blur_value;
	for(pix = 0; pix < size; pix++)
	{
		pixelIndex = tid * size + pix;
		blur_value = 0.0;
		num_neighbors = 0;
		for(i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++)
		{
			for(j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++)
			{
				curr_row = tid + i;
				curr_col = pix + j;
				if((curr_row > -1) && (curr_row < size) &&\
					(curr_col > -1) && (curr_col < size))
				{
					blur_value += in[curr_row * size + curr_col];
					num_neighbors += 1;
				}
			}
		}
		//printf("%f\n", blur_value/num_neighbors);
		out[pixelIndex] = blur_value/num_neighbors;
		
	}
    	return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
