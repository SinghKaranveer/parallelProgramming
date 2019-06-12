/* FIXME: Edit this file to complete the functionality of 2D separable 
 * convolution on the GPU. You may add additional kernel functions 
 * as necessary. 
 */


texture<float, 2> in_on_tex;
texture<float, 2> out_on_tex;

__global__ void 
convolve_rows_kernel_naive(float *in, float *out, int num_rows, int num_cols, float* kernel, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	int i, i1;
	int j, j1, j2;
	int x, y;
	y = row;
	x = col;
	j1 = x - half_width;
	j2 = x + half_width;
	if(j1 < 0)
		j1 = 0;
	if(j2 >= num_cols)
		j2 = num_cols - 1;
	
	i1 = j1 - x;
	
	j1 = j1 - x + half_width;
	j2 = j2 - x + half_width;
	
	out[y * num_cols + x] = 0.0f;
	for(i = i1, j = j1; j <= j2; j++, i++)
		out[y * num_cols + x] += kernel[j] * in[y * num_cols + x + i];
	//printf("%f\n", out[y * num_cols + x]);
	
	return; 
}

__global__ void 
convolve_columns_kernel_naive(float *in, float *out, int num_rows, int num_cols, float* kernel, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	int i, i1;
	int j, j1, j2;
	int x, y;
	y = row;
	x = col;
	j1 = y - half_width;
	j2 = y + half_width;
	
	if(j1 < 0)
		j1 = 0;
	if(j2 >= num_rows)
		j2 = num_rows - 1;
	
	i1 = j1 - y;
	
	j1 = j1 - y + half_width;
	j2 = j2 - y + half_width;

	out[y * num_cols + x] = 0.0f;
	for (i = i1, j = j1; j <= j2; j++, i++)
		out[y * num_cols + x] += kernel[j] * in[y * num_cols + x + (i * num_cols)];
    return;
}

__global__ void 
convolve_rows_kernel_optimized(float *in, float *out, int num_rows, int num_cols, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	int i, i1;
	int j, j1, j2;
	int x, y;
	y = row;
	x = col;
	j1 = x - half_width;
	j2 = x + half_width;
	int row_number = 32 * blockY + threadY;
	int col_number = 32 * blockX + threadX;
	
	if(j1 < 0)
		j1 = 0;
	if(j2 >= num_cols)
		j2 = num_cols - 1;
	
	i1 = j1 - x;
	
	j1 = j1 - x + half_width;
	j2 = j2 - x + half_width;
	float value;
	out[y * num_cols + x] = 0.0f;
	for(i = i1, j = j1; j <= j2; j++, i++)
	{
		value = tex2D (in_on_tex, col_number + i, row_number);
		out[y * num_cols + x] += kernel_c[j] * value;
	}
	
	return; 
}


__global__ void 
convolve_columns_kernel_optimized(float *in, float *out, int num_rows, int num_cols, int half_width)
{
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;
	int col = blockDim.x * blockX + threadX;
	int i, i1;
	int j, j1, j2;
	int x, y;
	y = row;
	x = col;
	j1 = y - half_width;
	j2 = y + half_width;
	int row_number = 32 * blockY + threadY;
	int col_number = 32 * blockX + threadX;
	float value;
	if(j1 < 0)
		j1 = 0;
	if(j2 >= num_rows)
		j2 = num_rows - 1;
	
	i1 = j1 - y;
	
	j1 = j1 - y + half_width;
	j2 = j2 - y + half_width;

	out[y * num_cols + x] = 0.0f;
	for (i = i1, j = j1; j <= j2; j++, i++)
	{
		value = tex2D (out_on_tex, col_number, row_number + i);
		out[y * num_cols + x] += kernel_c[j] *  value;
	}
    	return;
}




