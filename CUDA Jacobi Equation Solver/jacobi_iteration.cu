/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: May 13, 2019
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code. */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
//#define DEBUG

//extern __shared__ double ssd = 0;

int 
main (int argc, char** argv) 
{
	if (argc > 1) {
		printf ("This program accepts no arguments\n");
		exit (EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix. */
	matrix_t  B;                    /* N x 1 b matrix. */
	matrix_t reference_x;           /* Reference solution. */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel. */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel. */
	struct timeval start, stop;	
	/* Initialize the random number generator. */
	srand (time (NULL));

	/* Generate diagonally dominant matrix. */ 
	A = create_diagonally_dominant_matrix (MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf ("Error creating matrix\n");
        exit (EXIT_FAILURE);
	}
	
    /* Create the other vectors. */
    B = allocate_matrix_on_host (MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix (A);
	print_matrix (B);
	print_matrix (reference_x);
#endif

    /* Compute the Jacobi solution on the CPU. */
	printf ("Performing Jacobi iteration on the CPU\n");
    	gettimeofday (&start, NULL);
    compute_gold (A, reference_x, B);
    	gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution (A, reference_x, B); /* Display statistics. */
	
	/* Compute the Jacobi solution on the GPU. 
       The solutions are returned in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf ("\nPerforming Jacobi iteration on device. \n");

	compute_on_device (A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    display_jacobi_solution (A, gpu_naive_solution_x, B); /* Display statistics. */
    display_jacobi_solution (A, gpu_opt_solution_x, B); 
    
    free (A.elements); 
	free (B.elements); 
	free (reference_x.elements); 
	free (gpu_naive_solution_x.elements);
    free (gpu_opt_solution_x.elements);
	
    exit (EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform the Jacobi calculation on the GPU. */
void 
compute_on_device (const matrix_t A, matrix_t gpu_naive_sol_x, matrix_t gpu_opt_sol_x, const matrix_t B)
{
    	//unsigned int i, j, k;
	unsigned int i;
	unsigned int num_rows = A.num_rows;
	unsigned int num_cols = A.num_columns;
	struct timeval start, stop;
	float* A_on_device;
	float* B_on_device;
	float* X_on_device;
	double ssd = 0; 
	double mse;
	double* ssd_on_device = NULL;

	matrix_t new_naive_x = allocate_matrix_on_host (MATRIX_SIZE, 1, 0);
	float* new_naive_x_on_device;

	/* Initialize current jacobi solution for the naive solution. */
	for (i = 0; i < num_rows; i++)
        	gpu_naive_sol_x.elements[i] = B.elements[i];

	cudaMalloc ((void**) &A_on_device, num_rows * num_cols * sizeof (float));
	cudaMemcpy (A_on_device, A.elements, num_rows * num_cols * sizeof (float), cudaMemcpyHostToDevice);

	cudaMalloc ((void**) &B_on_device, MATRIX_SIZE * sizeof (float));
	cudaMemcpy (B_on_device, B.elements, MATRIX_SIZE * sizeof (float), cudaMemcpyHostToDevice);
	
	cudaMalloc ((void**) &X_on_device, MATRIX_SIZE * sizeof (float));
	cudaMemcpy (X_on_device, gpu_naive_sol_x.elements, MATRIX_SIZE * sizeof (float), cudaMemcpyHostToDevice);

	cudaMalloc ((void**) &new_naive_x_on_device, MATRIX_SIZE * sizeof (float));

	cudaMalloc ((void**) &ssd_on_device, 1 * sizeof (double));
	cudaMemcpy (ssd_on_device, &ssd, 1 * sizeof (double), cudaMemcpyHostToDevice);

	/* Perform Jacobi iteration. */
	unsigned int done = 0;
	//double mse;
	//ssd = 0.0;
	unsigned int num_iter = 0;
	int tile_size = 1;
	dim3 threads (tile_size, THREAD_BLOCK_SIZE, 1); 
	dim3 grid (1, num_rows);
	
	int *mutex_on_device = NULL;
	cudaMalloc ((void **) &mutex_on_device, sizeof (int));
	cudaMemset (mutex_on_device, 0, sizeof (int));

    	gettimeofday (&start, NULL);
	while (!done){ 
		//Activate Kernel
		jacobi_iteration_kernel_naive <<< grid, threads >>> (A_on_device, X_on_device, new_naive_x_on_device, B_on_device, num_rows, num_cols, ssd_on_device, mutex_on_device);
		cudaDeviceSynchronize();
		cudaMemcpy(new_naive_x.elements, new_naive_x_on_device, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	
		cudaMemcpy(gpu_naive_sol_x.elements, X_on_device, MATRIX_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy (&ssd, ssd_on_device, 1 * sizeof (double), cudaMemcpyDeviceToHost);

		//int i;
		//for(i = 0; i < MATRIX_SIZE; i++)
		//	printf("%f\n", new_naive_x.elements[i]);

		//Check for convergence and update the unknowns.
		//ssd = 0.0;
		//for (i = 0; i < num_rows; i++)
		//{
		//	ssd += (new_naive_x.elements[i] - gpu_naive_sol_x.elements[i]) * (new_naive_x.elements[i] - gpu_naive_sol_x.elements[i]);
		//	gpu_naive_sol_x.elements[i] = new_naive_x.elements[i];
	
		//}
		//copy_matrix_to_device(new_naive_cuda_x, new_naive_x);
		cudaMemcpy (X_on_device, new_naive_x.elements, MATRIX_SIZE * sizeof (float), cudaMemcpyHostToDevice);
		num_iter++;
		printf("%d\n ", ssd);
		mse = sqrt (ssd);
		printf("%d\n ", mse);
		printf ("Iteration: %d. MSE = %f\n", num_iter, mse);
		
		if (mse <= THRESHOLD)
			done = 1;
		
		ssd = 0; // Reset ssd on device
		cudaMemcpy(&ssd, ssd_on_device, sizeof (double), cudaMemcpyDeviceToHost);	

	}
    	gettimeofday (&stop, NULL);
	printf ("Execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));
	//gpu_naive_sol_x.elements = new_naive_x.elements;
	//copy_matrix_from_device(new_naive_x, new_naive_cuda_x);
 	printf ("\nConvergence achieved after %d iterations \n", num_iter);

	//free (new_naive_x.elements);
	cudaFree(new_naive_x_on_device);


	cudaFree (ssd_on_device);
	cudaFree (mutex_on_device);

	return;
}

/* Allocate matrix on the device of same size as M. */
matrix_t 
allocate_matrix_on_device (const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc ((void **) &Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t 
allocate_matrix_on_host (int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *) malloc (size * sizeof (float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to a device. */
void 
copy_matrix_to_device (matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof (float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy (Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host. */
void 
copy_matrix_from_device (matrix_t Mhost, const matrix_t Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof (float);
    cudaMemcpy (Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen. */
void 
print_matrix (const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf ("%f ", M.elements[i * M.num_rows + j]);
        }
		
        printf ("\n");
	} 
	
    printf ("\n");
    return;
}

/* Returns a floating-point value between min and max values. */
float 
get_random_number (int min, int max)
{
    float r = rand ()/(float) RAND_MAX;
	return (float) floor ((double) (min + (max - min + 1) * r));
}

/* Check for errors in kernel execution. */
void 
check_CUDA_error (const char *msg)
{
	cudaError_t err = cudaGetLastError ();
	if ( cudaSuccess != err) {
		printf ("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}	
    return;    
}

/* Checks the reference and GPU results. */
int 
check_results (float *reference, float *gpu_result, int num_elements, float eps)
{
    int check = 1;
    float max_eps = 0.0;
    
    for (int i = 0; i < num_elements; i++) {
        if (fabsf((reference[i] - gpu_result[i])/reference[i]) > eps) {
            check = 0;
			printf("Error at index %d\n",i);
			printf("Element r %f and g %f\n", reference[i] ,gpu_result[i]);
            break;
        }
	}
	
    int maxEle;
    for (int i = 0; i < num_elements; i++) {
        if (fabsf((reference[i] - gpu_result[i])/reference[i]) > max_eps) {
            max_eps = fabsf ((reference[i] - gpu_result[i])/reference[i]);
			maxEle=i;
        }
	}

    printf ("Max epsilon = %f at i = %d value at cpu %f and gpu %f \n", max_eps, maxEle, reference[maxEle], gpu_result[maxEle]); 
    
    return check;
}


/* Function checks if the matrix is diagonally dominant. */
int
check_if_diagonal_dominant (const matrix_t M)
{
	float diag_element;
	float sum;
	for (unsigned int i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (unsigned int j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return 0;
	}

	return 1;
}

/* Create a diagonally dominant matrix. */
matrix_t 
create_diagonally_dominant_matrix (unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *) malloc (size * sizeof (float));

	/* Create a matrix with random numbers between [-.5 and .5]. */
    unsigned int i, j;
	printf ("Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
		// M.elements[i] = ((float)rand ()/(float)RAND_MAX) - 0.5;
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs (M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant. */
	if (!check_if_diagonal_dominant (M)) {
		free (M.elements);
		M.elements = NULL;
	}
	
    return M;
}



