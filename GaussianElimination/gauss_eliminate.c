/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date created: February 7
 * Date of last update: April 10, 2019
 *
 * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

typedef struct thread_data_s
{
	int tid;
	int start, end;
	unsigned int chunk_size;
	unsigned int num_elements;
	int num_threads;
	float* U;
	
} thread_data_t;

/* Structure that defines the barrier. */
typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter. */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross. */
    int counter;                /* The value itself. */
} barrier_t;

barrier_t barrier;
barrier_t barrier2;
pthread_mutex_t lock;

/* Function prototypes. */
void* gaussian(void*);
extern int compute_gold (float *, unsigned int);
Matrix allocate_matrix (int, int, int);
void gauss_eliminate_using_pthreads (Matrix, int, unsigned int);
int perform_simple_check (const Matrix);
void print_matrix (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);
void barrier_sync (barrier_t *, int, int);

int
main (int argc, char **argv)
{
    /* Check command line arguments. */
    if (argc > 1) {
        printf ("Error. This program accepts no arguments.\n");
        exit (EXIT_FAILURE);
    }

    Matrix A;			    /* Input matrix. */
    Matrix U_reference;		/* Upper triangular matrix computed by reference code. */
    Matrix U_mt;			/* Upper triangular matrix computed by pthreads. */
    /* Initialize the random number generator with a seed value. */
    srand (time (NULL));

    A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	            /* Allocate and populate a random square matrix. */
    U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	/* Allocate space for the reference result. */
    U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	        /* Allocate space for the multi-threaded result. */

    /* Copy the contents of the A matrix into the U matrices. */
    for (int i = 0; i < A.num_rows; i++) {
        for (int j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    printf ("Performing gaussian elimination using the reference code.\n");
    struct timeval start, stop;
    gettimeofday (&start, NULL);
    
    int status = compute_gold (U_reference.elements, A.num_rows);
    
  
    gettimeofday (&stop, NULL);
    printf ("CPU run time = %0.2f s.\n", (float) (stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec) / (float) 1000000));

    if (status == 0) {
        printf ("Failed to convert given matrix to upper triangular. Try again. Exiting.\n");
        exit (EXIT_FAILURE);
    }
  
    status = perform_simple_check (U_reference);	/* Check that the principal diagonal elements are 1. */ 
    if (status == 0) {
        printf ("The upper triangular matrix is incorrect. Exiting.\n");
        exit (EXIT_FAILURE);
    }
    printf ("Single-threaded Gaussian elimination was successful.\n");
    
    
    /* Perform the Gaussian elimination using pthreads. The resulting upper 
     * triangular matrix should be returned in U_mt */
    gauss_eliminate_using_pthreads (U_mt,4, A.num_rows);

    /* check if the pthread result matches the reference solution within a specified tolerance. */
    int size = MATRIX_SIZE * MATRIX_SIZE;
    int res = check_results (U_reference.elements, U_mt.elements, size, 0.0001f);
    printf ("TEST %s\n", (1 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for the matrices. */
    free (A.elements);
    free (U_reference.elements);
    free (U_mt.elements);

    exit (EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads. */
void
gauss_eliminate_using_pthreads (Matrix U, int num_threads, unsigned int num_elements)
{
	int i;
	pthread_t* thread_id = (pthread_t *) malloc(sizeof(pthread_t) * num_threads); //This allocates the memory needed for the total amount of threads
	thread_data_t *thread_data_array = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);
	pthread_attr_t attributes; //
	pthread_attr_init (&attributes);
	int chunk_size = (int) floor(MATRIX_SIZE / num_threads); //chunk_size is how many rows to do
	barrier.counter = 0;
        sem_init (&barrier.counter_sem, 0, 1); /* Initialize the semaphore protecting the counter to unlocked. */
	sem_init (&barrier.barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to locked. */
	printf("BEFORE\n");
	for(i=0;i < num_elements * num_elements; i++)
	{
		//printf("%i  =  %f\n",i,U.elements[i]);
	}

	printf("Chunk_size=%i\n",chunk_size);
	for(i = 0; i < num_threads; i++)
	{
		thread_data_array[i].tid = i;
		thread_data_array[i].chunk_size = chunk_size;
		thread_data_array[i].num_elements = num_elements;
		thread_data_array[i].U = U.elements;
		thread_data_array[i].start = i * chunk_size;
		thread_data_array[i].end = (i * chunk_size) + chunk_size;
		thread_data_array[i].num_threads = num_threads;
	}
	for(i = 0; i < num_threads; i++)
	{
		pthread_create(&thread_id[i], &attributes, gaussian, (void *) &thread_data_array[i]);
	}
	for(i = 0; i < num_threads; i++)
	{
		pthread_join (thread_id[i], NULL);
	}
	printf("FIRST 512 IN MT\n");
	for(i=0;i < num_elements * num_elements; i++)
	{
		printf("%i  =  %f\n",i,U.elements[i]);
	}
	free((void *) thread_data_array);
	free((void *) thread_id);
}

void* gaussian(void* args)
{
	thread_data_t *thread_data = (thread_data_t *) args;
	//int chunk_size = thread_data->chunk_size;
	int tid = thread_data->tid;
	int num_elements = thread_data->num_elements;
	float* U = thread_data->U;
	int end = thread_data->end;
	int start = thread_data->start;
	int k,j,i;
	//n = chunk_size*tid*num_elements;//tid + 1;
	printf("TID=%i START=%i END=%i\n",tid, start, end);
	for(k = start; k < end; k++)
	{
        	//pthread_mutex_lock(&lock);
		for(j = (k + 1); j < num_elements; j++)
		{
			U[num_elements * k + j] = (float) (U[num_elements * k + j] / U[num_elements * k + k]);
		}
		U[num_elements * k + k] = 1;
        	//barrier_sync (&barrier, tid, thread_data->num_threads); 
	//	for(i = (k+1); i < num_elements; i++)
	//	{	
	//		for(j=k+1; j < num_elements; j++)
	//		{
	//			U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);
	//		//	printf("Thread=%i k=%i, i=%i j=%i\n",tid, k, i, j); 
	//		}
	//		U[num_elements*i+k] = 0;
	//	}
        //	pthread_mutex_unlock(&lock);
	}

        barrier_sync (&barrier, tid, thread_data->num_threads); /* Wait here for all threads to catch up before starting the next iteration. */
	for(k = start; k < num_elements; k++)
	{
        	pthread_mutex_lock(&lock);
		for(i = (k + 1); i < num_elements; i++)
		{
			for (j = (k + 1); j < num_elements; j++)
				U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);
			U[num_elements * i + k] = 0;
		}
        	pthread_mutex_unlock(&lock);
	}
		
	printf("TID=%i is done\n", tid);
	pthread_exit (NULL);
}
/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
    for (int i = 0; i < size; i++)
        if (fabsf (A[i] - B[i]) > tolerance)
            return 0;
    return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
    Matrix M;
    M.num_columns = M.pitch = num_columns;
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


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
    return (float)
        floor ((double)
                (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
    for (unsigned int i = 0; i < M.num_rows; i++)
        if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.0001)
            return 0;
  
    return 1;
}

void 
barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
    sem_wait (&(barrier->counter_sem));

    /* Check if all threads before us, that is num_threads - 1 threads have reached this point. */	  
    if (barrier->counter == (num_threads - 1)) {
        barrier->counter = 0; /* Reset the counter. */
        sem_post (&(barrier->counter_sem)); 
					 
        /* Signal the blocked threads that it is now safe to cross the barrier. */
        printf ("Thread number %d is signalling other threads to proceed\n", tid); 
        for (int i = 0; i < (num_threads - 1); i++)
            sem_post (&(barrier->barrier_sem));
    } 
    else {
        barrier->counter++;
        sem_post (&(barrier->counter_sem));
        sem_wait (&(barrier->barrier_sem)); /* Block on the barrier semaphore. */
    }
}
