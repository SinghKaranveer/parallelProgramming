/* Code for the Jacbi equation solver. 
 * Author: Naga Kandasamy
 * Date created: April 19, 2019
 * Date modified: April 20, 2019
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm -lpthread
 *
 * If you wish to see debug info add the -D DEBUG option when compiling the code.
 */

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <math.h>
#include "grid.h" 

typedef struct thread_data_s
{
	int tid;
	double* sum_array;
	unsigned int chunk_size;
	int num_threads;
	grid_t* G;
	double diff;
	float old, new; 
	int num_iter;
	int start, end;
} thread_data_t;

typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter. */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross. */
    int counter;                /* The value itself. */
} barrier_t;

extern int compute_gold (grid_t *);
void* jacobi(void*);
int compute_using_pthreads_jacobi (grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid (int, float, float);
grid_t *copy_grid (grid_t *);
void print_grid (grid_t *);
void print_stats (grid_t *);
double grid_mse (grid_t *, grid_t *);
void barrier_sync (barrier_t *, int, int);

int 
main (int argc, char **argv)
{	
	if (argc < 5) {
        printf ("Usage: %s grid-dimension num-threads min-temp max-temp\n", argv[0]);
        printf ("grid-dimension: The dimension of the grid\n");
        printf ("num-threads: Number of threads\n"); 
        printf ("min-temp, max-temp: Heat applied to the north side of the plate is uniformly distributed between min-temp and max-temp\n");
        exit (EXIT_FAILURE);
    }
    
    /* Parse command-line arguments. */
    int dim = atoi (argv[1]);
    int num_threads = atoi (argv[2]);
    float min_temp = atof (argv[3]);
    float max_temp = atof (argv[4]);
    
    /* Generate the grids and populate them with initial conditions. */
 	grid_t *grid_1 = create_grid (dim, min_temp, max_temp);
    /* Grid 2 should have the same initial conditions as Grid 1. */
    grid_t *grid_2 = copy_grid (grid_1); 

	/* Compute the reference solution using the single-threaded version. */
	printf ("\nUsing the single threaded version to solve the grid\n");
	struct timeval start, stop;

    gettimeofday (&start, NULL);
	int num_iter = compute_gold (grid_1);
	gettimeofday (&stop, NULL);

	printf ("CPU run time = %0.8f s.\n", (float) (stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec) / (float) 1000000));
	printf ("Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	printf ("Printing statistics for the interior grid points\n");
    print_stats (grid_1);
#ifdef DEBUG
    print_grid (grid_1);
#endif
	
	/* Use pthreads to solve the equation using the jacobi method. */
	printf ("\nUsing pthreads to solve the grid using the jacobi method\n");

    gettimeofday (&start, NULL);
	num_iter = compute_using_pthreads_jacobi (grid_2, num_threads);
	gettimeofday (&stop, NULL);
	printf ("CPU run time (MT) = %0.8f s.\n", (float) (stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec) / (float) 1000000));

	printf ("Convergence achieved after %d iterations\n", num_iter);			
    printf ("Printing statistics for the interior grid points\n");
	print_stats (grid_2);
#ifdef DEBUG
    print_grid (grid_2);
#endif
    
    /* Compute grid differences. */
    double mse = grid_mse (grid_1, grid_2);
    printf ("MSE between the two grids: %f\n", mse);

	/* Free up the grid data structures. */
	free ((void *) grid_1->element);	
	free ((void *) grid_1); 
	free ((void *) grid_2->element);	
	free ((void *) grid_2);

	exit (EXIT_SUCCESS);
}

/* FIXME: Edit this function to use the jacobi method of solving the equation. The final result should be placed in the grid data structure. */
int 
compute_using_pthreads_jacobi (grid_t *G, int num_threads)
{		
    int i;
	pthread_t* thread_id = (pthread_t *) malloc(sizeof(pthread_t) * num_threads); //This allocates the memory needed for the total amount of threads
	thread_data_t *thread_data_array = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);
	double *sum_array = (double *) malloc(sizeof(double) * num_threads);
	pthread_attr_t attributes; //
	pthread_attr_init (&attributes);
	int done = 0;
	double total_sum;
	int num_elements = G->dim - 2;
	int num_iter = 0;
	float eps = 1e-2; /* Convergence criteria. */ 
	int chunk_size = (int) floor(G->dim / num_threads); //chunk_size is how many rows to do
	
	while(!done)
	{
		total_sum = 0.0;
		
		for(i = 0; i < num_threads; i++)
		{
			thread_data_array[i].num_iter = 0;
			thread_data_array[i].tid = i;
			thread_data_array[i].chunk_size = chunk_size;
			thread_data_array[i].G = G;
			thread_data_array[i].num_threads = num_threads;
			thread_data_array[i].start = i * chunk_size;
			thread_data_array[i].end = (i * chunk_size) + chunk_size;
			thread_data_array[i].sum_array = sum_array;

		} 
		for(i = 0; i < num_threads; i++)
		{
			pthread_create(&thread_id[i], &attributes, jacobi, (void *) &thread_data_array[i]);
		}
		for (i = 0; i < num_threads; i++)
		{
	        	pthread_join (thread_id[i], NULL);
		}
		for(i = 0; i < num_threads; i++)
		{
			total_sum = total_sum + sum_array[i];
		}
		total_sum = total_sum / (num_elements * num_elements);
		if (total_sum < eps) 
    		done = 1;
		else
			num_iter++;
		printf ("Iteration %d. DIFF: %f.\n", num_iter, total_sum);
	}
	free((void *) thread_data_array);
	free((void *) thread_id);
	return num_iter;
}

void* jacobi(void* args)
{
	thread_data_t *thread_data = (thread_data_t *) args;
	int chunk_size = thread_data->chunk_size;
	grid_t* grid = thread_data->G;
	int tid = thread_data->tid;
	int done = 0;
	int i, j, k;
	int num_elements = 0;
	double diff = 0.0;
	thread_data->sum_array[tid] = 0;
	float old, new; 
    float eps = 1e-2; /* Convergence criteria. */ 
	int end = thread_data->end;
	int start = thread_data->start;
	if(tid == 0)
		start = 1;
	if(tid == thread_data->num_threads - 1)
		end = grid->dim - 1;
	//printf("TID=%i START=%i END=%i\n", tid, start, end);

      
	for (i = start; i < end; i++) { 
          	for (j = 1; j < (grid->dim - 1); j++) {	
			old = grid->element[i * grid->dim + j]; /* Store old value of grid point. */
	
			new = 0.25 * (grid->element[(i - 1) * grid->dim + j] +\
                             		grid->element[(i + 1) * grid->dim + j] +\
                             		grid->element[i * grid->dim + (j + 1)] +\
                             		grid->element[i * grid->dim + (j - 1)]);
               		
			grid->element[i * grid->dim + j] = new; /* Update the grid-point value. */
               diff = diff + fabs(new - old); /* Calculate the difference in values. */
               num_elements++;
		}
	}
	
    /* End of an iteration. Check for convergence. */
    //diff = diff/num_elements;
    //printf ("Iteration %d. DIFF: %f.\n", num_iter, diff);
 
    //if (diff < eps) 
    //    done = 1;
	thread_data->sum_array[tid] = diff;
	pthread_exit (NULL);
}

	
/* Create a grid with the specified initial conditions. */
grid_t * 
create_grid (int dim, float min, float max)
{
    grid_t *grid = (grid_t *) malloc (sizeof (grid_t));
    if (grid == NULL)
        return NULL;

    grid->dim = dim;
	printf("Creating a grid of dimension %d x %d\n", grid->dim, grid->dim);
	grid->element = (float *) malloc (sizeof (float) * grid->dim * grid->dim);
    if (grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < grid->dim; i++) {
		for (j = 0; j < grid->dim; j++) {
            grid->element[i * grid->dim + j] = 0.0; 			
		}
    }

    /* Initialize the north side, that is row 0, with temperature values. */ 
    srand ((unsigned) time (NULL));
	float val;		
    for (j = 1; j < (grid->dim - 1); j++) {
        val =  min + (max - min) * rand ()/(float)RAND_MAX;
        grid->element[j] = val; 	
    }

    return grid;
}

/* Creates a new grid and copies over the contents of an existing grid into it. */
grid_t *
copy_grid (grid_t *grid) 
{
    grid_t *new_grid = (grid_t *) malloc (sizeof (grid_t));
    if (new_grid == NULL)
        return NULL;

    new_grid->dim = grid->dim;
	new_grid->element = (float *) malloc (sizeof (float) * new_grid->dim * new_grid->dim);
    if (new_grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < new_grid->dim; i++) {
		for (j = 0; j < new_grid->dim; j++) {
            new_grid->element[i * new_grid->dim + j] = grid->element[i * new_grid->dim + j] ; 			
		}
    }

    return new_grid;
}

/* This function prints the grid on the screen. */
void 
print_grid (grid_t *grid)
{
    int i, j;
    for (i = 0; i < grid->dim; i++) {
        for (j = 0; j < grid->dim; j++) {
            printf ("%f\t", grid->element[i * grid->dim + j]);
        }
        printf ("\n");
    }
    printf ("\n");
}


/* Print out statistics for the converged values of the interior grid points, including min, max, and average. */
void 
print_stats (grid_t *grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0;
    int num_elem = 0;
    int i, j;

    for (i = 1; i < (grid->dim - 1); i++) {
        for (j = 1; j < (grid->dim - 1); j++) {
            sum += grid->element[i * grid->dim + j];

            if (grid->element[i * grid->dim + j] > max) 
                max = grid->element[i * grid->dim + j];

             if(grid->element[i * grid->dim + j] < min) 
                min = grid->element[i * grid->dim + j];
             
             num_elem++;
        }
    }
                    
    printf("AVG: %f\n", sum/num_elem);
	printf("MIN: %f\n", min);
	printf("MAX: %f\n", max);
	printf("\n");
}

/* Calculate the mean squared error between elements of two grids. */
double
grid_mse (grid_t *grid_1, grid_t *grid_2)
{
    double mse = 0.0;
    int num_elem = grid_1->dim * grid_1->dim;
    int i;

    for (i = 0; i < num_elem; i++) 
        mse += (grid_1->element[i] - grid_2->element[i]) * (grid_1->element[i] - grid_2->element[i]);
                   
    return mse/num_elem; 
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
        //printf ("Thread number %d is signalling other threads to proceed\n", tid); 
        for (int i = 0; i < (num_threads - 1); i++)
            sem_post (&(barrier->barrier_sem));
    } 
    else {
        barrier->counter++;
        sem_post (&(barrier->counter_sem));
        sem_wait (&(barrier->barrier_sem)); /* Block on the barrier semaphore. */
    }
}
