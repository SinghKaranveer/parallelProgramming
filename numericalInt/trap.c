/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n, num_threads
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids, with num_threads.
 *
 * Compile: gcc -o trap trap.c -O3 -std=c99 -Wall -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * Author: Naga Kandasamy
 * Date modified: 4/1/2019
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>
#include <time.h>
#include <pthread.h>

typedef struct thread_data_s
{
	int tid;
	float lower_limit;
	float upper_limit;
	int num_trapezoids;
	float base;
	int num_threads;
	double* sum_array;
	float b;
} thread_data_t;

void* integrate(void*);
double compute_using_pthreads (float, float, int, float, int);
double compute_gold (float, float, int, float);

int 
main (int argc, char **argv) 
{
    if (argc < 5) {
        printf ("Usage: trap lower-limit upper-limit num-trapezoids num-threads\n");
        printf ("lower-limit: The lower limit for the integral\n");
        printf ("upper-limit: The upper limit for the integral\n");
        printf ("num-trapezoids: Number of trapeziods used to approximate the area under the curve\n");
        printf ("num-threads: Number of threads to use in the calculation\n");
        exit (EXIT_FAILURE);
    }
	struct timeval start, stop;	


    float a = atoi (argv[1]); /* Lower limit */
	float b = atof (argv[2]); /* Upper limit */
	float n = atof (argv[3]); /* Number of trapezoids */

	float h = (b - a)/(float) n; /* Base of each trapezoid */  
	gettimeofday (&start, NULL);
	double reference = compute_gold (a, b, n, h);
	gettimeofday (&stop, NULL);
    	printf ("Reference solution computed using single-threaded version = %f \n", reference);
	printf ("Execution time = %fs. \n\n", (float)(stop.tv_sec - start.tv_sec +\
        	(stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Write this function to complete the trapezoidal rule using pthreads. */
    int num_threads = atoi (argv[4]); /* Number of threads */
	gettimeofday (&start, NULL);
	double pthread_result = compute_using_pthreads (a, b, n, h, num_threads);
	gettimeofday (&stop, NULL);
	printf ("Solution computed using %d threads = %f \n", num_threads, pthread_result);
	printf ("Execution time = %fs. \n\n", (float)(stop.tv_sec - start.tv_sec +\
        	(stop.tv_usec - start.tv_usec)/(float)1000000));

    exit(EXIT_SUCCESS);
} 


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Defines the integrand
 * Input args:  x
 * Output: sqrt((1 + x^2)/(1 + x^4))

 */
float 
f (float x) 
{
    return sqrt ((1 + x*x)/(1 + x*x*x*x));
}

/*------------------------------------------------------------------
 * Function:    compute_gold
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids using a single-threaded version
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double 
compute_gold (float a, float b, int n, float h) 
{
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;

   for (k = 1; k <= n-1; k++) 
     integral += f(a+k*h);
   
   integral = integral*h;

   return integral;
}  

/* FIXME: Complete this function to perform the trapezoidal rule using pthreads. */
/*
 * a = start
 * b = end
 * n = number of trapazoids
 * h = base of trapazoid
 * num_threads = number of threads
 */
double 
compute_using_pthreads (float a, float b, int n, float h, int num_threads)
{
	int i;
	double integral = 0.0;
	pthread_t *thread_id = (pthread_t *) malloc (num_threads * sizeof (pthread_t));
 	pthread_attr_t attributes;
   	pthread_attr_init (&attributes);
	thread_data_t *thread_data_array = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);
	double *sum_array = (double *) malloc(sizeof(double) * num_threads);
	int chunk_size = (int) ceil(n*1.0 / num_threads); // Chunk size is how many trapezoids each thread will calculate
	for(i = 0; i < num_threads; i++)
	{
		thread_data_array[i].lower_limit = i * chunk_size * h;
		thread_data_array[i].num_trapezoids = chunk_size;
		if(i == num_threads - 1)
		{
			thread_data_array[i].num_trapezoids = n - chunk_size * (num_threads - 1);
			
		}
		thread_data_array[i].base = h;
		thread_data_array[i].tid = i;
		thread_data_array[i].num_threads = num_threads;
		thread_data_array[i].sum_array = sum_array;
		thread_data_array[i].b = b;
	} 
	for(i = 0; i < num_threads; i++)
	{
		pthread_create(&thread_id[i], &attributes, integrate, (void *) &thread_data_array[i]);
	}
	for (i = 0; i < num_threads; i++)
	{
        	pthread_join (thread_id[i], NULL);
	}
	for (i = 0; i < num_threads; i++)
	{
		integral = integral + sum_array[i];
	}
	free((void *) thread_data_array);
	free((void *) sum_array);
	return integral;
}

void* integrate(void *args)
{
	thread_data_t *thread_data = (thread_data_t *) args;
	double integral;
	int i;
	float upper_limit;
	if(thread_data->tid == thread_data->num_threads - 1)
	{
		upper_limit = thread_data->b;
		integral = (f(thread_data->lower_limit) + f(upper_limit))/2.0;
		
		for (i = 1; i < thread_data->num_trapezoids; i++)
     			integral += f(thread_data->lower_limit+i*thread_data->base);
		
		integral = integral*thread_data->base;
	}
	else
	{
		upper_limit = thread_data->num_trapezoids * thread_data->base + thread_data->lower_limit;
		integral = (f(thread_data->lower_limit) + f(upper_limit))/2.0;

		for (i = 1; i < thread_data->num_trapezoids; i++)
     			integral += f(thread_data->lower_limit+i*thread_data->base);

		integral = integral*thread_data->base;
	}
	thread_data->sum_array[thread_data->tid] = integral;
  	pthread_exit (NULL);
	
}
