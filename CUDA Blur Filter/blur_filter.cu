/* Reference code implementing the box blur filter.

  Build and execute as follows: 
    make clean && make 
    ./blur_filter size

  Author: Naga Kandasamy
  Date created: May 3, 2019
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* #define DEBUG */

/* Include the kernel code. */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold (const image_t, image_t);
void compute_on_device (const image_t, image_t);
int check_results (const float *, const float *, int, float);
void print_image (const image_t);

int 
main (int argc, char **argv)
{
    if (argc < 2) {
        printf ("Usage: %s size\n", argv[0]);
        printf ("size: Height of the image. The program assumes size x size image.\n");
        exit (EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images. */
    int size = atoi (argv[1]);

    printf ("Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *) malloc (sizeof (float) * size * size);
    out_gold.element = (float *) malloc (sizeof (float) * size * size);
    out_gpu.element = (float *) malloc (sizeof (float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand (time (NULL));
    for (int i = 0; i < size * size; i++)
        in.element[i] = rand ()/ (float) RAND_MAX -  0.5;
        // in.element[i] = 1;
  
   /* Calculate the blur on the CPU. The result is stored in out_gold. */
    printf ("Calculating blur on the CPU\n");
   compute_gold (in, out_gold); 
#ifdef DEBUG 
   print_image (in);
   print_image (out_gold);
#endif

   /* Calculate the blur on the GPU. The result is stored in out_gpu. */
   printf ("Calculating blur on the GPU\n");
   compute_on_device (in, out_gpu);

   /* Check the CPU and GPU results for correctness. */
   printf ("Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = 1e-6;
   int check = check_results (out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 1) 
       printf ("TEST PASSED\n");
   else
       printf ("TEST FAILED\n");
   
   /* Free data structures on the host. */
   free ((void *) in.element);
   free ((void *) out_gold.element);
   free ((void *) out_gpu.element);

    exit (EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU. */
void 
compute_on_device (const image_t in, image_t out)
{
    return;
}

/* Function to check correctness of the results. */
int 
check_results (const float *pix1, const float *pix2, int num_elements, float eps) 
{
    for (int i = 0; i < num_elements; i++)
        if (fabsf ((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return 0;
    
    return 1;
}

/* Function to print out the image contents. */
void 
print_image (const image_t img)
{
    for (int i = 0; i < img.size; i++) {
        for (int j = 0; j < img.size; j++) {
            float val = img.element[i * img.size + j];
            printf ("%0.4f ", val);
        }
        printf ("\n");
    }

    printf ("\n");
    return;
}
