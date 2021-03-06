/* Reference code for the box blur filter.
 *
 * Author: Naga Kandasamy
 * Date created: May 3, 2019 
 */

#include <stdlib.h>
#include "blur_filter.h"

extern "C" void compute_gold (const image_t, image_t);

void 
compute_gold (const image_t in, image_t out)
{
    int pix, i, j;
    int size;
    int row, col;
    int curr_row, curr_col;
    float blur_value;
    int num_neighbors;

    size = in.size;
    for (pix = 0; pix < size * size; pix++) { /* Iterate over pixels in image. */
        row = pix/size;             /* Obtain the row number of the pixel. */
        col = pix % size;           /* Obtain the column number of the pixel. */

        /* Apply the blur filter to the current pixel. */
        blur_value = 0.0;
        num_neighbors = 0;
        for (i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
            for (j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
                /* Accumulate the values of the neighbors while checking for 
                 * boundary conditions. */
                curr_row = row + i;
                curr_col = col + j;
                if ((curr_row > -1) && (curr_row < size) &&\
                        (curr_col > -1) && (curr_col < size)) {
                    blur_value += in.element[curr_row * size + curr_col];
                    num_neighbors += 1;
                }
            }
        }

        /* Write the averaged blurred value out. */
        out.element[pix] = blur_value/num_neighbors;
    }
    
    return;
}

