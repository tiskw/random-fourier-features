// C source
//
// Author: Tetsuya Ishikawa <tiskw111@gmail.com>
// Date  : Nov 10, 2018
///////////////////////////////////// SOURCE START ///////////////////////////////////

#include <stdlib.h>
#include "cosine_table.h"

static float t[4096];
static float c[4096];
static float s[4096];

float rff_fast_inference(float* x, float* W, float* a, int dim_input, int dim_rff)
{   // {{{

    float y = 0.0;
    int index;

    for (int n = 0; n < dim_rff; ++n)
    {
        t[n] = 0.0;
        for (int m = 0; m < dim_input; ++m)
            t[n] += x[m] * W[m * dim_rff + n];
    }

    for (int n = 0; n < dim_rff; ++n)
    {
        index = (int) (t[n] * DELTA_THETA_INV);
        if      (index < 1024) { c[n] = + COSINE[index];        s[n] = + COSINE[1023 - index]; }
        else if (index < 2048) { c[n] = + COSINE[2047 - index]; s[n] = + COSINE[index - 1024]; }
        else if (index < 3072) { c[n] = - COSINE[index - 2048]; s[n] = - COSINE[3071 - index]; }
        else if (index < 4096) { c[n] = - COSINE[4095 - index]; s[n] = - COSINE[index - 3072]; }
    }

    for (int n = 0; n < dim_rff; ++n) y += c[n] * a[n];
    for (int n = 0; n < dim_rff; ++n) y += s[n] * a[dim_rff + n];

    return y;

}   // }}}

///////////////////////////////////// SOURCE FINISH //////////////////////////////////
// Ganerated by grasp version 0.0
