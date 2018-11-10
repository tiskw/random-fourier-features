// C source
//
// Author: Tetsuya Ishikawa <tiskw111@gmail.com>
// Date  : Nov 10, 2018
///////////////////////////////////// SOURCE START ///////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float rff_fast_inference(float* x, float* W, float* a, int dim_input, int dim_rff)
{   // {{{

    float  y = 0.0;
    float* t = (float*) malloc(sizeof(float) * dim_rff);
    float* c = (float*) malloc(sizeof(float) * dim_rff);
    float* s = (float*) malloc(sizeof(float) * dim_rff);

    for (int n = 0; n < dim_rff; ++n)
    {
        t[n] = 0.0;
        for (int m = 0; m < dim_input; ++m)
            t[n] += x[m] * W[m * dim_rff + n];
    }

    for (int n = 0; n < dim_rff; ++n) c[n] = cos(t[n]);
    for (int n = 0; n < dim_rff; ++n) s[n] = sin(t[n]);
    for (int n = 0; n < dim_rff; ++n) y += c[n] * a[n];
    for (int n = 0; n < dim_rff; ++n) y += s[n] * a[dim_rff + n];

    free(t);
    free(c);
    free(s);

    return y;

}   // }}}

///////////////////////////////////// SOURCE FINISH //////////////////////////////////
// Ganerated by grasp version 0.0
