#include <stdio.h>

#define DIM_I  4
#define DIM_J  4
#define DIM_K  4
#define UNROLL 2 

float A[DIM_I][DIM_J];
float B[DIM_J][DIM_K];
float C[DIM_I][DIM_K];
float output[DIM_I][DIM_K];

void kernel(float A[][DIM_J], float B[][DIM_K], float C[][DIM_K], float alpha, float beta, float output[][DIM_K]);

int main()
{
    float alpha = 1.0, beta = -0.5;
    kernel(A, B, C, alpha, beta, output);
    return 0;
}

void kernel(float A[][DIM_J], float B[][DIM_K], float C[][DIM_K], float alpha, float beta, float output[][DIM_K]){
    int i, j, k;

    // Initialization
    // for(i = 0; i < DIM_I; i++){
    //     for(k = 0; k < DIM_K; k++){
    //         output[i][k] = beta * C[i][k];
    //     }
    // }

    // Matrix multiplication with unrolled j loop
    for(i = 0; i < DIM_I; i++){
        for(k = 0; k < DIM_K; k++){
            // Unrolled loop
            for(j = 0; j + UNROLL - 1 < DIM_J; j += UNROLL){
                // Unroll body
                #if UNROLL > 0
                output[i][k] += alpha * A[i][j] * B[j][k];
                #endif
                #if UNROLL > 1
                output[i][k] += alpha * A[i][j+1] * B[j+1][k];
                #endif
                #if UNROLL > 2
                output[i][k] += alpha * A[i][j+2] * B[j+2][k];
                #endif
                #if UNROLL > 3
                output[i][k] += alpha * A[i][j+3] * B[j+3][k];
                #endif
                
            }
            // Handle remainder
            // for(; j < DIM_J; j++){
            //     output[i][k] += alpha * A[i][j] * B[j][k];
            // }
        }
    }
}
