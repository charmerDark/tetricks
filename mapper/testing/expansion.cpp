/*
einsum("i,j -> ij",A,B)
*/

#define DIM_B 4
#define DIM_I  4
#define DIM_J  4
#define DIM_K  4


float A[DIM_B][DIM_I][DIM_J];
float B[DIM_B][DIM_J][DIM_K];
float output[DIM_B][DIM_I][DIM_K];


void kernel(float A[][DIM_I][DIM_J], float B[][DIM_J][DIM_K], float output[][DIM_I][DIM_K]);

int main()
{
  kernel(A, B, output);
  return 0;
}

void kernel(float A[][DIM_I][DIM_J], float B[][DIM_J][DIM_K], float output[][DIM_I][DIM_K]){
  int b = 0, i = 0, j = 0, k = 0;
//  Assume output is all zeros
for (b =0; b<DIM_B;b++){
    for(i = 0; i < DIM_I; i++){
        for(k = 0; k < DIM_K; k++){
            // for(j = 0; j < DIM_J; j++){
            //     output[b][i][k] += A[b][i][j] * B[b][j][k];
            // }
          output[b][i][k] += A[b][i][0] * B[b][0][k];
          output[b][i][k] += A[b][i][1] * B[b][1][k];
          output[b][i][k] += A[b][i][2] * B[b][2][k];
          output[b][i][k] += A[b][i][3] * B[b][3][k];
        }
    }
}

}