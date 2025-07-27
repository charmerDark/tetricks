/*
einsum("bijk -> ",A)
*/

#define DIM_B 4
#define DIM_I  4
#define DIM_J  4
#define DIM_K  4


float A[DIM_B][DIM_I][DIM_J][DIM_K];



float kernel(float A[][DIM_I][DIM_J][DIM_K]);

int main()
{
  float output = kernel(A);
  return 0;
}

float kernel(float A[][DIM_I][DIM_J][DIM_K]){
  int b = 0, i = 0, j = 0, k = 0;
  float output = 0;
//  Assume output is all zeros
for (b =0; b<DIM_B;b++){
    for(i = 0; i < DIM_I; i++){
        for(k = 0; k < DIM_K; k++){
            for(j = 0; j < DIM_J; j++){
                output += A[b][i][j][k];
                // output += A[b][i][0][k];
                // output += A[b][i][1][k];
                // output += A[b][i][2][k];
                // output += A[b][i][3][k];
            }
        }
    }
}
return output;
}