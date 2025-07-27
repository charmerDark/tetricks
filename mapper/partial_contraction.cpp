/*
einsum("bijkl -> bik",A)
*/

#define DIM_B 4
#define DIM_I  4
#define DIM_J  4
#define DIM_K  4
#define DIM_L  4


float A[DIM_B][DIM_I][DIM_J][DIM_K][DIM_L];
float output[DIM_B][DIM_I][DIM_K];


void kernel(float A[][DIM_I][DIM_J][DIM_K][DIM_L], float output[][DIM_I][DIM_K]);

int main()
{
  kernel(A, output);
  return 0;
}

void kernel(float A[][DIM_I][DIM_J][DIM_K][DIM_L], float output[][DIM_I][DIM_K]){
  int b = 0, i = 0, j = 0, k = 0,l=0;
//  Assume output is all zeros
    for (b =0; b<DIM_B;b++){
        for(i = 0; i < DIM_I; i++){
            for(k = 0; k < DIM_K; k++){
                for(j = 0; j < DIM_J; j++){
                    for(l =0; l<DIM_L;l++){
                        output[b][i][k] += A[b][i][j][k][l];
                    }
                }
            }
        }
    }

}