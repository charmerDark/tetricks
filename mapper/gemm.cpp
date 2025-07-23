/*
alpha * A *B + beta* C
alpha * einsum("ij,jk ->ik") + beta * C
*/

#define DIM_I  4
#define DIM_J  4
#define DIM_K  4


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
  int i = 0, j = 0, k = 0;

//   for(i = 0; i < DIM_I; i++){
//       for(k = 0; k < DIM_K; k++){
//           output[i][k] = beta * C[i][k];
//       }
//   }
  
  for(i = 0; i < DIM_I; i++){
      for(k = 0; k < DIM_K; k++){
          for(j = 0; j < DIM_J; j++){
              output[i][k] += alpha * A[i][j] * B[j][k];
          }
      }
  }
}