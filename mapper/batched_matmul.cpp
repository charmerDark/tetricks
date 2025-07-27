/*
einsum("bij,bjk -> bik",A,B)
*/

#define DIM_B 4
#define DIM_I  4
#define DIM_J  4
#define DIM_K  4

float A[DIM_B][DIM_I][DIM_J];
float B[DIM_B][DIM_J][DIM_K];
float output[DIM_B][DIM_I][DIM_K];

// Add __restrict__ to the function signature
void kernel(const float (* __restrict__ A)[DIM_I][DIM_J],
            const float (* __restrict__ B)[DIM_J][DIM_K],
            float (* __restrict__ output)[DIM_I][DIM_K]);

int main()
{
  kernel(A, B, output);
  return 0;
}

void kernel(const float (* __restrict__ A)[DIM_I][DIM_J],
            const float (* __restrict__ B)[DIM_J][DIM_K],
            float (* __restrict__ output)[DIM_I][DIM_K])
{
    // Assume output is all zeros
    for (int b = 0; b < DIM_B; b++) {
        for (int i = 0; i < DIM_I; i++) {
            for (int k = 0; k < DIM_K; k++) {
                for (int j = 0; j < DIM_J; j++) {
                    output[b][i][k] += A[b][i][j] * B[b][j][k];
                }
            }
        }
    }
}
