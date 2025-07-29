/* Basic scalar multiplication and vector addition*/


#define SIZE 32


float output[SIZE];
float alpha = -1.5;
float X[SIZE];
float Y[SIZE];


void kernel(float X[], float Y[], float alpha, float output[]);

int main()
{
float a = 3.9;
  kernel(X, Y, a, output);

  return 0;
}

void kernel(float X[], float Y[], float alpha, float output[])
/*
scalar multiplication of alpha * X followed by vector addition of Y.
*/
{
int i;

for (i = 0; i < SIZE; ++i) {
output[i] = alpha * X[i] + Y[i];
}

}