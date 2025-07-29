// vector_add.cpp
#include <iostream>
#include <vector>

void vector_add(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 8;
    float a[N] = {1,2,3,4,5,6,7,8};
    float b[N] = {8,7,6,5,4,3,2,1};
    float c[N] = {0};

    vector_add(a, b, c, N);

    for (int i = 0; i < N; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
