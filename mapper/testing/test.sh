#!/bin/bash


echo "Compiling mapper.cpp..."
clang++ mapper.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o mapper.o


# Format: "source.cpp:kernel_name"
declare -a kernels=(
    "batched_matmul.cpp:_Z6kernelPA4_A4_KfS2_PA4_A4_f"
    "contraction.cpp:_Z6kernelPA4_A4_A4_f"
    "expansion.cpp:_Z6kernelPA4_A4_fS1_S1_"
    "gemm.cpp:_Z6kernelPA4_fS0_S0_ffS0_"
    "partial_contraction.cpp:_Z6kernelPA4_A4_A4_A4_fPS0_" 
)

for entry in "${kernels[@]}"; do
    src="${entry%%:*}"
    kernel="${entry##*:}"
    base="${src%.cpp}"

    echo "Processing $src with kernel $kernel..."

    clang++ -emit-llvm -fno-unroll-loops -O1 -S "$src" -o "${base}.ll"

    ./mapper.o -f "${base}.ll" -fn "$kernel" -dotfile "${base}.dot"

    dot -Tpng "${base}.dot" -o "${base}.png"

    echo "Generated ${base}.png"
done

echo "All done!"
