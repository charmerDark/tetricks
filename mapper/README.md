clang++ mapper.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -o mapper.o

clang++ -emit-llvm -fno-unroll-loops -O0 -S saxpy.cpp 
./mapper.o -f saxpy.ll -fn _Z6kernelPfS_fS_
dot -Tpng temp.dot  -o temp.png