cd frontend
dune build
cd ../backend
clang++ -std=c++17 -o tetricks_backend tetricks_backend.cpp