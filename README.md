nvcc -std=c++17 ttt.cu -arch=sm_50 -o ttt -lcublas -lcuda -lcusolver --expt-relaxed-constexpr
