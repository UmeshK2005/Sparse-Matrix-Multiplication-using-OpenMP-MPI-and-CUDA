# Makefile for sparse_matrix_mult

NVCC      := nvcc
NVCCFLAGS := -O3 -use_fast_math -Xcompiler "-O3 -march=native -mtune=native -fopenmp -funroll-loops -ffast-math -fstrict-aliasing"
MPICXX    := mpicxx
CXXFLAGS  := -O3 -march=native -mtune=native -fopenmp -funroll-loops -ffast-math -fstrict-aliasing

# default target
all: a4

# link into 'a4'
a4: sparse_matrix_mult.o
	@echo "Linking --> a4"
	$(MPICXX) $(CXXFLAGS) $^ -o $@ -lcudart

# compile CUDA source to object
sparse_matrix_mult.o: sparse_matrix_mult.cu
	@echo "Compiling $< --> $@"
	@unset OMP_NUM_THREADS; \
	$(NVCC) $(NVCCFLAGS) -arch=sm_35 -c $< -o $@

# clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f sparse_matrix_mult.o a4
