#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<iostream>
#include<cuda_runtime.h>


using namespace std;

//----------------------------------- Structures and Globals---------------------------------------------

typedef struct {
	int dimension1;
	int dimension2;
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory
float *A_CPU, *B_CPU, *C_CPU, *C_GPU_result;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------------

void allocateAndInitializeAB();
void computeCpuMMM();
void computeGpuMMM();
void copyMatricesToGPU();
void copyResultFromGPU();
void compareHostAndGpuOutput();
void die(const char *error);
void check_error(cudaError e);
long long start_timer();
long long stop_timer(long long start_time, const char *name);

//----------------------------------- CUDA function definitions -----------------------------------------


//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;

	printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
	printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
	printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);
	if (A_MD.dimension2 != B_MD.dimension1) die("Dimension inconsistent for two matrices");

	allocateAndInitializeAB();

	// matrix multiplication in the CPU
	long long CPU_start_time = start_timer();
	//computeCpuMMM();
	long long CPU_time = stop_timer(CPU_start_time, "\nCPU");

	// matrix multiplication on the GPU
	long long GPU_start_time = start_timer();
	computeGpuMMM();
	long long GPU_time = stop_timer(GPU_start_time, "\tTotal");

	// compareHostAndGpuOutput();
	// Compute the speedup or slowdown
	// if (GPU_time > CPU_time) {
	// 	printf("\nCPU outperformed GPU by %.2fx\n", (float) GPU_time / (float) CPU_time);
	// } else {
	// 	printf("\nGPU outperformed CPU by %.2fx\n", (float) CPU_time / (float) GPU_time);
	// }

	return 0;
}

__global__ void mult_matrix_kernel(float *A, float *B, float *C, int dim_1, int dim_2, int dim_3) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	if (r <= dim_1 && c <= dim_3) {
		float sum = 0;
		for (int i = 0; i < dim_2; i ++) {
			sum += A[r * dim_2 + i] * B[i * dim_3 + c];
		}
		C[r * dim_3 + c] = sum;
	}
}

// allocate and initialize A and B using a random number generator
void allocateAndInitializeAB() {

	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	A = (float*) malloc(sizeofA);

	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++) {
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A[index] = (rand() % 1000) * 0.001;
		}
	}

	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.dimension1; i++) {
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B[index] = (rand() % 1000) * 0.001;
		}
	}

	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_CPU = (float*) malloc(sizeofC);
}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	long long memory_start_time = start_timer();
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A, sizeofA, cudaMemcpyHostToDevice));

	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B, sizeofB, cudaMemcpyHostToDevice));

	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
	stop_timer(memory_start_time, "\nGPU:\tTransfer to GPU");

}

// copy results from C_GPU which is in GPU card memory to C_CPU which is in the host CPU for result comparison
void copyResultFromGPU() {
	long long memory_start_time = start_timer();
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMemcpy(C_CPU, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
	stop_timer(memory_start_time, "\tTransfer from GPU");
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {

	// allocate the result matrix for the CPU computation
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C = (float*) malloc(sizeofC);

	// compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.dimension1; i++) {
		int a_i = i * A_MD.dimension2;
		int c_i = i * C_MD.dimension2;
		for (int j = 0; j < B_MD.dimension2; j++) {
			int c_index = c_i + j;
			C[c_index] = 0;
			for (int k = 0; k < B_MD.dimension1; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.dimension2 + j;
				C[c_index] += A[a_index] * B[b_index];
			}
		}
	}
}

// GPU version MM
void computeGpuMMM() {
	//initialize matrices in GPU global memory and copy CPU matrices to it

	copyMatricesToGPU();
	int dim_1 = A_MD.dimension1;
	int dim_2 = A_MD.dimension2;
	int dim_3 = B_MD.dimension2;
	int thread_x = 32;
	int thread_y = 32;
	dim3 grid(dim_1/thread_x, dim_3/thread_y);
	dim3 thread(thread_x, thread_y);
	long long exec_start_time = start_timer();
	mult_matrix_kernel <<<grid, thread>>> (A_GPU, B_GPU, C_GPU, dim_1, dim_2, dim_3);
	cudaThreadSynchronize();
	stop_timer(exec_start_time, "\tkernal excution time");
	//copy the result from GPU
	// copyResultFromGPU();
}

// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
void compareHostAndGpuOutput() {
	int totalElements = C_MD.dimension1 * C_MD.dimension2;
	int missmatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C[i] - C_CPU[i]) > 0.01) {
			missmatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C[i], C_CPU[i]);
		}
	}
	if (missmatchCount > 0) {
		printf("Computation is incorrect: outputs do not match in %d indexes\n", missmatchCount);
	} else {
		printf("Computation is correct: CPU and GPU outputs match\n");
	}
}

// Prints the specified error message and then exits
void die(const char *error) {
        printf("%s", error);
        exit(1);
}

// If the specified error code refers to a real error, report it and quit the program
void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

// Returns the current time in microseconds
long long start_timer() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}


// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *label) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
	printf("%s: %.5f sec\n", label, ((float) (end_time - start_time)) / (1000 * 1000));
	return end_time - start_time;
}
