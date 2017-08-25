//matrix_mult.cu
//template provided by Prof. Andrew Grimshaw
//implementation by Jerry Sun(ys7va) 2017.05.08
//the program will take 4 parameters to specify the size of two matrices
//if only provided 1 value N, it will calculate the multiplication of two N * N matrices
#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<iostream>
#include<cuda_runtime.h>
using namespace std;

//Macro to specify block size
#define T_block 32

//----------------------------------- Structures and Globals---------------------------------------------
//store dimension of a matrix
typedef struct {
	int dimension1;
	int dimension2;
} ArrayMetadata2D;

// metadata variables describing dimensionalities of all data structures involved in the computation
ArrayMetadata2D A_MD, B_MD, C_MD;
// pointers for input and output arrays in the host memory
// *_CPU is for CPU calculation
// C_GPU_result is for storing GPU calculation result
float *A_CPU, *B_CPU, *C_CPU, *C_GPU_result;
// pointers for input and output arrays in the device memory (NVIDIA DRAM)
float *A_GPU, *B_GPU, *C_GPU;

//----------------------------------- host function definitions -----------------------------------------
void allocateAndInitializeHost();       //allocate and initialize all necessary memory on host machine
void computeCpuMMM();                   //matrix multiplication on CPU
void computeGpuMMM();                   //matrix multiplication on GPU, may use different kernel method
void copyMatricesToGPU();               //copy value in A_CPU & B_CPU to A_GPU & B_GPU respectively
void copyResultFromGPU();               //copy calculated value in C_GPU back into C_GPU_result
void compareHostAndGpuOutput();         //check if the result in C_GPU_result and C_CPU is identical
void die(const char *error);            //end the program
void check_error(cudaError e);          //check memory allocation on cuda
long long start_timer();                //timer for measurement
long long stop_timer(long long start_time, const char *name);  //timer for measurement

//----------------------------------- CUDA function definitions -----------------------------------------
//baseline approach for kernel method, each thread is responsible for one cell in final result
__global__ void mult_matrix_baseline(float *A, float *B, float *C, int dim_1, int dim_2, int dim_3);
//shared memory version for kernel method, a block of threads read data from DRAM together into shared
//memory and then do calculation block-wise
__global__ void mult_matrix_shared(float *A, float *B, float *C, int dim_1, int dim_2, int dim_3);


//-------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    //parse the command-line argument
	A_MD.dimension1 = (argc > 1) ? atoi(argv[1]) : 100;
	A_MD.dimension2 = (argc > 2) ? atoi(argv[2]) : A_MD.dimension1;
	B_MD.dimension1 = (argc > 3) ? atoi(argv[3]) : A_MD.dimension2;
	B_MD.dimension2 = (argc > 4) ? atoi(argv[4]) : B_MD.dimension1;
	C_MD.dimension1 = A_MD.dimension1;
	C_MD.dimension2 = B_MD.dimension2;

	printf("Matrix A is %d-by-%d\n", A_MD.dimension1, A_MD.dimension2);
	printf("Matrix B is %d-by-%d\n", B_MD.dimension1, B_MD.dimension2);
	printf("Matrix C is %d-by-%d\n", C_MD.dimension1, C_MD.dimension2);
    //if dim2 of A and dim1 of B is different then they can't be multiplied
	if (A_MD.dimension2 != B_MD.dimension1) die("Dimension inconsistent for two matrices");

    //allocate all necessary memory on host
	allocateAndInitializeHost();

	// matrix multiplication in the CPU, commented for large-scale
	// long long CPU_start_time = start_timer();
	// computeCpuMMM();
	// long long CPU_time = stop_timer(CPU_start_time, "\nCPU");

	// matrix multiplication on the GPU
	long long GPU_start_time = start_timer();
	computeGpuMMM();
	long long GPU_time = stop_timer(GPU_start_time, "\tTotal");

    //check the final result
	//commented when CPU result is not available
    //compareHostAndGpuOutput();

	return 0;
}


__global__ void mult_matrix_baseline(float *A, float *B, float *C, int dim_1, int dim_2, int dim_3) {
    // retrieve the corresponding row & col in final output matrix
    int r = blockIdx.x * T_block + threadIdx.x;
    int c = blockIdx.y * T_block + threadIdx.y;
    // check if index is in bound
    if (r < dim_1 && c < dim_3) {
        float sum = 0;
        // calculate inner product of two vectors
        for (int i = 0; i < dim_2; i++) {
                sum += A[r * dim_1 + i] * B[i * dim_2 + c];
            }
        // assign final results
        C[r * dim_3 + c] = sum;
    }
}

// Compute C = A * B
__global__ void mult_matrix_shared(float *A, float *B, float *C, int dim_1, int dim_2, int dim_3) {

  // store corresponding value in registers
  int b_x = blockIdx.x;
  int b_y = blockIdx.y;
  int t_x = threadIdx.x;
  int t_y = threadIdx.y;

  // retrieve row & col number in final output
  int r = b_y * T_block + t_y;
  int c = b_x * T_block + t_x;

  float s = 0;
  // initiate share memory space
  __shared__ float block_A[T_block][T_block];
  __shared__ float block_B[T_block][T_block];

  // bool variable to check if inbound
  bool inplace = r < dim_1 && c < dim_3;

  // iterate through all blocks in using a ceiling function to deal with corner cases
  for (int m = 0; m < (dim_2 - 1) / T_block + 1; m++) {
    // column num for the retrieved cell in matrix A
    int col = m * T_block + t_x;
    // load value from matrix A, if not available assign 0
    block_A[t_y][t_x] = (r < dim_1 && col < dim_2) ? A[r * dim_1 + col] : 0.0;
    // row num for the retrieved cell in matrix B
	int row = m * T_block + t_y;
    // load value from matrix B, if not available assign 0
	block_B[t_y][t_x] = (row < dim_2 && c < dim_3) ? B[row * dim_3 + c] : 0.0;
    // sync all threads, wait till all threads finish loading
    __syncthreads();

    //if inplace calculate the inner product within two blocks in A and B
	if (inplace)
		for (int i = 0; i < T_block; i++)
			s += block_A[t_y][i] * block_B[i][t_x];
    //sync threads, wait till all threads finish using shared memory in current iteration
    __syncthreads();
  }

  //assign final result
  if (inplace)
    C[r * dim_3 + c] = s;
}

// GPU version MM
void computeGpuMMM() {
	copyMatricesToGPU();
    //for a matrix multiplication problem, only three dimensions are needed
    //two dims for the final matrix, and one for dim2 of A and dim1 of B(identical)
	int dim_1 = A_MD.dimension1;
	int dim_2 = A_MD.dimension2;
	int dim_3 = B_MD.dimension2;
    //initialize gridblock, and threadblock size
    //here we assume each thread always responsible for cell
    dim3 thread(T_block, T_block);
    //if dim_1 not divisible by T_block, we use ceiling function
    //in order to handle corner cases
    dim3 grid((dim_1 - 1) / T_block + 1, (dim_3 - 1) / T_block + 1);
	long long exec_start_time = start_timer();
	//call kernel method, passing in three GPU pointers and three dimensions
    mult_matrix_shared <<<grid, thread>>> (A_GPU, B_GPU, C_GPU, dim_1, dim_2, dim_3);
    //synchroniztion
	cudaThreadSynchronize();
	stop_timer(exec_start_time, "\tkernal excution time");
	//copy the result from GPU
	copyResultFromGPU();
}



// allocate and initialize A and B using a random number generator,
// also initialize C_CPU and C_GPU_resul
void allocateAndInitializeHost() {
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	A_CPU = (float*) malloc(sizeofA);
	srand(time(NULL));
  	for (int i = 0; i < A_MD.dimension1; i++) {
		for (int j = 0; j < A_MD.dimension2; j++) {
			int index = i * A_MD.dimension2 + j;
			A_CPU[index] = (rand() % 1000) * 0.001;
		}
	}

	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	B_CPU = (float*) malloc(sizeofB);
  	for (int i = 0; i < B_MD.dimension1; i++) {
		for (int j = 0; j < B_MD.dimension2; j++) {
			int index = i * B_MD.dimension2 + j;
			B_CPU[index] = (rand() % 1000) * 0.001;
		}
	}

	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	C_GPU_result = (float*) malloc(sizeofC);
	C_CPU = (float*) malloc(sizeofC);

}

// allocate memory in the GPU for all matrices, and copy A and B content from the host CPU memory to the GPU memory
void copyMatricesToGPU() {
	long long memory_start_time = start_timer();
	size_t sizeofA = A_MD.dimension1 * A_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &A_GPU, sizeofA));
	check_error(cudaMemcpy(A_GPU, A_CPU, sizeofA, cudaMemcpyHostToDevice));

	size_t sizeofB = B_MD.dimension1 * B_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &B_GPU, sizeofB));
	check_error(cudaMemcpy(B_GPU, B_CPU, sizeofB, cudaMemcpyHostToDevice));

	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMalloc((void **) &C_GPU, sizeofC));
	stop_timer(memory_start_time, "\nGPU:\tTransfer to GPU");

}

// copy results from C_GPU which is in GPU card memory to C_CPU_result which is in the host CPU for result comparison
void copyResultFromGPU() {
	long long memory_start_time = start_timer();
	size_t sizeofC = C_MD.dimension1 * C_MD.dimension2 * sizeof(float);
	check_error(cudaMemcpy(C_GPU_result, C_GPU, sizeofC, cudaMemcpyDeviceToHost));
	stop_timer(memory_start_time, "\tTransfer from GPU");
}

// do a straightforward matrix-matrix multiplication in the CPU
// notice that this implementation can be massively improved in the CPU by doing proper cache blocking but we are
// not providing you the efficient CPU implementation as that reveals too much about the ideal GPU implementation
void computeCpuMMM() {

	// compute C[i][j] as the sum of A[i][k] * B[k][j] for all columns k of A
	for (int i = 0; i < A_MD.dimension1; i++) {
		int a_i = i * A_MD.dimension2;
		int c_i = i * C_MD.dimension2;
		for (int j = 0; j < B_MD.dimension2; j++) {
			int c_index = c_i + j;
			C_CPU[c_index] = 0;
			for (int k = 0; k < B_MD.dimension1; k++) {
				int a_index = a_i + k;
				int b_index = k * B_MD.dimension2 + j;
				C_CPU[c_index] += A_CPU[a_index] * B_CPU[b_index];
			}
		}
	}
}

// function to determine if the GPU computation is done correctly by comparing the output from the GPU with that
void compareHostAndGpuOutput() {
	int totalElements = C_MD.dimension1 * C_MD.dimension2;
	int missmatchCount = 0;
	for (int i = 0; i < totalElements; i++) {
		if (fabs(C_GPU_result[i] - C_CPU[i]) > 0.01) {
			missmatchCount++;
			printf("mismatch at index %i: %f\t%f\n", i, C_CPU[i], C_GPU_result[i]);
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
