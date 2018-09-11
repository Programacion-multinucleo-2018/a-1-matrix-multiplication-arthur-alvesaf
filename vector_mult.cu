// Program by Arthur Alves Araujo Ferreira - All rights reserved
// ITESM ID: A01022593
#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;

// Function that multiplies 2 matrixes and returns the result
__global__ void matrixMultiplyGPU(int *A, int *B, int *C, const int n) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < n && iy < n) {
        for(int k = 0; k < n; k++) {
            C[iy * n + ix] += A[iy * n + k] * B[k * n + ix];
        }
    }
}

// Function that opens a file and converts the matrix inside to a matrix of type float**
int* createMatrix(int n) {
	// Allocate space for variable matrix
	int* matrix = (int*) malloc(n*n * sizeof(int));

	// Fill variable matrix with contents from the file
	for (int i = 0; i < n*n; i++)
	{
			matrix[i] = i + 1;
	}

	return matrix;
}

// Function that runs through a matrix and prints it
void printMatrix(int** matrix, int cols, int rows) {
	// Prints the matrix with numbers shortened to 2 decimals and tabs separating them
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << matrix[rows + i + j] << " ";
		}
		std::cout << std::endl;
	}

	return;
}

int main(int argc, char **argv) {
	// Set up device
	int dev = 0;
	cudaDevideProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
    SAFE_CALL(cudaSetDevice(dev), "Error setting device");
    
	int repetitions = 1;
    int n = 1000;
	int bytes = n*n * sizeof(int)

    int* h_A = createMatrix(n);
    int* h_B = createMatrix(n);

    int* hostRef = (int *)malloc(bytes);
    int* gpuRef = (int *)malloc(bytes);
	// printMatrix(h_A, n, n);
	// printMatrix(h_B, n, n);

	// zero matrix
	memSet(hostRef, 0, bytes);
	memSet(gpuRef, 0, bytes);

	int *d_A, *d_B, *d_C;
    SAFE_CALL(cudaMalloc((void **)&d_A, bytes), "Error allocating d_A");
    SAFE_CALL(cudaMalloc((void **)&d_B, bytes), "Error allocating d_B");
    SAFE_CALL(cudaMalloc((void **)&d_C, bytes), "Error allocating d_C");

	SAFE_CALL(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "Error copying d_A");
    SAFE_CALL(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "Error copying d_B");
    SAFE_CALL(cudaMemset(d_C, 0, bytes), "Error setting d_C to zeros");

	dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    printf("grid.x %d grid.y %d block.x %d block.y %d\n", grid.x, grid.y, block.x, block.y);

    double totalTime = 0;
    for (int i = 0; i < repetitions; i++) {

        // Multiply matrix1 x matrix2 and store the result in the variable m3
        auto start = std::chrono::high_resolution_clock::now();
		matrixMultiplyGPU<<<grid, block>>>(d_a, d_b, d_c, n);
		cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<float, std::milli> duration_ms = end - start;
        totalTime += duration_ms.count();
    }

    std::cout << "time omp (ms): " << totalTime / repetitions << std::endl;

	// Free memory that was allocated for matrixes
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);

	SAFE_CALL(cudaDeviceReset(), "Error reseting");
	return 0;
}