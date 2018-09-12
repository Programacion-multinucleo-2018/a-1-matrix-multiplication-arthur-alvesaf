// Program by Arthur Alves Araujo Ferreira - All rights reserved
// ITESM ID: A01022593

#include <iostream>
#include <chrono>

const bool CPU_AND_COMPARE = true;

// Function that multiplies 2 matrixes with cuda
__global__ void matrixMultiplyGPU(int *A, int *B, int *C, const int n) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < n) {
        for (int iy = 0; iy < n; iy++) {
            for(int k = 0; k < n; k++) {
                C[iy * n + ix] += A[iy * n + k] * B[k * n + ix];
            }
        }
    }
}

// Function that multiplies 2 matrixes with cpu
void matrixMultiply(int *A, int *B, int *C, const int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            for(int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[j + k * n];
            }
        }
    }
}

// Compares two matrices
bool checkEquals(int *hostRef,int *gpuRef, const int n) {
    double ep = 1.0E-8;

    bool same = true;
    for (int i = 0; i < n*n; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > ep) {
            same = false;
            printf("[%d] host %d gpu %d\n", i, hostRef[i], gpuRef[i]);
            return same;
        }
    }

    return same;
}

int main(int argc, char* argv[]) {
    // Device setup
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // Code configuration
    int repetitions = 20;
    int n = 50;
    int nBytes = n*n * sizeof(int*);

    // Input matrix initialization and fill
    int *h_A = (int*)malloc(nBytes);
    int *h_B = (int*)malloc(nBytes);
    for(int i = 0; i < n*n; i++)  {
      h_A[i] = i+1;
      h_B[i] = i+1;
    }

    // Result matrixes initialization and zero fill
    int *gpuRef = (int*)malloc(nBytes);
    int *hostRef = (int*)malloc(nBytes);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // Device matrix global memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    // Transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, nBytes);  // Initialize matrix with 0s

    // Kernel execution configuration
    int dimx = 128;
    dim3 block(dimx, 1);
    dim3 grid((n + block.x - 1) / block.x, 1);
    printf("grid.x %d grid.y %d block.x %d block.y %d\n", grid.x, grid.y, block.x, block.y);

    // Variable initialization for repetitions
    double totalTimeGPU = 0;
    double totalTimeCPU = 0;
    std::chrono::duration<float, std::milli> duration_ms;

    // Repeat however may times was configured
    for (int i = 0; i < repetitions; i++) {
        // Multiply on GPU
        auto start = std::chrono::high_resolution_clock::now();
        matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, n);
        cudaDeviceSynchronize();
        auto end =  std::chrono::high_resolution_clock::now();

        duration_ms = end - start;
        totalTimeGPU += duration_ms.count();

        if (CPU_AND_COMPARE) {
            // Multiply on CPU
            start = std::chrono::high_resolution_clock::now();
            matrixMultiply(h_A, h_B, hostRef, n);
            end =  std::chrono::high_resolution_clock::now();

            duration_ms = end - start;
            totalTimeCPU += duration_ms.count();
        }

        // Copy result from device to host
        cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

        // Check if equals
        if (CPU_AND_COMPARE) {
            if (checkEquals(hostRef, gpuRef, n)) {
              printf("Matrix equal %d\n", i);
            } else {
              printf("Matrixes not equal %d\n", i);
              break;
            }
        }
    }

    // Print results
    printf("GPU matrix multiplication done in  %f ms\n", totalTimeGPU / repetitions);
    if (CPU_AND_COMPARE)
        printf("CPU matrix multiplication done in %f ms\n", totalTimeCPU / repetitions);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();

    return 0;
}
