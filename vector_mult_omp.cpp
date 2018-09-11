// Program by Arthur Alves Araujo Ferreira - All rights reserved
// ITESM ID: A01022593
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>

using namespace std;

// Funcion que aloca memoria
int** matrixAlloc() {
	int** matrix = (int**) malloc(rowsA*sizeof(int*));	
	// Allocate space for matrix
	for (i = 0; i < rowsA; i++) {
		matrix[i] = (int*) malloc(colsB*sizeof(int));
	}
	return matrix
}

// Function that opens a file and converts the matrix inside to a matrix of type float**
int** createMatrix(int cols, int rows) {
	// Allocate space for variable matrix
	int** matrix = matrixAlloc();

	// Fill variable matrix with contents from the file
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; ++j)
		{
			matrix[i][j] = rows * i + j + 1;
		}
	}

	return matrix;
}

// Function that multiplies 2 matrixes and returns the result
int** matrixMultiply(int** A, int** B, int rowsA, int colsA, int rowsB, int colsB) {
	// Return NULL in the case that the multiplication can't take place
	if (colsA != rowsB)
		return NULL;

	int** matrix = allocMatrix();
	int i = 0;
	#pragma omp parallel for private(i) shared(A, B, matrix)
	// Populate matrix with the multiplication AB
	for (i = 0; i < rowsA; i++)
	{
		for (int j = 0; j < colsB; ++j)
		{
			for (int k = 0; k < colsA; k++) {
				matrix[i][j] += A[i][k]*B[k][j];
			}
		}
	}

	return matrix;
}

// Function that clears the memory allocated for matrixes
void freeMatrix(int** matrix, int cols) {
	// Standard freeing of 2D array
	for (int i = 0; i < cols; i++) {
		free(matrix[i]);
	}
	free(matrix);
	return;
}

// Function that runs through a matrix and prints it
void printMatrix(int** matrix, int cols, int rows) {
	// Prints the matrix with numbers shortened to 2 decimals and tabs separating them
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}

	return;
}

int main() {
    int repetitions = 1;
    int rows = 1000;
    int cols = 1000;

    int** m1 = createMatrix(rows, cols);
    int** m2 = createMatrix(rows, cols);
    int** m3;

	omp_set_num_threads(4);

    double totalTime = 0;
    for (int i = 0; i < repetitions; i++) {
        // Multiply matrix1 x matrix2 and store the result in the variable m3
        auto start = std::chrono::high_resolution_clock::now();
        m3 = matrixMultiply(m1, m2, rows, cols, rows, cols);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<float, std::milli> duration_ms = end - start;
        totalTime += duration_ms.count();
    }

    std::cout << "time omp (ms): " << totalTime / repetitions << std::endl;

	// Free memory that was allocated for matrixes
	freeMatrix(m1, cols);
	freeMatrix(m2, cols);
	freeMatrix(m3, cols);
	return 0;
}