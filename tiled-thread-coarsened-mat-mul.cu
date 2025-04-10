#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel for matrix multiplication
// CUDA kernel for matrix multiplication
#define TILE_WIDTH 32
#define COARSE_FACTOR 2

__global__ void tiled_thread_coarsened_matrix_multiply_kernel(
    float *M, float *N, float *P, int M_height, int M_width, int N_width) {

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  float Pvalues[COARSE_FACTOR];
  for (int i = 0; i < COARSE_FACTOR; ++i) {
    Pvalues[i] = 0.0f;
  }

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * blockDim.y + ty;
  int col = blockIdx.x * blockDim.x * COARSE_FACTOR + tx;

  for (int ph = 0; ph < ((M_width + TILE_WIDTH - 1) / TILE_WIDTH); ++ph) {
    // Load from M
    if ((row < M_height) && ((ph * TILE_WIDTH + tx) < M_width)) {
      Mds[ty][tx] = M[row * M_width + (ph * TILE_WIDTH) + tx];
    } else {
      Mds[ty][tx] = 0.0f;
    }

    // Each thread is responsible for COARSE_FACTOR values
    for (int c = 0; c < COARSE_FACTOR; ++c) {
      int coarsed_col = (c * TILE_WIDTH) + col;

      // Load from N
      if ((coarsed_col < N_width) && ((ph * TILE_WIDTH + ty) < M_width)) {
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * N_width + coarsed_col];
      } else {
        Nds[ty][tx] = 0.0f;
      }
      __syncthreads();
      for (int i = 0; i < TILE_WIDTH; ++i) {
        Pvalues[c] += Mds[ty][i] * Nds[i][tx];
      }
      __syncthreads();
    }
  }

  for (int i = 0; i < COARSE_FACTOR; ++i) {
    if (row < M_height && (col + i * TILE_WIDTH) < N_width) {
      P[row * N_width + col + i * TILE_WIDTH] = Pvalues[i];
    }
  }
}

// CPU version for verification
void matrixMultiplyCPU(float *A, float *B, float *C, int m, int n, int p) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * p + j];
      }
      C[i * p + j] = sum;
    }
  }
}

// Function to initialize matrix with random values
void initializeMatrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = (float)(rand() % 100) / 10.0f;
  }
}

// Function to print matrix
void printMatrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%.2f\t", matrix[i * cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main() {
  // Matrix dimensions
  int m = 40; // rows of A
  int n = 50; // columns of A, rows of B
  int p = 43; // columns of B

  // Host matrices
  float *h_A, *h_B, *h_C, *h_C_cpu;
  // Device matrices
  float *d_A, *d_B, *d_C;

  // Allocate host memory
  h_A = (float *)malloc(m * n * sizeof(float));
  h_B = (float *)malloc(n * p * sizeof(float));
  h_C = (float *)malloc(m * p * sizeof(float));
  h_C_cpu = (float *)malloc(m * p * sizeof(float));

  // Initialize matrices with random values
  srand(time(NULL));
  initializeMatrix(h_A, m, n);
  initializeMatrix(h_B, n, p);

  // Allocate device memory
  cudaMalloc(&d_A, m * n * sizeof(float));
  cudaMalloc(&d_B, n * p * sizeof(float));
  cudaMalloc(&d_C, m * p * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, n * p * sizeof(float), cudaMemcpyHostToDevice);

  // Set up grid and block dimensions
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch kernel
  tiled_thread_coarsened_matrix_multiply_kernel<<<blocksPerGrid,
                                                  threadsPerBlock>>>(
      d_A, d_B, d_C, m, n, p);

  // Copy result back to host
  cudaMemcpy(h_C, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

  // Compute CPU result for verification
  matrixMultiplyCPU(h_A, h_B, h_C_cpu, m, n, p);

  // Verify results
  float maxError = 0.0f;
  int errorCount = 0;
  for (int i = 0; i < m * p; i++) {
    float diff = fabs(h_C[i] - h_C_cpu[i]);
    if (diff > maxError)
      maxError = diff;
    if (diff > 1e-5)
      errorCount++;
  }

  // Print matrices and results
  // printf("Matrix A (%d x %d):\n", m, n);
  // printMatrix(h_A, m, n);
  // printf("Matrix B (%d x %d):\n", n, p);
  // printMatrix(h_B, n, p);
  // printf("GPU Result (%d x %d):\n", m, p);
  // printMatrix(h_C, m, p);
  // printf("CPU Result (%d x %d):\n", m, p);
  // printMatrix(h_C_cpu, m, p);

  printf("Verification Results:\n");
  printf("Maximum error: %e\n", maxError);
  printf("Number of elements with significant differences: %d\n", errorCount);
  printf("Test %s\n",
         (maxError < 1e-5 && errorCount == 0) ? "PASSED" : "FAILED");

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
