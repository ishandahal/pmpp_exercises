#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define RADIUS 3
#define FILTER_DIM (2 * RADIUS + 1)
#define TILE_DIM 32

__constant__ float F[FILTER_DIM][FILTER_DIM];

__global__ void conv2d_tiled_cached_kernel(float *N, float *P, int radius,
                                           int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float Nds[TILE_DIM][TILE_DIM];
  // row and col are never negative so only need to check the right and bottom
  // bounds
  if (row < height && col < width) {
    Nds[threadIdx.y][threadIdx.x] = N[row * width + col];
  }
  __syncthreads();

  float Pvalue = 0.0f;
  if (row < height && col < width) {
    for (int frow = -radius; frow < radius + 1; ++frow) {
      for (int fcol = -radius; fcol < radius + 1; ++fcol) {
        int tileRow = threadIdx.y + frow;
        int tileCol = threadIdx.x + fcol;
        if (tileRow >= 0 && tileRow < TILE_DIM && tileCol >= 0 &&
            tileCol < TILE_DIM && row + frow >= 0 && row + frow < height &&
            col + fcol >= 0 && col + fcol < width) {
          Pvalue +=
              F[frow + radius][fcol + radius] *
              Nds[tileRow][tileCol]; // Add back radius to undo the addition
        } else {
          if (row + frow >= 0 && row + frow < height && col + fcol >= 0 &&
              col + fcol < width) {
            Pvalue += F[frow + radius][fcol + radius] *
                      N[(row + frow) * width + (col + fcol)];
          }
        }
      }
      P[row * width + col] = Pvalue;
    }
  }
}

// Helper function to print a small matrix
void printMatrix(float *matrix, int width, int height) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      std::cout << matrix[y * width + x] << " ";
    }
    std::cout << std::endl;
  }
}

// CPU implementation of 2D convolution with index clamping
void cpuConvolution2D(float *input, float *output, float *kernel, int width,
                      int height, int kernelWidth, int kernelHeight) {
  int radius = (kernelWidth - 1) / 2;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float sum = 0.0f;
      for (int ky = -radius; ky <= radius; ky++) {
        for (int kx = -radius; kx <= radius; kx++) {
          int inputX = x + kx;
          int inputY = y + ky;
          int kernelIdx = (ky + radius) * kernelWidth + (kx + radius);
          if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
            sum += input[inputY * width + inputX] * kernel[kernelIdx];
          }
        }
      }
      output[y * width + x] = sum;
    }
  }
}

int main() {
  // Define dimensions
  const int width = 100;
  const int height = 90;
  const int radius = RADIUS;
  const int kernelWidth = FILTER_DIM;
  const int kernelHeight = FILTER_DIM;
  const int outputWidth = width;
  const int outputHeight = height;

  // Calculate sizes
  size_t inputSize = width * height * sizeof(float);
  size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
  size_t outputSize = outputWidth * outputHeight * sizeof(float);

  // Host memory allocation
  float *h_input = (float *)malloc(inputSize);
  float *h_kernel = (float *)malloc(kernelSize);
  float *h_output_gpu = (float *)malloc(outputSize);
  float *h_output_cpu = (float *)malloc(outputSize);

  // Initialize input and kernel with random values
  srand(time(NULL));
  // srand(1);
  for (int i = 0; i < width * height; i++) {
    h_input[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < kernelWidth * kernelHeight; i++) {
    h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Device memory allocation
  float *d_input, *d_kernel, *d_output;
  cudaMalloc(&d_input, inputSize);
  cudaMalloc(&d_kernel, kernelSize);
  cudaMalloc(&d_output, outputSize);

  // Copy data to device
  cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F, h_kernel, kernelWidth * kernelHeight * sizeof(float));

  // Define block and grid dimensions
  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim((outputWidth + TILE_DIM - 1) /
                   TILE_DIM, // Ceiling division by TILE_DIM
               (outputHeight + TILE_DIM - 1) / TILE_DIM);
  // Launch kernel

  conv2d_tiled_cached_kernel<<<gridDim, blockDim>>>(d_input, d_output, radius,
                                                    width, height);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  // Synchronize device
  cudaDeviceSynchronize();

  // Copy GPU results back to host
  cudaMemcpy(h_output_gpu, d_output, outputSize, cudaMemcpyDeviceToHost);

  // Compute CPU reference result
  cpuConvolution2D(h_input, h_output_cpu, h_kernel, width, height, kernelWidth,
                   kernelHeight);

  // Compare GPU and CPU results
  const float tolerance =
      1e-5f; // Small tolerance for floating-point differences
  bool match = true;
  for (int i = 0; i < outputWidth * outputHeight; i++) {
    float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
    if (diff > tolerance) {
      match = false;
      std::cout << "Mismatch at index " << i << " (y=" << i / width
                << ", x=" << i % width << "): " << "GPU=" << h_output_gpu[i]
                << ", CPU=" << h_output_cpu[i] << ", diff=" << diff
                << std::endl;
    }
  }

  if (match) {
    std::cout << "GPU and CPU results match within tolerance!" << std::endl;
  } else {
    std::cout << "GPU and CPU results differ!" << std::endl;
  }
  /*
      // Optional: Print samples
      std::cout << "\nInput sample:" << std::endl;
      printMatrix(h_input, width, std::min(5, height));
      std::cout << "\nKernel:" << std::endl;
      printMatrix(h_kernel, kernelWidth, kernelHeight);
      std::cout << "\nGPU Output sample:" << std::endl;
      printMatrix(h_output_gpu, outputWidth, std::min(5, outputHeight));
      std::cout << "\nCPU Output sample:" << std::endl;
      printMatrix(h_output_cpu, outputWidth, std::min(5, outputHeight));
      */

  // Clean up
  cudaFree(d_input);
  cudaFree(d_kernel);
  cudaFree(d_output);
  free(h_input);
  free(h_kernel);
  free(h_output_gpu);
  free(h_output_cpu);

  return 0;
}
