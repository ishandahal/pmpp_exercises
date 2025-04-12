#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define RADIUS 3
#define FILTER_DIM (2 * RADIUS + 1)
#define INPUT_TILE_DIM 16 // *** Block dim needs to match this ***
#define OUTPUT_TILE_DIM ((INPUT_TILE_DIM) - 2 * RADIUS)

__constant__ float F[FILTER_DIM][FILTER_DIM];

__global__ void conv2d_tiled_kernel(float *N, float *P, int radius, int width,
                                    int height) {
  int row = blockIdx.y * OUTPUT_TILE_DIM + threadIdx.y - radius;
  int col = blockIdx.x * OUTPUT_TILE_DIM + threadIdx.x - radius;

  __shared__ float Nds[INPUT_TILE_DIM][INPUT_TILE_DIM];
  // Load input tile after checking the boundary condition
  if (row >= 0 && row < height && col >= 0 && col < width) {
    Nds[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    Nds[threadIdx.y][threadIdx.x] = 0.0f;
  }
  __syncthreads();

  int tileRow = threadIdx.y - radius;
  int tileCol = threadIdx.x - radius;
  if (row >= 0 && row < height && col >= 0 && col < width) {
    if (tileRow >= 0 && tileRow < OUTPUT_TILE_DIM && tileCol >= 0 &&
        tileCol < OUTPUT_TILE_DIM) {
      float Pvalue = 0.0f;
      for (int fRow = 0; fRow < FILTER_DIM; ++fRow) {
        for (int fCol = 0; fCol < FILTER_DIM; ++fCol) {
          Pvalue += F[fRow][fCol] * Nds[fRow + tileRow][fCol + tileCol];
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
  const int width = 1500;
  const int height = 1300;
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
  // cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(F, h_kernel, kernelWidth * kernelHeight * sizeof(float));

  // Define block and grid dimensions
  dim3 blockDim(INPUT_TILE_DIM, INPUT_TILE_DIM);
  dim3 gridDim((outputWidth + OUTPUT_TILE_DIM - 1) /
                   OUTPUT_TILE_DIM, // Ceiling division by OUTPUT_TILE_DIM
               (outputHeight + OUTPUT_TILE_DIM - 1) / OUTPUT_TILE_DIM);
  // Launch kernel

  conv2d_tiled_kernel<<<gridDim, blockDim>>>(d_input, d_output, radius, width,
                                             height);

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
