#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

#define RADIUS 3
#define FILTER_DIM (2 * RADIUS + 1)
#define TILE_DIM 8 // Smaller tile for 3D

__global__ void conv3d_kernel(float *N, float *P, float *F, int radius,
                              int width, int height, int depth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  float Pvalue = 0.0f;
  int filterDim = 2 * radius + 1;
  for (int dep = 0; dep < filterDim; ++dep) {
    for (int row = 0; row < filterDim; ++row) {
      for (int col = 0; col < filterDim; ++col) {
        int inputz = z - radius + dep;
        int inputy = y - radius + row;
        int inputx = x - radius + col;
        if (inputz >= 0 && inputz < depth && inputy >= 0 && inputy < height &&
            inputx >= 0 && inputx < width) {
          Pvalue += F[dep * filterDim * filterDim + row * filterDim + col] *
                    N[inputz * height * width + inputy * width + inputx];
        }
      }
    }
  }
  if (z >= 0 && z < depth && y >= 0 && y < height && x >= 0 && x < width) {
    P[z * height * width + y * width + x] = Pvalue;
  }
}

// CPU implementation of 3D convolution with boundary checks
void cpuConvolution3D(float *input, float *output, float *kernel, int width,
                      int height, int depth, int kernelWidth, int kernelHeight,
                      int kernelDepth) {
  int radius = kernelWidth / 2;
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        float sum = 0.0f;
        for (int kz = -radius; kz <= radius; kz++) {
          for (int ky = -radius; ky <= radius; ky++) {
            for (int kx = -radius; kx <= radius; kx++) {
              int ix = x + kx;
              int iy = y + ky;
              int iz = z + kz;
              if (ix >= 0 && ix < width && iy >= 0 && iy < height && iz >= 0 &&
                  iz < depth) {
                int inputIdx = iz * width * height + iy * width + ix;
                int kernelIdx = (kz + radius) * kernelWidth * kernelHeight +
                                (ky + radius) * kernelWidth + (kx + radius);
                sum += input[inputIdx] * kernel[kernelIdx];
              }
            }
          }
        }
        output[z * width * height + y * width + x] = sum;
      }
    }
  }
}

int main() {
  const int width = 36;
  const int height = 36;
  const int depth = 36;
  const int radius = RADIUS;
  const int kernelDim = FILTER_DIM;
  const size_t inputSize = width * height * depth * sizeof(float);
  const size_t kernelSize = kernelDim * kernelDim * kernelDim * sizeof(float);
  const size_t outputSize = inputSize;

  // Allocate host memory
  float *h_input = (float *)malloc(inputSize);
  float *h_kernel = (float *)malloc(kernelSize);
  float *h_output_gpu = (float *)malloc(outputSize);
  float *h_output_cpu = (float *)malloc(outputSize);

  // Initialize input and kernel
  srand(time(NULL));
  for (int i = 0; i < width * height * depth; ++i) {
    h_input[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < kernelDim * kernelDim * kernelDim; ++i) {
    h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate device memory
  float *d_input, *d_output, *d_kernel;
  cudaMalloc((void **)&d_input, inputSize);
  cudaMalloc((void **)&d_output, outputSize);
  cudaMalloc((void **)&d_kernel, kernelSize);

  // Copy to device
  cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(F, h_kernel, kernelSize);

  cudaDeviceSynchronize();
  cudaError_t syncErr = cudaGetLastError();
  if (syncErr != cudaSuccess) {
    std::cerr << "Sync error: " << cudaGetErrorString(syncErr) << std::endl;
    return 1;
  }

  // Configure CUDA grid/block
  dim3 blockDim(TILE_DIM, TILE_DIM, TILE_DIM);
  dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
               (height + TILE_DIM - 1) / TILE_DIM,
               (depth + TILE_DIM - 1) / TILE_DIM);

  // Launch your kernel here
  conv3d_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, radius,
                                       width, height, depth);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  cudaDeviceSynchronize();

  cudaMemcpy(h_output_gpu, d_output, outputSize, cudaMemcpyDeviceToHost);

  // Run CPU version
  cpuConvolution3D(h_input, h_output_cpu, h_kernel, width, height, depth,
                   kernelDim, kernelDim, kernelDim);

  // Compare results
  const float tolerance = 1e-4f;
  bool match = true;
  for (int i = 0; i < width * height * depth; ++i) {
    if (fabs(h_output_gpu[i] - h_output_cpu[i]) > tolerance) {
      match = false;
      std::cout << "Mismatch at index " << i << ": GPU=" << h_output_gpu[i]
                << ", CPU=" << h_output_cpu[i] << std::endl;
    }
  }

  if (match) {
    std::cout << "GPU and CPU results match within tolerance!" << std::endl;
  } else {
    std::cout << "GPU and CPU results differ!" << std::endl;
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_kernel);
  free(h_input);
  free(h_kernel);
  free(h_output_gpu);
  free(h_output_cpu);

  return 0;
}
