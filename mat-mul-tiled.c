/* Tiled matrix multiplication. The code assumes the height and width of the
 * matrices are multiples of TILE_WIDTH which is also the dimension of the block
 * in both the x and y direction
 */

#define TILE_WIDTH 16

__global__ void mat_mul_kernel(float *M, float *N, float *P, int width) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // Identify the row and column of the P to work on
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float Pvalue = 0.0f;
  for (int ph = 0; ph < (width / TILE_WIDTH); ++ph) {
    // Load value into shared memory
    Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH) + tx];
    Nds[ty][tx] = N[width * (ph * TILE_WIDTH + ty) + col];
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; ++i) {
      Pvalue += Mds[ty][i] * Nds[i][tx];
    }
    __syncthreads();
  }
  P[row * width + col] = Pvalue;
}

__global__ void mat_mul_kernel(float *M, float *N, float *P, int width) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // Identify the row and column of the P to work on
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float Pvalue = 0.0f;
  for (int ph = 0; ph < (width / TILE_WIDTH); ++ph) {
    // Load value into shared memory
    if (TILE_WIDTH * ph + tx < width) {
      Mds[ty][tx] = M[row * width + (ph * TILE_WIDTH) + tx];
    } else {
      Mds[ty][tx] = 0.0f;
    }
		if (TILE_WIDTH * ph + ty < width) {
			Nds[ty][tx] = N[width * (ph * TILE_WIDTH + ty) + col];
		} else {
			Nds[ty][tx] = 0.0f;
		}
    __syncthreads();

    // Perform partial dot product for current cell
    for (int i = 0; i < TILE_WIDTH; ++i) {
      Pvalue += Mds[ty][i] * Nds[i][tx];
    }
    __syncthreads();
  }
	if (row < width && col < width) {
	P[row * width + col] = Pvalue;
	}
}

torch::Tensor mat_mul_tiled(torch::Tensor M, torch::Tensor N) {
  CHECK_INPUT(M);
  CHECK_INPUT(N);
  int h_m = M.size(0);
  int w_m = M.size(1);
  int h_n = N.size(0);
  int w_n = N.size(1);
  printf("M height: %d, M width: %d\n", h_m, w_m);
  printf("N height: %d, N width: %d\n", h_n, w_n);

  auto output = torch::empty({h_m, w_n}, M.options());

  int threadPerBlock = TILE_WIDTH;
  dim3 dimBlock(threadPerBlock, threadPerBlock, 1);
  dim3 dimGrid(cdiv(w_m, dimBlock.x), cdiv(h_m, dimBlock.y), 1);

  mat_mul_kernel<<<dimGrid, dimBlock>>>(
      M.data_ptr<float>(), N.data_ptr<float>(), output.data_ptr<float>(), w_m);

  /*
  int threads = 256;
  rgb_to_grayscale_kernel<<<cdiv(w*h,threads), threads>>>(
      input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w*h);
      */

  cudaDeviceSynchronize();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
