//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <helper_cuda.h>
//#include <helper_math.h>
//#include <vector_types.h>
#include "functionGPU.cuh"
//typedef Real Real;

__global__ void WaveIterationKernel(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b = Real(0.0))
{
    //unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    //if (i >= N * M) return;
    //unsigned int x = i % M;
    //unsigned int y = i / M;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y>=N) return;
    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) { U[x + y * M + N * M * ((t + 1) % 3)] = Real(0.0); return; }
    if (x == x_s && y == y_s) U[x + y * M + (t % 3) * N * M] += F[t - 1]; // Source

    Real val;
    if (U[x + y * M + 3 * N * M] > Real(0.0)) {
        val = Real(-4.0) * U[x + y * M + N * M * (t % 3)];
        val += U[x - 1 + y * M + N * M * (t % 3)];// (U[x - 1 + y * M + 3 * N * M] > Real(0.0)) ? U[x - 1 + y * M + N * M * (t % 3)] : Real(0.0);//U[(x+1+y*M)*4+t%3];
        val += U[x + 1 + y * M + N * M * (t % 3)];// (U[x + 1 + y * M + 3 * N * M] > Real(0.0)) ? U[x + 1 + y * M + N * M * (t % 3)] : Real(0.0);//U[(x-1+y*M)*4+t%3];
        val += U[x + M + y * M + N * M * (t % 3)];// (U[x + M + y * M + 3 * N * M] > Real(0.0)) ? U[x + M + y * M + N * M * (t % 3)] : Real(0.0);//U[(x-M+y*M)*4+t%3];
        val += U[x - M + y * M + N * M * (t % 3)];// (U[x - M + y * M + 3 * N * M] > Real(0.0)) ? U[x - M + y * M + N * M * (t % 3)] : Real(0.0);//U[(x+M+y*M)*4+t%3];
        val *= (U[x + y * M + 3 * N * M] * U[x + y * M + 3 * N * M]) * v * v * d_t * d_t / (d_x * d_x); // ???
        val += Real(2.0) * U[x + y * M + N * M * (t % 3)] - U[x + y * M + N * M * ((t-1) % 3)] * (1.0 - d_t * b * Real(0.5));
        val /= (Real(1.0) + d_t * b * Real(0.5));
    }
    else { val = Real(0.0); }
    U[x + y * M + N * M * ((t + 1) % 3)] = val;
    if (x == x_ir && y == y_ir) IR[t - 1] = val; // (t-1) because we run with t+1
    //}
}

__global__ void WaveIterationSimple(Real* U, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, const Real b = Real(0.0))
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= N) return;
    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) { U[x + y * M + N * M * ((t + 1) % 3)] = Real(0.0); return; }

    Real val;
    if (U[x + y * M + 3 * N * M] > Real(0.0)) {
        val = Real(-4.0) * U[x + y * M + N * M * (t % 3)];
        val += U[x - 1 + y * M + N * M * (t % 3)];// (U[x - 1 + y * M + 3 * N * M] > Real(0.0)) ? U[x - 1 + y * M + N * M * (t % 3)] : Real(0.0);//U[(x+1+y*M)*4+t%3];
        val += U[x + 1 + y * M + N * M * (t % 3)];// (U[x + 1 + y * M + 3 * N * M] > Real(0.0)) ? U[x + 1 + y * M + N * M * (t % 3)] : Real(0.0);//U[(x-1+y*M)*4+t%3];
        val += U[x + M + y * M + N * M * (t % 3)];// (U[x + M + y * M + 3 * N * M] > Real(0.0)) ? U[x + M + y * M + N * M * (t % 3)] : Real(0.0);//U[(x-M+y*M)*4+t%3];
        val += U[x - M + y * M + N * M * (t % 3)];// (U[x - M + y * M + 3 * N * M] > Real(0.0)) ? U[x - M + y * M + N * M * (t % 3)] : Real(0.0);//U[(x+M+y*M)*4+t%3];
        val *= (U[x + y * M + 3 * N * M] * U[x + y * M + 3 * N * M]) * v * v * d_t * d_t / (d_x * d_x); // ???
        val += Real(2.0) * U[x + y * M + N * M * (t % 3)] - U[x + y * M + N * M * ((t - 1) % 3)] * (Real(1.0) - d_t * b * Real(0.5));
        val /= (Real(1.0) + d_t * b * Real(0.5));
    }
    else { val = Real(0.0); }
    U[x + y * M + N * M * ((t + 1) % 3)] = val;
}

__global__ void MultipleIterationsKernel(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b = Real(0.0), const unsigned int iterations=1)
{
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(M / 32 + ((M % 32 == 0)? 0:1), N / 32 + ((N % 32 == 0) ? 0 : 1), 1);
    for (int time = t; time < t+iterations; ++time) {
        U[x_s + y_s * M + (time % 3) * N * M] += F[time - 1]; // Source
        WaveIterationSimple << <gridDim, blockDim >> > (U, N, M, time, v, d_x, d_t, x_ir, y_ir, x_s, y_s, b);
        cudaDeviceSynchronize();
        IR[time - 1] = U[x_ir + y_ir * M + ((time + 1) % 3) * N * M];
    }
}

#define BLOCK_DIM 32

__global__ void WaveIterationKernelSM(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b = Real(0.0))
{
    extern __shared__ Real s[];
    //__shared__ Real values[(BLOCK_DIM + 2) * (BLOCK_DIM + 2)],
    //                prev_values[BLOCK_DIM  * BLOCK_DIM],
    //                walls[(BLOCK_DIM + 2) * (BLOCK_DIM + 2)];
    Real* values = s;
    Real* prev_values = values + (blockDim.x + 2) * (blockDim.y + 2);
    Real* walls = prev_values + blockDim.x * blockDim.y;

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= N) return;

    prev_values[threadIdx.y * blockDim.x + threadIdx.x] = U[x + y * M + N * M * ((t-1) % 3)];
    values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] = U[x + y * M + N * M * (t % 3)];
    walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] = U[x + y * M + N * M * 3];
    if (threadIdx.x == 0 && x>0) { 
        values[(threadIdx.y + 1) * (blockDim.x + 2)] = U[x - 1 + y * M + N * M * (t % 3)];
        walls[(threadIdx.y + 1) * (blockDim.x + 2)] = U[x - 1 + y * M + N * M * 3];
    }
    if (threadIdx.x == blockDim.x - 1 && x<M-1) { 
        values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2] = U[x + 1 + y * M + N * M * (t % 3)];
        walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2] = U[x + 1 + y * M + N * M * 3];
    }
    if (threadIdx.y == 0 && y>0) {
        values[threadIdx.x + 1] = U[x + (y - 1) * M + N * M * (t % 3)]; 
        walls[threadIdx.x + 1] = U[x + (y - 1) * M + N * M * 3];
    }
    if (threadIdx.y == blockDim.y - 1 && y<N-1) { 
        values[(threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1] = U[x + (y + 1) * M + N * M * (t % 3)]; 
        walls[(threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1] = U[x + (y + 1) * M + N * M * 3];
    }

    __syncthreads();

    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) { U[x + y * M + N * M * ((t + 1) % 3)] = Real(0.0); return; }
    if (x == x_s && y == y_s) {
        U[x + y * M + (t % 3) * N * M] += F[t - 1]; // Source
        values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] += F[t - 1]; // Source
    }

    Real val;
    if (walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] > Real(0.0)) {
        val = -4 * values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1];
        //val += (walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x] > Real(0.0)) ? values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x] : Real(0.0);
        //val += (walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2] > Real(0.0)) ? values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2] : Real(0.0);
        //val += (walls[(threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1] > Real(0.0)) ? values[(threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1] : Real(0.0);
        //val += (walls[threadIdx.y * (blockDim.x + 2) + threadIdx.x + 1] > Real(0.0)) ? values[threadIdx.y * (blockDim.x + 2) + threadIdx.x + 1] : Real(0.0);
        val += values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x];
        val += values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 2];
        val += values[(threadIdx.y + 2) * (blockDim.x + 2) + threadIdx.x + 1];
        val += values[threadIdx.y * (blockDim.x + 2) + threadIdx.x + 1];

        val *= (walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] * walls[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1]) * v * v * d_t * d_t / (d_x * d_x); // ???
        val += Real(2.0) * values[(threadIdx.y + 1) * (blockDim.x + 2) + threadIdx.x + 1] - prev_values[threadIdx.y * blockDim.x + threadIdx.x] * (Real(1.0) - d_t * b * Real(0.5));
        val /= (Real(1.0) + d_t * b * Real(0.5));
    }
    else { val = Real(0.0); }
    U[x + y * M + N * M * ((t + 1) % 3)] = val;
    if (x == x_ir && y == y_ir) IR[t - 1] = val; // (t-1) because we run with t+1
    //}
}


__global__ void WaveIterationKernelAlt0(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b)
{    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    Real* walls = U + M * N * 3 * 2;
    Real* frame = U + M * N * 3 * (t % 2);
    Real* nextframe = U + M * N * 3 * ((t + 1) % 2);
    const Real C = Real(0.5);
    if (x == x_s && y == y_s)
        frame[(x_s + y_s * M) * 3] += F[t - 1]; // Source

    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
        nextframe[(x + y * M) * 3] = Real(0.0);
        nextframe[(x + y * M) * 3 + 1] = Real(0.0);;
        nextframe[(x + y * M) * 3 + 2] = Real(0.0);
        return;
    }

    Real val1, val2, val3;
    Real Y, lY, bY;
    Real B, lB, bB;
    //Real R = Real(0.9);
    Real gradx, grady;
    Real vx, vy;
    Real lnf, bnf;

    B = walls[x + y * M];
    lB = walls[x - 1 + y * M];
    bB = walls[x - M + y * M];

    Y = (B > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
    lY = (lB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
    bY = (bB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);

    vx = frame[(x + y * M) * 3 + 1];
    vy = frame[(x + y * M) * 3 + 2];

    val1 = B * (frame[(x + y * M) * 3] - C *
        ((frame[(x + 1 + y * M) * 3 + 1] - vx) + (frame[(x + M + y * M) * 3 + 2] - vy)));

    nextframe[(x + y * M) * 3] = val1;

    __syncthreads();

    lnf = nextframe[(x - 1 + y * M) * 3];
    bnf = nextframe[(x - M + y * M) * 3];

    gradx = val1 - lnf;
    grady = val1 - bnf;
    val2 = B * lB * (vx - C * gradx)
        + (lB - B) * (B * lY + lB * Y) * (B * val1 + lB * lnf);
    val3 = B * bB * (vy - C * grady)
        + (bB - B) * (B * bY + bB * Y) * (B * val1 + bB * bnf);

    nextframe[(x + y * M) * 3 + 1] = val2;
    nextframe[(x + y * M) * 3 + 2] = val3;

    if (x == x_ir && y == y_ir)
        IR[t - 1] = nextframe[(x_ir + y_ir * M) * 3];

}

__global__ void WaveIterationKernelAltPart1(Real* frame, Real* nextframe, Real* walls, const Real C, const unsigned int N, const unsigned int M, const Real v, const Real d_x, const Real d_t, const Real b)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
  //      nextframe[(x + y * M) * 3] = Real(0.0);
  //      nextframe[(x + y * M) * 3 + 1] = Real(0.0);
  //      nextframe[(x + y * M) * 3 + 2] = Real(0.0);
        return;
    }

    Real val1;// , val2, val3;
    //Real Y, lY, bY;
    Real B;// , lB, bB;
    //Real R = Real(0.9);
    //Real gradx, grady;
    Real vx, vy;
    //Real lnf, bnf;

    B = walls[x + y * M];
    //lB = walls[x - 1 + y * M];
    //bB = walls[x - M + y * M];

    //Y = (B > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
    //lY = (lB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
    //bY = (bB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);

    vx = frame[(x + y * M) * 3 + 1];
    vy = frame[(x + y * M) * 3 + 2];

    val1 = B * (frame[(x + y * M) * 3] - C *
        ((frame[(x + 1 + y * M) * 3 + 1] - vx) + (frame[(x + M + y * M) * 3 + 2] - vy)));

    nextframe[(x + y * M) * 3] = val1;

    //__syncthreads();

    //lnf = nextframe[(x - 1 + y * M) * 3];
    //bnf = nextframe[(x - M + y * M) * 3];

    //gradx = val1 - lnf;
    //grady = val1 - bnf;
    //val2 = B * lB * (vx - C * gradx)
    //    + (lB - B) * (B * lY + lB * Y) * (B * val1 + lB * lnf);
    //val3 = B * bB * (vy - C * grady)
    //    + (bB - B) * (B * bY + bB * Y) * (B * val1 + bB * bnf);

    //nextframe[(x + y * M) * 3 + 1] = val2;
    //nextframe[(x + y * M) * 3 + 2] = val3;
}

__global__ void WaveIterationKernelAltPart2(Real* frame, Real* nextframe, Real* walls, const Real C, const unsigned int N, const unsigned int M, const Real v, const Real d_x, const Real d_t, const Real b)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x + 1 >= M || y + 1 >= N || x < 1 || y < 1) {
        //nextframe[(x + y * M) * 3] = Real(0.0);
        //nextframe[(x + y * M) * 3 + 1] = Real(0.0);;
        //nextframe[(x + y * M) * 3 + 2] = Real(0.0);
        return;
    }

    Real val1, val2, val3;
    Real Y, lY, bY;
    Real B, lB, bB;
    //Real R = Real(0.9);
    Real gradx, grady;
    Real vx, vy;
    Real lnf, bnf;

    B = walls[x + y * M];
    lB = walls[x - 1 + y * M];
    bB = walls[x - M + y * M];

    Y = (B > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
    lY = (lB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);
    bY = (bB > Real(0.0)) ? Real(1.0) : (Real(1.0) - b) * (Real(1.0) + b);

    vx = frame[(x + y * M) * 3 + 1];
    vy = frame[(x + y * M) * 3 + 2];

    //val1 = B * (frame[(x + y * M) * 3] - C *
    //    ((frame[(x + 1 + y * M) * 3 + 1] - vx) + (frame[(x + M + y * M) * 3 + 2] - vy)));

    //nextframe[(x + y * M) * 3] = val1;
    val1 = nextframe[(x + y * M) * 3];

    lnf = nextframe[(x - 1 + y * M) * 3];
    bnf = nextframe[(x - M + y * M) * 3];

    gradx = val1 - lnf;
    grady = val1 - bnf;
    val2 = B * lB * (vx - C * gradx)
        + (lB - B) * (B * lY + lB * Y) * (B * val1 + lB * lnf);
    val3 = B * bB * (vy - C * grady)
        + (bB - B) * (B * bY + bB * Y) * (B * val1 + bB * bnf);

    nextframe[(x + y * M) * 3 + 1] = val2;
    nextframe[(x + y * M) * 3 + 2] = val3;

}

__global__ void WaveIterationKernelAlt(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, int blocksize, const Real b)
{
    Real* walls = U + M * N * 3 * 2;
    Real* frame = U + M * N * 3 * (t % 2);
    Real* nextframe = U + M * N * 3 * ((t + 1) % 2);

    const Real C = Real(0.5);
    frame[(x_s + y_s * M) * 3] += F[t - 1];

    dim3 blockDim(blocksize, blocksize, 1);
    dim3 gridDim(M / blocksize + ((M % blocksize == 0) ? 0 : 1), N / blocksize + ((N % blocksize == 0) ? 0 : 1), 1);

    WaveIterationKernelAltPart1 << <gridDim, blockDim >> > (frame, nextframe, walls, C, N, M, v, d_x, d_t, b);
    cudaDeviceSynchronize();
    WaveIterationKernelAltPart2 << <gridDim, blockDim >> > (frame, nextframe, walls, C, N, M, v, d_x, d_t, b);
    cudaDeviceSynchronize();

    IR[t - 1] = nextframe[(x_ir + y_ir * M) * 3];
}