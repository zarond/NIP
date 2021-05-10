#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <vector_types.h>
//#include "types.h"
typedef float Real;
//typedef double Real;

__global__ void WaveIterationKernel(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b);
__global__ void MultipleIterationsKernel(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real, const unsigned int iterations);
__global__ void WaveIterationSimple(Real* U, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, const Real b);
__global__ void WaveIterationKernelSM(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, const Real b);
__global__ void WaveIterationKernelAlt(Real* U, Real* IR, const unsigned int N, const unsigned int M, const int t, const Real v, const Real d_x, const Real d_t, const int x_ir, const int y_ir, const int x_s, const int y_s, Real* F, int blocksize=32, const Real b=Real(0.0));
