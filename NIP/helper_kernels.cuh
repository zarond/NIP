#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <vector_types.h>
//#include "types.h"

typedef float Real;
//typedef double Real;

__global__ void dataToColormap(uchar4* d_output, unsigned int imageW, unsigned int imageH, Real* dev_Input, int frame = 0, Real intensity = 1.0, int wall = 3);
__global__ void dataToColormap2(uchar4* d_output, unsigned int imageW, unsigned int imageH, Real* dev_Input, int frame = 0, Real intensity = 1.0, int wall = 3);