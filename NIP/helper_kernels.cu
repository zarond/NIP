//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <helper_cuda.h>
//#include <helper_math.h>
//#include <vector_types.h>
#include "helper_kernels.cuh"
//typedef Real Real;

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((unsigned char)vec.x, (unsigned char)vec.y, (unsigned char)vec.z, (unsigned char)vec.w);
}

__device__ __inline__ float4 colorramp(float v, float w = 0.0f) {
    if (v >= 0) {
        v = 1 - exp(-v);
        return make_float4(0.0, 0.0, 0.0, 1.0)*(1-v) + v*make_float4(1.0, 0.0, 0.0, 1.0) + make_float4(0.0, 1.0f - w, 0.0, 0.0);
    }
    else{
        v = -1 + exp(v);
        return make_float4(0.0, 0.0, 0.0, 1.0)*(1+v) + (-v)*make_float4(0.0, 0.0, 1.0, 1.0) + make_float4(0.0, 1.0f -w, 0.0, 0.0);
    }
}

__global__ void dataToColormap(uchar4* d_output, unsigned int imageW, unsigned int imageH, Real* dev_Input, int frame, Real intensity, int wall) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageW || y >= imageH) return;
    unsigned int i = y * imageW + x;// 
    float v = dev_Input[i + imageW * imageH * frame] * intensity;
    float4 color = colorramp(v, dev_Input[i + imageW * imageH * wall]);//make_float4(v, v, v, 1.0);
    //if (dev_Input[i + imageW * imageH * 3] <= 0.5) color = make_float4(0.0f,1.0f,0.0f,1.0f);
    d_output[i] = to_uchar4(color * 255.0);
}

__global__ void dataToColormap2(uchar4* d_output, unsigned int imageW, unsigned int imageH, Real* dev_Input, int frame, Real intensity, int wall) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageW || y >= imageH) return;
    unsigned int i = y * imageW + x;// 
    float v = dev_Input[i * 3 + 3 * imageW * imageH * frame] * intensity;
    float4 color = colorramp(v, dev_Input[i + imageW * imageH * 2 * 3]);//make_float4(v, v, v, 1.0);
    //if (dev_Input[i + imageW * imageH * 3] <= 0.5) color = make_float4(0.0f,1.0f,0.0f,1.0f);
    d_output[i] = to_uchar4(color * 255.0);
}