#include "../common/book.h"
#include <curand_kernel.h>
#include<stdio.h>
#include<stdlib.h>

#define EPISODE 10000
#define BURNIN   9990
#define DIM     8
#define BLOCKS 5
#define THREADS 1024
#define STEPS   5
#define DT0     0.00001
#define SWITCH  false
#define decaydt 0.1
#define decayenergy 0.1
#define USE_INIT true

const float qinit[DIM] ={0.1,0.1,0.1,0.1,0.1,0.1,13.600000000000001,0.1};
__device__ float U(float* q)
{
  float x5 = q[0];
  float x6 = q[1];
  float x7 = q[2];
  float x8 = q[3];
  float x10= q[4];
  float x11= q[5];
  float x13= q[6];
  float x14= q[7];
  return (1140. - x10 + 4*x11 + 8*x13 - 23*x14 + 10*x5 - 10*x6 + 73*x7 + 33*x8)/100.;
}
__device__ void dU(float*q, float*dudq)
{
  float g[] = {0.1,-0.1,0.73,0.33,-0.01,0.04,0.08,-0.23};
  for(int i=0;i<8;i++)
    dudq[i] = g[i];
}
__device__ void ddU(float* q,float* h)
{
}
__device__ bool outbnd(float* q)
{
  float x5 = q[0];
  float x6 = q[1];
  float x7 = q[2];
  float x8 = q[3];
  float x10= q[4];
  float x11= q[5];
  float x13= q[6];
  float x14= q[7];
  return 6 - x11 - x5 < 0 || 6 + x7 + x8 < 0 || 2 - x14 - x6 - x8 < 0 || x10 + x13 - x5 - x6 + x7 < 0 || 14 - x10 - x11 - x13 - x14 < 0 || -5 + x13 + x14 < 0 || x5 < 0 || x6 < 0 || x7 < 0 || x8 < 0 || x10 < 0 || x11 < 0 || x13 < 0 || x14 < 0;
}

#include "sampler.cu"
