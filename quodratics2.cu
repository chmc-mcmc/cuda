// cuda by examples
#include "../common/book.h"
#include <curand_kernel.h>
#include<stdio.h>
#include<stdlib.h>

#define EPISODE 10000
#define BURNIN  9990
#define DIM     2
// my laptop 1050 Multiprocessor count:  5
#define BLOCKS 5
// my laptop 1050:Max threads per block:  1024
#define THREADS 1024
#define STEPS 10
#define DT0 0.000000001
#define SWITCH true
#define decaydt 0.1
#define decayenergy 0.1
#define USE_INIT true

const float rho = 0.9999999;
float qinit[DIM]={0.1,0.1};

__device__ float U(float* q)
{
  float x=q[0];
  float y=q[1];
  return 0.5*(x*x+y*y-2.*x*y*rho)/(1.-rho*rho);
}
__device__ void dU(float*q, float*dudq)
{
  float x=q[0];
  float y=q[1];
  dudq[0]=0.5*(2.*x-2.*y*rho)/(1.-rho*rho);
  dudq[1]=0.5*(2.*y-2.*x*rho)/(1.-rho*rho);
}
__device__ void ddU(float* q,float* h)
{
  h[0] = 1./(1.-rho*rho);
  h[1] = -rho/(1.-rho*rho);
  h[2] = -rho/(1.-rho*rho);
  h[3] = 1./(1.-rho*rho);
}
__device__ bool outbnd(float* q)
{
  return false;
}
#include "sampler.cu"
