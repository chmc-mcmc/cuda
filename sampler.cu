 #include <time.h>
#define CHAINS  (BLOCKS*THREADS)

#define AT(aa,ii,jj) (*(aa+(ii)*(DIM+1)+jj))
__device__ int solve(float *a, float *x)
{
  for(int j=0; j<DIM; j++) {
    for(int i=0; i<DIM; i++) {
      if(i!=j) {
        float b=AT(a,i,j)/AT(a,j,j);
        for(int k=0; k<=DIM; k++) { 
          AT(a,i,k)=AT(a,i,k)-b*AT(a,j,k);
        }
      }
    }
  }
  for(int i=0; i<DIM; i++) {
    x[i]=AT(a,i,DIM)/AT(a,i,i);
  }
  return 0;
}
__device__ inline float clip(float v, float lower, float upper)
{
  if(v<lower)
    return lower;
  else if(v>upper)
    return upper;
  else
    return v;

}
__device__ float dot(const float* x, const float *y, const int n)
{
	float sum=0.0;
	for (int i=0;i<n;i++)
		sum += x[i]*y[i];
	return sum;
}

struct SHARE{
   float cache[CHAINS];
   float s[CHAINS];
   float S[CHAINS];
  float h[CHAINS][DIM][DIM];
  float a[CHAINS][DIM][DIM+1];
  float q[CHAINS][DIM];
  float p[CHAINS][DIM];
  float x[CHAINS][DIM];
  float q0[CHAINS][DIM];
  float q1[CHAINS][DIM];
  float UE[CHAINS][STEPS+1];
   float dt;
   float scale;
   bool vanilla;
  float Htotal1;
  float Htotal2;
  float dt1;
  float dt2;
  float Htotal;
  float KtotalNew;
  float Utotal;
  curandState state[CHAINS];
};
__global__ void sample(float *QS, float *qinit, SHARE *shr, long seed)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, shr->state + id);

  //1. init qAll
  for(int i = 0; i < DIM; i++){
    if(USE_INIT){
      shr->q[id][i] = qinit[i];
    }else{
      shr->q[id][i] = curand_normal(shr->state + id);
    }
    
  }
  //2. Utotal
  shr->cache[id] = U(shr->q[id]);
  __syncthreads();
  if (id == 0){
    float Utotal = 0;
    for(int i=0;i< blockDim.x ;i++){
      Utotal += shr->cache[i+ blockIdx.x * blockDim.x];
    }
    shr->Htotal1 = 2*Utotal;
    shr->Htotal2 = 2*Utotal;
    shr->dt1 = DT0;
    shr->dt2 = DT0;
    shr->vanilla = true;
  }
  
  //3.
  
  for(int j=0; j<EPISODE;j++){
    __syncthreads();
    //4.
    for(int i = 0; i < DIM; i++){
      shr->p[id][i] = curand_normal(shr->state + id);
    }

    //5.
    if(shr->vanilla){
      shr->cache[id] = dot(shr->p[id], shr->p[id], DIM)/2.;
    }else{
      ddU(shr->q[id], shr->h[id][0]);
      for(int i = 0; i < DIM; i++){
        for(int j=0; j < DIM; j++){
          shr->a[id][i][j]=shr->h[id][i][j];
        }
        shr->a[id][i][DIM] = shr->p[id][i];
      }
      solve(shr->a[id][0], shr->x[id]);
      shr->cache[id] = dot(shr->p[id], shr->x[id], DIM) / 2.;
    }
    __syncthreads();
    
    if (id == 0){
      float ktn = 0;
      for(int i=0;i<blockDim.x;i++){
        ktn += shr->cache[i+blockIdx.x*blockDim.x];
      }
      shr->KtotalNew = ktn;
    }
    //6.
    shr->cache[id] = U(shr->q[id]);
    __syncthreads();
    if (id == 0){
      float Utotal = 0;
      for(int i=0;i<blockDim.x;i++){
        Utotal += shr->cache[i+blockIdx.x*blockDim.x];
      }
      shr->Utotal = Utotal;
      if(shr->vanilla){
        shr->Htotal=shr->Htotal1;
        shr->dt = shr->dt1;
      }else{
        shr->Htotal=shr->Htotal2;
        shr->dt = shr->dt2;
      }
      //7.
      float Ktotal = shr->Htotal - shr->Utotal;
      shr->scale = sqrt(abs(Ktotal/shr->KtotalNew));
    }
    __syncthreads();
    //8.
    for(int i=0;i<DIM;i++){
      shr->p[id][i]=shr->p[id][i]*shr->scale;
    }
    //9.
    bool bad = false;
    //10.

    memcpy(shr->q0[id], shr->q[id], DIM*sizeof(float));

    shr->UE[id][0] = U(shr->q[id]);
    //11.
    for(int k = 0; k < STEPS; k++){
      float dudq[DIM];
      dU(shr->q[id],dudq);
      for(int h = 0; h < DIM; h++){
        shr->p[id][h] -= shr->dt * dudq[h];
      }
      //float q1[DIM];
      memcpy(shr->q1[id],shr->q[id],sizeof(float)*DIM);
      //12.
      if(shr->vanilla){
        for(int h = 0; h < DIM; h++){
          shr->q[id][h]+=shr->dt*shr->p[id][h];
        }
      }else{

        ddU(shr->q[id], shr->h[id][0]);
        for(int i = 0; i < DIM; i++){
          for(int j=0; j < DIM; j++){
            shr->a[id][i][j]=shr->h[id][i][j];
          }
          shr->a[id][i][DIM] = shr->p[id][i];
        }
        solve(shr->a[id][0], shr->x[id]);
        for(int h=0;h<DIM;h++){
          shr->q[id][h] += shr->dt *shr->x[id][h];
        }
      }
      if(outbnd(shr->q[id])){
        //printf("bad");
        bad = true;
        memcpy(shr->q[id],shr->q1[id],sizeof(float)*DIM);
      }
      //printf("%f %f\t",shr->q[id][0],shr->q[id][0]);
      //13.
      if(j<BURNIN){
        shr->UE[id][k+1]=U(shr->q[id]);
      }
    }

    //14. compute Si and si
    if(j<BURNIN){
      int mini = 0;
      int maxi = 0;
      for(int k=1; k<= STEPS; k++){
        if(shr->UE[id][k]>shr->UE[id][maxi])
          maxi = k;
        else if(shr->UE[id][k]<shr->UE[id][mini])
          mini = k;
      }
      shr->s[id] = mini;
      shr->S[id] = maxi;
    }
    float alpha = 0.0;
    if(!bad){
      alpha = exp(clip(U(shr->q0[id])-U(shr->q[id]),-200.0,0.0));
      //printf("%f\n", alpha);
    }
    shr->cache[id]=alpha;
    //15.
    if(alpha < curand_uniform(shr->state + id)){
      memcpy(shr->q[id], shr->q0[id], DIM*sizeof(float));
    }
    if(j>=BURNIN){
      memcpy(QS+(j-BURNIN)*gridDim.x*blockDim.x*DIM + id*DIM, shr->q[id], DIM*sizeof(float));
      //printf("%f\n",shr->q[id][0]);
    }
    //16.
    __syncthreads();
    //tune
    if (id == 0 && j < BURNIN){
      int s0=0,s1=0,S0=0,S1=0;
      float ap = 0;
      for(int k = 0; k< blockDim.x; k++){
        ap += shr->cache[k];
        if(shr->s[k]==0)
          s0++;
        if(shr->s[k]==STEPS)
          s1++;
        if(shr->S[k]==0)
            S0++;
        if(shr->S[k]==STEPS)
          S1++;
      }
      ap /= blockDim.x;
      if(s0==blockDim.x && S1==blockDim.x){
        //too large
        shr->dt = shr->dt/(1+decaydt);
      }else if(s0+s1==blockDim.x && S0+S1==blockDim.x){
        //too small
        shr->dt = shr->dt*(1+decaydt);
      }
      if(ap > 0.9){
        // too high
        shr->Htotal = (shr->Htotal-shr->Utotal)*(1+decayenergy)+shr->Utotal;
      }else if(ap <0.1){
        // too low
        shr->Htotal = (shr->Htotal-shr->Utotal)/(1+decayenergy)+shr->Utotal;
      }
      if(shr->vanilla){
        shr->Htotal1=shr->Htotal;
        shr->dt1 = shr->dt;
      }else{
        shr->Htotal2=shr->Htotal;
        shr->dt2 = shr->dt;
      }
      if(SWITCH){
        shr->vanilla = !shr->vanilla;
      }
    }
    __syncthreads();
  }
}

int main()
{
  cudaSetDevice(0);
  cudaDeviceProp  prop;

  int count;
  HANDLE_ERROR( cudaGetDeviceCount(&count) );
  HANDLE_ERROR( cudaGetDeviceProperties(&prop, 0) );

  int M = CHAINS * (EPISODE-BURNIN);
  float *QS = (float*) malloc(M * DIM * sizeof(float));

  float * dev_QS;
  float * dev_qinit;
  SHARE * dev_share;
  HANDLE_ERROR(cudaMalloc((void**)&dev_QS, M * DIM * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_qinit, DIM * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_share, sizeof(SHARE)));
  
  if(USE_INIT){
    HANDLE_ERROR(cudaMemcpy(dev_qinit, qinit, DIM*sizeof(float), cudaMemcpyHostToDevice));
  }
  srand((unsigned int)time(NULL));

  clock_t start, end;
  double cpu_time_used;
     
  start = clock();

  sample<<<BLOCKS,THREADS>>>(dev_QS, dev_qinit, dev_share, rand());

  HANDLE_ERROR(cudaMemcpy(QS, dev_QS, M*DIM*sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
 // usefull?
  cudaDeviceReset();

 end = clock();
 cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
 printf("%f seconds",cpu_time_used);
 FILE *fp = fopen("QS.csv", "w");
  for(int i=0;i<M;i++){
    for(int j=0;j<DIM;j++)
      if(j<DIM-1)
        fprintf(fp,"%f,",QS[i*DIM+j]);
      else
        fprintf(fp,"%f\n",QS[i*DIM+j]);
  }
  fclose(fp);

  free(QS);
 // useful?
 cudaFree(dev_QS);
 cudaFree(dev_share);
 cudaFree(dev_qinit);  
return 0;

}
