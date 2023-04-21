#include <stdio.h>
#include <time.h>

#include "uk.h"

#define M 8
#define N 12
#define K 512
static float A[K * M];
static float B[K * N];
static float C[N * M];
static float C2[N * M];
static float C3[N * M];
static float C4[N * M];

#define Aref(a1,a2)  A[ (a2)*(Alda)+(a1) ]
#define Bref(a1,a2)  B[ (a2)*(Blda)+(a1) ]
#define Cref(a1,a2)  C3[ (a2)*(Clda)+(a1) ]


void simplegemm(){
   int Alda = M, Clda =  M;
   int Blda = N;   
   int    i, j, p;
   for ( p=0; p<K; p++ )
	   for ( j=0; j<N; j++ )
		   for ( i=0; i<M; i++ )
			   Cref(i,j) = Cref(i,j) + Aref(i,p) * Bref(j,p);
/*   for (int k=0; k<K;k++)
     for(int j=0; j<N;j++)
      for(int i = 0; i < M; i++)
	   C3[j*M+i] += A[k*M+i] * B[j*K+k];*/
}

void initialize() {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i * K + j] = (i * K + j)*0.1;//3.2;
    }
  }
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i * N + j] = (i * N + j)*0.2;
    }
  }
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0.0;
      C2[i * N + j] = 0.0;
      C3[i * N + j] = 0.0;
      C4[i * N + j] = 0.0;
    }
  }
  return;
}

int main() {
  clock_t start, end;
  float msec;
  int reps=1;
//int M=8, N=12;
  initialize();
  double gflops = (2.0*M*N*K)/1e9;
  // Calling original matmul
  start = clock();
    //base_ukernel(NULL, M, N, K, C, A, B);
  for (int i = 0; i < reps; i++)
    example_sgemm(NULL, K, C, A, B);
  end = clock();
  
  msec = ((double)(end - start) / (double)CLOCKS_PER_SEC)/(1.0*reps);
  double tt =  msec;
  double gf = gflops/tt;
  printf("Time taken for original matmul: %f seconds -> %f gflops\n",
      tt, gf);

  // Calling scheduled matmul
  start = clock();
  for (int i = 0; i < reps; i++)
    uk_8x12(NULL, K, C2, A,B); //, M,1,N,1);
    //uk_8x12(NULL, K, (struct exo_win_2f32){C2,{1,M}}, (struct exo_win_2f32c){A,{1,M}}, (struct exo_win_2f32c){B,{1,K}}); //, M,1,N,1);
  end = clock();

  //msec = (end - start) * 1000 / CLOCKS_PER_SEC;
  msec = ((double)(end - start) / (double) CLOCKS_PER_SEC)/reps;
  //printf("Time taken for scheduled matmul: %d seconds %d milliseconds\n",
  tt =  msec;
  gf = gflops/tt;
  printf("Time taken for scheduled matmul: %f seconds -> %f gflops\n",
      tt, gf);
 /* 
  start = clock();
  for (int i = 0; i < reps; i++)
    uk_8x12_windowed(NULL, K, (struct exo_win_2f32){C3,{M,1}}, (struct exo_win_2f32c){A,{M,1}}, (struct exo_win_2f32c){B,{N,1}}); //, M,1,N,1);
  end = clock();

  //msec = (end - start) * 1000 / CLOCKS_PER_SEC;
  msec = ((double)(end - start) / (double) CLOCKS_PER_SEC)/reps;
  //printf("Time taken for scheduled matmul: %d seconds %d milliseconds\n",
  tt =  msec;
  gf = gflops/tt;
  printf("Time taken for scheduled matmul_windowed: %f seconds -> %f gflops\n",
      tt, gf);
  start = clock();
  for (int i = 0; i < reps; i++)
    uk_assert_8x12(NULL, K, (struct exo_win_2f32){C4,{M,1}}, (struct exo_win_2f32c){A,{M,1}}, (struct exo_win_2f32c){B,{N,1}});
  end = clock();

  //msec = (end - start) * 1000 / CLOCKS_PER_SEC;
  msec = ((double)(end - start) / (double) CLOCKS_PER_SEC)/reps;
  //printf("Time taken for scheduled assert matmul: %d seconds %d milliseconds\n",
  tt =  msec;
  gf = gflops/tt;
  printf("Time taken for scheduled assert matmul_windowed: %f seconds -> %f gflops\n",
      tt, gf);
      //msec / 1000, msec % 1000);
      //
      */
  for (int i = 0; i < reps; i++)
  simplegemm();
  for(int i = 0; i< M; i++)
  for(int j = 0; j< N; j++){
	  //if(C[i* N + j]==C2[i*N+j] && C2[i*N+j] == C3[i*N+j])
	  if(C2[i* N + j]== C3[i*N+j])
		  continue;
	  else
	  	 printf("ERROR %f %f %f\n",C[i*N+j],C2[i*N+j],C3[i*N+j]);
          //printf("C[%d]=%f, C2[%d]=%f,C3[%d]=%f\n",i* N + j,C[i* N + j],i* N + j,C2[i* N + j],i* N + j,C3[i* N + j]);
  }
  printf("PERFECTO!\n");
  return (0);
}
