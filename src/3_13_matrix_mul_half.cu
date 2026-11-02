/*
**  PROGRAM: Matrix Multiply
**
**  PURPOSE: This is a simple matrix multiply program. 
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**
**  HISTORY: Written by Tim Mattson, Nov 1999.
*            Modified and extended by Jonathan Rouzaud-Cornabas, Oct 2022
*/


#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cmath>
#include <cuda_fp16.h>

#define AVAL 3.14f
#define BVAL 5.42f
#define TOL  0.05

__global__ void matmul_kernel(const __half *A, const __half *B, __half *C, int M, int P)
{
    __shared__ __half total;
    int i = blockIdx.y;
    int j = blockIdx.x;

    if (threadIdx.x == 0) total = __float2half(0.0f);
    __syncthreads();

    __half tmp = __float2half(0.0f);
    for (int k = threadIdx.x; k < P; k += blockDim.x)
        tmp = __hadd(tmp, __hmul(A[(long long)i * P + k], B[(long long)k * M + j]));

    atomicAdd(&total, tmp);
    __syncthreads();

    if (threadIdx.x == 0) C[(long long)i * M + j] = total;
}

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
    int nthreads = 128;                          // threads par bloc
	int i,j;
	__half *A, *B, *C;
	double cval, err, maxrel;
    double dN, dM, dP, mflops;

    // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
            Ndim = atoi( argv[ ++i ] );
            printf( "  User N is %d\n", Ndim );
        } else if ( ( strcmp( argv[ i ], "-M" ) == 0 )) {
            Mdim = atoi( argv[ ++i ] );
            printf( "  User M is %d\n", Mdim );
        } else if ( ( strcmp( argv[ i ], "-P" ) == 0 )) {
            Pdim = atoi( argv[ ++i ] );
            printf( "  User P is %d\n", Pdim );
        } else if ( ( strcmp( argv[ i ], "-T" ) == 0 )) {
            nthreads = atoi( argv[ ++i ] );
            printf( "  User nthreads is %d\n", nthreads );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Matrix multiplication Options:\n" );
            printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
            printf( "  -M <int>:              Size of the dimension M (by default 1000)\n" );
            printf( "  -P <int>:              Size of the dimension P (by default 1000)\n" );
            printf( "  -T <int>:              Number of threads per block (by default 128)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }


      A = (__half *)malloc((size_t)Ndim*Pdim*sizeof(__half));
      B = (__half *)malloc((size_t)Pdim*Mdim*sizeof(__half));
      C = (__half *)malloc((size_t)Ndim*Mdim*sizeof(__half));

	/* Initialize matrices */

	for (i=0; i<Ndim; i++)
		for (j=0; j<Pdim; j++)
			*(A+((size_t)i*Pdim+j)) = __float2half(AVAL);

	for (i=0; i<Pdim; i++)
		for (j=0; j<Mdim; j++)
			*(B+((size_t)i*Mdim+j)) = __float2half(BVAL);

	for (i=0; i<Ndim; i++)
		for (j=0; j<Mdim; j++)
			*(C+((size_t)i*Mdim+j)) = __float2half(0.0f);

	/* Do the matrix product */

    __half *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, (size_t)Ndim * Pdim * sizeof(__half)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
    if (cudaMalloc(&d_B, (size_t)Pdim * Mdim * sizeof(__half)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
    if (cudaMalloc(&d_C, (size_t)Ndim * Mdim * sizeof(__half)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
    cudaMemcpy(d_A, A, (size_t)Ndim * Pdim * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size_t)Pdim * Mdim * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 grille(Mdim, Ndim);

    // Timer products.
    auto begin = std::chrono::high_resolution_clock::now();

    matmul_kernel<<<grille, nthreads>>>(d_A, d_B, d_C, Mdim, Pdim);
    cudaError_t erreur = cudaGetLastError();
    if (erreur != cudaSuccess) { printf("Erreur CUDA : %s\n", cudaGetErrorString(erreur)); return 4; }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - begin).count();

    cudaMemcpy(C, d_C, (size_t)Ndim * Mdim * sizeof(__half), cudaMemcpyDeviceToHost);

	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

      dN = (double)Ndim;
      dM = (double)Mdim;
      dP = (double)Pdim;
      mflops = 2.0 * dN * dM * dP / (1000000.0 * time);

	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);
	std::ofstream("stats_cuda_matrices.csv", std::ios::app) << "3_13_matrix_mul_half," << nthreads << "," << Ndim << "," << Mdim << "," << Pdim << "," << time << "," << mflops << "\n";

	cval = (double)Pdim * 3.14 * 5.42;
	maxrel = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			err = fabs( (double)__half2float(*(C+(size_t)i*Mdim+j)) - cval ) / cval;
			if (err > maxrel) maxrel = err;
		}
	}

	if (maxrel > TOL)
		printf("\n Errors in multiplication");
	else
		printf("\n Hey, it worked");

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	free(A); free(B); free(C);

	printf("\n all done \n");
}
