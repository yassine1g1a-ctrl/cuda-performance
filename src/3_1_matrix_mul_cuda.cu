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

#define AVAL 3.14
#define BVAL 5.42
#define TOL  0.001

__global__ void matmul_kernel(const double *A, const double *B, double *C, int M, int P)
{
    int i = blockIdx.y;
    int j = blockIdx.x;
    double tmp = 0.0;
    for (int k = 0; k < P; k++)
        tmp += A[(long long)i * P + k] * B[(long long)k * M + j];
    C[(long long)i * M + j] = tmp;
}

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
	int i,j;
	double *A, *B, *C;
	double cval, err, errsq;
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
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Matrix multiplication Options:\n" );
            printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
            printf( "  -M <int>:              Size of the dimension M (by default 1000)\n" );
            printf( "  -P <int>:              Size of the dimension P (by default 1000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }


      A = (double *)malloc((size_t)Ndim*Pdim*sizeof(double));
      B = (double *)malloc((size_t)Pdim*Mdim*sizeof(double));
      C = (double *)malloc((size_t)Ndim*Mdim*sizeof(double));

	/* Initialize matrices */

	for (i=0; i<Ndim; i++)
		for (j=0; j<Pdim; j++)
			*(A+((size_t)i*Pdim+j)) = AVAL;

	for (i=0; i<Pdim; i++)
		for (j=0; j<Mdim; j++)
			*(B+((size_t)i*Mdim+j)) = BVAL;

	for (i=0; i<Ndim; i++)
		for (j=0; j<Mdim; j++)
			*(C+((size_t)i*Mdim+j)) = 0.0;

	/* Do the matrix product */

    double *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, (size_t)Ndim * Pdim * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
    if (cudaMalloc(&d_B, (size_t)Pdim * Mdim * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
    if (cudaMalloc(&d_C, (size_t)Ndim * Mdim * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
    cudaMemcpy(d_A, A, (size_t)Ndim * Pdim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (size_t)Pdim * Mdim * sizeof(double), cudaMemcpyHostToDevice);

    dim3 grille(Mdim, Ndim);

    // Timer products.
    auto begin = std::chrono::high_resolution_clock::now();

    matmul_kernel<<<grille, 1>>>(d_A, d_B, d_C, Mdim, Pdim);
    cudaError_t erreur = cudaGetLastError();
    if (erreur != cudaSuccess) { printf("Erreur CUDA : %s\n", cudaGetErrorString(erreur)); return 4; }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - begin).count();

    cudaMemcpy(C, d_C, (size_t)Ndim * Mdim * sizeof(double), cudaMemcpyDeviceToHost);

	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

      dN = (double)Ndim;
      dM = (double)Mdim;
      dP = (double)Pdim;
      mflops = 2.0 * dN * dM * dP / (1000000.0 * time);

	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);
	std::ofstream("stats_cuda_matrices.csv", std::ios::app) << "3_1_matrix_mul_cuda," << 1 << "," << Ndim << "," << Mdim << "," << Pdim << "," << time << "," << mflops << "\n";

	cval = (double)Pdim * 3.14 * 5.42;
	errsq = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			err = (double)*(C+(size_t)i*Mdim+j) - cval;
		    errsq += err * err;
		}
	}

	if (errsq > TOL)
		printf("\n Errors in multiplication: %f",errsq);
	else
		printf("\n Hey, it worked");

	cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
	free(A); free(B); free(C);

	printf("\n all done \n");
}
