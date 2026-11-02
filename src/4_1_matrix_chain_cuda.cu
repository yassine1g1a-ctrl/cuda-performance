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
#define TOL  1e-9

__global__ void matmul_kernel(const double *A, const double *B, double *C, int M, int P)
{
    __shared__ double total;
    int i = blockIdx.y;
    int j = blockIdx.x;

    if (threadIdx.x == 0) total = 0.0;
    __syncthreads();

    double tmp = 0.0;
    for (int k = threadIdx.x; k < P; k += blockDim.x)
        tmp += A[(long long)i * P + k] * B[(long long)k * M + j];

    atomicAdd(&total, tmp);
    __syncthreads();

    if (threadIdx.x == 0) C[(long long)i * M + j] = total;
}

void produit(const double *A, const double *B, double *C, int N, int M, int P, int T, cudaStream_t flux)
{
    dim3 grille(M, N);
    matmul_kernel<<<grille, T, 0, flux>>>(A, B, C, M, P);
}

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
    int nthreads = 32;                           // threads par bloc : meilleur valeur d'apres mesures partie 3
    int nstreams = 4;
	int i,j;
	double *h_A;
	double *d_in[8], *d_AB, *d_CD, *d_EF, *d_GH, *d_ABCD, *d_EFGH, *d_R, *h_R;
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
        } else if ( ( strcmp( argv[ i ], "-S" ) == 0 )) {
            nstreams = atoi( argv[ ++i ] );
            printf( "  User nstreams is %d\n", nstreams );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Matrix multiplication Options:\n" );
            printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
            printf( "  -M <int>:              Size of the dimension M (by default 1000)\n" );
            printf( "  -P <int>:              Size of the dimension P (by default 1000)\n" );
            printf( "  -T <int>:              Number of threads per block (by default 32)\n" );
            printf( "  -S <int>:              Number of streams (by default 4)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }

    size_t octets = (size_t)Ndim * Ndim * sizeof(double);

    h_A = (double *)malloc(octets);
    h_R = (double *)malloc(octets);
    for (size_t x = 0; x < (size_t)Ndim * Ndim; x++)
        h_A[x] = AVAL;

    double **tous[] = { &d_AB, &d_CD, &d_EF, &d_GH, &d_ABCD, &d_EFGH, &d_R };
    for (int m = 0; m < 8; m++) {
        if (cudaMalloc(&d_in[m], octets) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
        cudaMemcpy(d_in[m], h_A, octets, cudaMemcpyHostToDevice);
    }
    for (int m = 0; m < 7; m++)
        if (cudaMalloc(tous[m], octets) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }

    cudaStream_t flux[4];
    for (int s = 0; s < nstreams; s++) cudaStreamCreate(&flux[s]);
    cudaEvent_t fini[6];                         // AB, CD, EF, GH, ABCD, EFGH
    for (int e = 0; e < 6; e++) cudaEventCreateWithFlags(&fini[e], cudaEventDisableTiming);

    // Timer products.
    auto begin = std::chrono::high_resolution_clock::now();

    // 4 produits independants
    produit(d_in[0], d_in[1], d_AB, Ndim, Mdim, Pdim, nthreads, flux[0 % nstreams]);
    cudaEventRecord(fini[0], flux[0 % nstreams]);
    produit(d_in[2], d_in[3], d_CD, Ndim, Mdim, Pdim, nthreads, flux[1 % nstreams]);
    cudaEventRecord(fini[1], flux[1 % nstreams]);
    produit(d_in[4], d_in[5], d_EF, Ndim, Mdim, Pdim, nthreads, flux[2 % nstreams]);
    cudaEventRecord(fini[2], flux[2 % nstreams]);
    produit(d_in[6], d_in[7], d_GH, Ndim, Mdim, Pdim, nthreads, flux[3 % nstreams]);
    cudaEventRecord(fini[3], flux[3 % nstreams]);

    // ABCD attend AB et CD, EFGH attend EF et GH
    cudaStreamWaitEvent(flux[0], fini[0]);
    cudaStreamWaitEvent(flux[0], fini[1]);
    produit(d_AB, d_CD, d_ABCD, Ndim, Mdim, Pdim, nthreads, flux[0]);
    cudaEventRecord(fini[4], flux[0]);

    cudaStreamWaitEvent(flux[1 % nstreams], fini[2]);
    cudaStreamWaitEvent(flux[1 % nstreams], fini[3]);
    produit(d_EF, d_GH, d_EFGH, Ndim, Mdim, Pdim, nthreads, flux[1 % nstreams]);
    cudaEventRecord(fini[5], flux[1 % nstreams]);

    // on attend ABCD et EFGH
    cudaStreamWaitEvent(flux[0], fini[4]);
    cudaStreamWaitEvent(flux[0], fini[5]);
    produit(d_ABCD, d_EFGH, d_R, Ndim, Mdim, Pdim, nthreads, flux[0]);

    cudaError_t erreur = cudaGetLastError();
    if (erreur != cudaSuccess) { printf("Erreur CUDA : %s\n", cudaGetErrorString(erreur)); return 4; }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - begin).count();

    cudaMemcpy(h_R, d_R, octets, cudaMemcpyDeviceToHost);

	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

      dN = (double)Ndim;
      dM = (double)Mdim;
      dP = (double)Pdim;
      mflops = 7.0 * 2.0 * dN * dM * dP / (1000000.0 * time);

	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);
	std::ofstream("stats_cuda_chaine.csv", std::ios::app) << "4_1_matrix_chain_cuda," << nstreams << "," << Ndim << "," << Mdim << "," << Pdim << "," << time << "," << mflops << "\n";

	cval = pow((double)Ndim, 7) * pow(AVAL, 8);
	maxrel = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			err = fabs( *(h_R+(size_t)i*Mdim+j) - cval ) / cval;
			if (err > maxrel) maxrel = err;
		}
	}

	if (maxrel > TOL)
		printf("\n Errors in multiplication");
	else
		printf("\n Hey, it worked");

    // on libère tout
	for (int m = 0; m < 8; m++) cudaFree(d_in[m]);
	for (int m = 0; m < 7; m++) cudaFree(*tous[m]);
	for (int s = 0; s < nstreams; s++) cudaStreamDestroy(flux[s]);
	for (int e = 0; e < 6; e++) cudaEventDestroy(fini[e]);
	free(h_A); free(h_R);


	printf("\n all done \n");
}
