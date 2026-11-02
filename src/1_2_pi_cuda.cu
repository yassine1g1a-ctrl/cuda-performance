/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <climits>

static long long num_steps = 100000000;
static long long spt = 256;            // pas par thread (S)
static int nthreads = 128;        // threads par bloc (T)
static int nblocks;               // calcul : ceil(N / (T * S))
double step;

__global__ void pi_kernel(double *partial, double *block_sum, long long num_steps, double step, long long steps_per_thread)
{
    // id thread = num bloc * nb threads par bloc + num thread dans le bloc
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long start = id * steps_per_thread + 1;
    long long end   = start + steps_per_thread;
    double sum = 0.0;
    for (long long i = start; i < end && i <= num_steps; i++) {
        double x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    partial[id] = sum;                 // niv 0 : 1 sum / thread, en mémoire globale
    __syncthreads();                   // attendre tous les threads du bloc
    if (threadIdx.x == 0) {            // niv 1 : le thread 0 additionne les threads de son bloc
        double s = 0.0;
        for (int t = 0; t < blockDim.x; t++)
            s += partial[(long long)blockIdx.x * blockDim.x + t];
        block_sum[blockIdx.x] = s;
    }
}


int main (int argc, char** argv)
{

      // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atoll( argv[ ++i ] );
            printf( "  User num_steps is %lld\n", num_steps );
        } else if ( strcmp( argv[ i ], "-S" ) == 0 ) {
            spt = atoll( argv[ ++i ] );
            printf( "  User steps per thread is %lld\n", spt );
        } else if ( strcmp( argv[ i ], "-T" ) == 0 ) {
            nthreads = atoi( argv[ ++i ] );
            printf( "  User nthreads is %d\n", nthreads );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }

      step = 1.0 / (double) num_steps;
      long long nb = (num_steps + (long long)nthreads * spt - 1) / ((long long)nthreads * spt);
      if (nb > INT_MAX) {
          printf("Erreur : %lld blocs, au-dessus de la limite de la grille\n", nb);
          return 2;
      }
      nblocks = (int) nb;
      long long total_threads = (long long)nblocks * nthreads;
      long long steps_per_thread = spt;

      //D'abord on alloue le tableau des sommes partielles sur le GPU
      double *d_partial, *d_block_sum;
      if (cudaMalloc(&d_partial,   total_threads * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
      if (cudaMalloc(&d_block_sum, nblocks * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }

      auto begin = std::chrono::high_resolution_clock::now();

      //  Puis on peut lancer le kernel
      pi_kernel<<<nblocks, nthreads>>>(d_partial, d_block_sum, num_steps, step, steps_per_thread);

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) { printf("Erreur CUDA : %s\n", cudaGetErrorString(err)); return 4; }

      // On copie les résultats partiels du GPU vers le CPU
      double *h_partial = (double*) malloc(nblocks * sizeof(double));
      cudaMemcpy(h_partial, d_block_sum, nblocks * sizeof(double), cudaMemcpyDeviceToHost);

      //Puis on additionne les sommes partielles sur le CPU
      double sum = 0.0;
      for (int b = 0; b < nblocks; b++) sum += h_partial[b];
      double pi = step * sum;

      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end - begin).count();

      printf("\n pi with %lld steps is %lf in %lf seconds\n ", num_steps, pi, time);

      std::ofstream("stats_cuda_pi.csv", std::ios::app)
          << "1_2_pi_cuda," << num_steps << "," << nthreads << "," << spt << "," << nblocks << "," << time << "\n";

      cudaFree(d_partial);
      cudaFree(d_block_sum);
      free(h_partial);
      return 0;
}
