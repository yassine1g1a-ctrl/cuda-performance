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
static int nblocks;               // calculé : ceil(N / (T * S))
double step;

__global__ void pi_kernel(double *result, long long num_steps, double step, long long steps_per_thread)
{
    __shared__ double block_total;     // 1 copie / bloc en mémoire partagée

    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long start = id * steps_per_thread + 1;
    long long end   = start + steps_per_thread;

    if (threadIdx.x == 0) block_total = 0.0;       // un seul thread l'initialise
    __syncthreads();                               // attendre que tous les threads du bloc aient initialisé block_total

    double sum = 0.0;
    for (long long i = start; i < end && i <= num_steps; i++) {
        double x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    atomicAdd(&block_total, sum);                  // chaque thread ajoute sa somme au total du bloc
    __syncthreads();                               // attendre les threads

    if (threadIdx.x == 0)
        atomicAdd(result, block_total);            // total ajouter au résultat global
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
      long long steps_per_thread = spt;

      //D'abord on alloue le tableau des sommes partielles sur le GPU
      double *d_result;
      if (cudaMalloc(&d_result, sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
      cudaMemset(d_result, 0, sizeof(double));      // initialisation à 0


      auto begin = std::chrono::high_resolution_clock::now();

      //  Puis on peut lancer le kernel
      pi_kernel<<<nblocks, nthreads>>>(d_result, num_steps, step, steps_per_thread);

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) { printf("Erreur CUDA : %s\n", cudaGetErrorString(err)); return 4; }

      // On copie le résultat global
      double sum;
      cudaMemcpy(&sum, d_result, sizeof(double), cudaMemcpyDeviceToHost);
      double pi = step * sum;

      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end - begin).count();

      printf("\n pi with %lld steps is %lf in %lf seconds\n ", num_steps, pi, time);

      std::ofstream("stats_cuda_pi.csv", std::ios::app)
          << "1_3_pi_cuda," << num_steps << "," << nthreads << "," << spt << "," << nblocks << "," << time << "\n";

      cudaFree(d_result);
      return 0;
}
