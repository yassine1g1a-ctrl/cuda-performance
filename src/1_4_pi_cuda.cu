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

__global__ void pi_kernel(double *block_sum, long long num_steps, double step, long long steps_per_thread)
{
    extern __shared__ double sdata[];
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long start = id * steps_per_thread + 1;
    long long end   = start + steps_per_thread;
    double sum = 0.0;
    for (long long i = start; i < end && i <= num_steps; i++) {
        double x = (i - 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Réduction en arbre : on divise le nombre de threads actifs par 2 à chaque tour
    int nb_threads_actifs = blockDim.x / 2; // au 1er tour, la moitié des threads travaillent

    while (nb_threads_actifs > 0) {
        // On vérifie si le thread travaille à ce tour ?
        if (threadIdx.x < nb_threads_actifs) {
            int ma_case      = threadIdx.x;
            int case_voisine = threadIdx.x + nb_threads_actifs;
            double ma_valeur      = sdata[ma_case];
            double valeur_voisine = sdata[case_voisine];
            sdata[ma_case] = ma_valeur + valeur_voisine;
        }
        __syncthreads(); // Tout le monde attend avant le tour suivant
        nb_threads_actifs = nb_threads_actifs / 2; // moitié moins de threads travaillent
    }
    if (threadIdx.x == 0) block_sum[blockIdx.x] = sdata[0];
}

//second kernel pour additioner les résultats de chaque bloc
__global__ void reduce_kernel(const double *in, double *out, long long n)
{
    extern __shared__ double sdata[];
    long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Chaque thread lit une valeur et quand il n'y en a plus (id >= n), il met 0.
    if (id < n)
        sdata[threadIdx.x] = in[id];
    else
        sdata[threadIdx.x] = 0.0;
    __syncthreads();

    int nb_threads_actifs = blockDim.x / 2;
    while (nb_threads_actifs > 0) {
        if (threadIdx.x < nb_threads_actifs) {
            int ma_case      = threadIdx.x;
            int case_voisine = threadIdx.x + nb_threads_actifs;
            double ma_valeur      = sdata[ma_case];
            double valeur_voisine = sdata[case_voisine];
            sdata[ma_case] = ma_valeur + valeur_voisine;
        }
        __syncthreads();
        nb_threads_actifs = nb_threads_actifs / 2;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sdata[0];
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


      // On alloue deux tableaux de nblocks valeurs sur le GPU.
      // Ils servent d'entrée et de sortie, à tour de rôle.
      double *d_a, *d_b;
      if (cudaMalloc(&d_a, nblocks * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }
      if (cudaMalloc(&d_b, nblocks * sizeof(double)) != cudaSuccess) { printf("Erreur : memoire insuffisante\n"); return 3; }

      auto begin = std::chrono::high_resolution_clock::now();

    // Taille de la mémoire partagée par bloc
      size_t shmem = nthreads * sizeof(double);

      // calcul + réduction dans chaque bloc
      // Résultat : nblocks totaux dans d_a
      pi_kernel<<<nblocks, nthreads, shmem>>>(d_a, num_steps, step, steps_per_thread);

      // On réduit le tableau jusqu'à n'avoir plus qu'une valeur
      long long n = nblocks;              // nb de blocs restant à additionner
      double *entree = d_a;
      double *sortie = d_b;
      // Avec 1 thread par bloc, la réduction ne diminuerait jamais n (boucle infinie) :
      // les étapes de réduction utilisent au moins 32 threads (une puissance de 2)
      int rthreads = (nthreads >= 32) ? nthreads : 32;
      size_t rshmem = rthreads * sizeof(double);
      while (n > 1) {
          long long blocs = (n + rthreads - 1) / rthreads;     // chaque bloc avale rthreads valeurs
          reduce_kernel<<<blocs, rthreads, rshmem>>>(entree, sortie, n);
          n = blocs;                                       // il reste autant de valeurs que de blocs
          double *tmp = entree;                            // échange entrée/sortie
          entree = sortie;
          sortie = tmp;
      }
      // Le résultat final est dans entree[0]


      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) { printf("Erreur CUDA : %s\n", cudaGetErrorString(err)); return 4; }

      // On copie le résultat global
      double sum;
      cudaMemcpy(&sum, entree, sizeof(double), cudaMemcpyDeviceToHost);
      double pi = step * sum;

      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end - begin).count();

      printf("\n pi with %lld steps is %lf in %lf seconds\n ", num_steps, pi, time);

      std::ofstream("stats_cuda_pi.csv", std::ios::app)
          << "1_4_pi_cuda," << num_steps << "," << nthreads << "," << spt << "," << nblocks << "," << time << "\n";

      cudaFree(d_a);
      cudaFree(d_b);
      return 0;
}
