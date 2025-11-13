#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// calcula el brillo de una estrella dada su galaxia y su indice
__device__ int brilloDet(int g, int e)
{
    return (g * 3 + e * 7) % 10;
}

// kernel que calcula el brillo promedio de una galaxia
__global__ void galaxia_fase(int galaxiaId, int estrellasPorGalaxia)
{
    extern __shared__ int brillos[];

    int g = galaxiaId;
    int e = threadIdx.x;

    if (e < estrellasPorGalaxia)
    {
        brillos[e] = brilloDet(g, e);
    }

    // Sincronizar para asegurar que todos los hilos han escrito su brillo
    __syncthreads();

    if (e == 0)
    {
        printf(">>> Galaxia %d completa:\n", g);
        for (int i = 0; i < estrellasPorGalaxia; ++i)
        {
            printf(" Estrella %d -> Brillo %d\n", i, brillos[i]);
        }
        printf("\n");
    }
}

// main del programa
int main(int argc, char **argv)
{
    int galaxias = (argc > 1) ? std::atoi(argv[1]) : 3;
    int estrellas = (argc > 2) ? std::atoi(argv[2]) : 4;

    if (estrellas <= 0)
    {
        fprintf(stderr, "El número de estrellas debe ser mayor que 0\n");
        return 1;
    }

    // Un bloque por galaxia un hilo por estrella
    dim3 block(estrellas);
    size_t sharedBytes = estrellas * sizeof(int);

    for (int g = 0; g < galaxias; ++g)
    {
        // Lanzar kernel para cada galaxia
        galaxia_fase<<<1, block, sharedBytes>>>(g, estrellas);
        cudaDeviceSynchronize();
    }

    return 0;
}
