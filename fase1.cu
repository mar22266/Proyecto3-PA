#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// calcula el brillo de una estrella dada su galaxia y su indice
__device__ int brilloDet(int g, int e)
{
    return (g * 3 + e * 7) % 10;
}

// kernel que calcula el brillo de las estrellas en todas las galaxias
__global__ void galaxia_fase(int estrellasPorGalaxia)
{
    int g = blockIdx.x;
    int e = threadIdx.x;

    if (e >= estrellasPorGalaxia)
        return;

    int brillo = brilloDet(g, e);
    printf("Galaxia %d - Estrella %d -> Brillo: %d\n", g, e, brillo);
}

// main del programa
int main(int argc, char **argv)
{
    int galaxias = (argc > 1) ? std::atoi(argv[1]) : 3;
    int estrellas = (argc > 2) ? std::atoi(argv[2]) : 4;

    // Configurar y lanzar kernel
    dim3 grid(galaxias);
    dim3 block(estrellas);

    galaxia_fase<<<grid, block>>>(estrellas);

    // Esperar a que termine el kernel
    cudaDeviceSynchronize();

    return 0;
}
