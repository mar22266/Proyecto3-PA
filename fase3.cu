#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// calcula el brillo de una estrella dada su galaxia y su indice
__device__ int brilloDet(int g, int e)
{
    return (g * 3 + e * 7) % 10;
}

// kernel que calcula el brillo promedio de una galaxia
__global__ void galaxia_fase(int galaxiaId, int estrellasPorGalaxia, float *promedios)
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
        int suma = 0;
        for (int i = 0; i < estrellasPorGalaxia; ++i)
        {
            suma += brillos[i];
        }

        float promedio = (float)suma / (float)estrellasPorGalaxia;

        promedios[g] = promedio;

        printf("Galaxia %d - Brillo promedio: %.1f\n", g, promedio);
    }
}

// main funcion principal
int main(int argc, char **argv)
{
    int galaxias = (argc > 1) ? std::atoi(argv[1]) : 3;
    int estrellas = (argc > 2) ? std::atoi(argv[2]) : 4;

    // Validar entradas
    if (estrellas <= 0)
    {
        fprintf(stderr, "El número de estrellas debe ser mayor que 0\n");
        return EXIT_FAILURE;
    }

    // Reservar memoria en dispositivo para promedios
    float *dPromedios = nullptr;

    // Configurar y lanzar kernel
    dim3 block(estrellas);
    size_t sharedBytes = estrellas * sizeof(int);

    // Lanzar kernel para cada galaxia
    for (int g = 0; g < galaxias; ++g)
    {
        galaxia_fase<<<1, block, sharedBytes>>>(g, estrellas, dPromedios);
    }

    // Copiar resultados de vuelta al host
    float *hPromedios = (float *)std::malloc(galaxias * sizeof(float));
    if (!hPromedios)
    {
        fprintf(stderr, "Error al reservar memoria en host\n");
        return EXIT_FAILURE;
    }

    // Encontrar la galaxia con el mayor brillo promedio
    int mejor = 0;
    for (int g = 1; g < galaxias; ++g)
    {
        if (hPromedios[g] > hPromedios[mejor])
        {
            mejor = g;
        }
    }

    // Imprimir resultado
    printf("Galaxia mas brillante: %d (promedio %.1f)\n",
           mejor, hPromedios[mejor]);

    // Liberar memoria del host
    std::free(hPromedios);
    return 0;
}
