#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ======================= Parámetros Globales =======================
constexpr int D = 8192;        // Dimensión para NLU
constexpr int K = 10;          // Número de intenciones (aumentado para comandos)
constexpr int MAX_QUERY = 512; // Tamaño máximo de query
constexpr int C = 7;           // Columnas CSV

// ======================= Utilidades CUDA =======================
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

inline int ceilDiv(int a, int b) { 
    return (a + b - 1) / b; 
}

// ======================= Enumeraciones =======================
enum Intent {
    CONSUMO = 0,
    PUERTA = 1,
    LUCES = 2,
    AC = 3,
    ESTADO = 4,
    AYUDA = 5,
    ESTADISTICAS = 6,
    COMPARAR = 7,
    CONTROL = 8,      // comandos de control
    MODO_MANUAL = 9   // activar/desactivar modo manual
};

// ======================= Nombres de Intenciones =======================
static const char* intentNames[K] = {
    "CONSUMO", "PUERTA", "LUCES", "A/C",
    "ESTADO", "AYUDA", "ESTADISTICAS", "COMPARAR",
    "CONTROL", "MODO_MANUAL"
};

#endif // COMMON_H