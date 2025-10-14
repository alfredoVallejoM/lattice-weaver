# Reporte de Análisis de Benchmarks del Compilador Multiescala
**Fecha de Generación**: 2025-10-14 12:20:43
**Número Total de Benchmarks**: 160

---

## Resumen Ejecutivo

Se ejecutaron **160** benchmarks en total, evaluando **7** estrategias de compilación en **7** tipos de problemas CSP diferentes.

## Estadísticas por Estrategia

| Estrategia | Tiempo Total (s) | Tiempo Compilación (s) | Tiempo Resolución (s) | Memoria (MB) | Tasa de Éxito (%) | Ratio Compresión |
|------------|------------------|------------------------|----------------------|--------------|-------------------|------------------|
| NoCompilation | 1.0980 ± 4.2895 | 0.0000 | 1.0980 | 0.00 | 73.9 | 1.00x |
| FixedLevel_L1 | 0.8474 ± 3.8861 | 0.0015 | 0.8455 | 0.02 | 73.9 | 1.00x |
| FixedLevel_L2 | 0.8639 ± 3.9660 | 0.0014 | 0.8621 | 0.02 | 73.9 | 1.00x |
| FixedLevel_L3 | 0.8644 ± 3.9391 | 0.0015 | 0.8626 | 0.02 | 73.9 | 1.00x |
| FixedLevel_L4 | 0.8947 ± 4.0072 | 0.0016 | 0.8927 | 0.02 | 72.7 | 1.00x |
| FixedLevel_L5 | 0.8604 ± 3.9490 | 0.0016 | 0.8585 | 0.02 | 73.9 | 1.00x |
| FixedLevel_L6 | 0.8544 ± 3.9202 | 0.0016 | 0.8524 | 0.02 | 73.9 | 1.00x |

## Análisis de Sobrecarga de Compilación

Mejora promedio de cada estrategia de compilación vs estrategia sin compilación (NoCompilation):

| Estrategia | Mejora Promedio (%) | Mejora Mediana (%) | Desviación Estándar |
|------------|---------------------|--------------------|--------------------||
| FixedLevel_L1 | -0.23 | -2.19 | 53.96 |
| FixedLevel_L2 | +1.71 | -3.31 | 51.01 |
| FixedLevel_L3 | -6.29 | -3.01 | 55.75 |
| FixedLevel_L4 | -4.71 | -2.51 | 59.61 |
| FixedLevel_L5 | +0.35 | -2.41 | 53.30 |
| FixedLevel_L6 | -0.57 | -2.40 | 55.26 |

**Nota**: Valores negativos indican que la estrategia es más lenta que el baseline.

## Estadísticas por Tipo de Problema

| Tipo de Problema | Tiempo Total (s) | Tasa de Éxito (%) | Número de Benchmarks |
|------------------|------------------|-------------------|---------------------|
| Graph-Coloring-0.2 | 0.2184 | 66.7 | 21 |
| Graph-Coloring-0.3 | 0.0955 | 66.7 | 21 |
| Graph-Coloring-0.5 | 0.0260 | 0.0 | 21 |
| N-Queens | 0.0151 | 100.0 | 35 |
| Simple-CSP-0.2 | 0.0020 | 100.0 | 20 |
| Simple-CSP-0.3 | 6.3885 | 100.0 | 21 |
| Simple-CSP-0.5 | 0.0835 | 66.7 | 21 |

## Observaciones y Recomendaciones

### Rendimiento Negativo de la Compilación

Las siguientes estrategias muestran un rendimiento **peor** que la estrategia sin compilación:

- **FixedLevel_L1**: -0.23% (más lento que el baseline)
- **FixedLevel_L3**: -6.29% (más lento que el baseline)
- **FixedLevel_L4**: -4.71% (más lento que el baseline)
- **FixedLevel_L6**: -0.57% (más lento que el baseline)

**Recomendación**: Investigar la sobrecarga de compilación en estos niveles y optimizar el proceso de compilación.

### Baja Tasa de Éxito

Las siguientes estrategias tienen una tasa de éxito inferior al 90%:

- **NoCompilation**: 73.9%
- **FixedLevel_L1**: 73.9%
- **FixedLevel_L2**: 73.9%
- **FixedLevel_L3**: 73.9%
- **FixedLevel_L4**: 72.7%
- **FixedLevel_L5**: 73.9%
- **FixedLevel_L6**: 73.9%

**Recomendación**: Investigar los problemas que no se resuelven y ajustar el solucionador o los generadores de problemas.

