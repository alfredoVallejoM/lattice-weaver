# Reporte de Benchmarking CSP

Este reporte presenta los resultados de los experimentos de benchmarking realizados con LatticeWeaver.

## Resumen General

- **Número total de experimentos:** 2
- **Familias de problemas testeadas:** 1
- **Solvers utilizados:** 2

## Resultados Agregados por Problema y Solver

| _       | _           |   Tiempo Promedio (s) |   Desviación Estándar Tiempo (s) |   Soluciones Encontradas |   Nodos Explorados |   Backtracks |   Restricciones Chequeadas |   _ |
|:--------|:------------|----------------------:|---------------------------------:|-------------------------:|-------------------:|-------------:|---------------------------:|----:|
| nqueens | default     |             0.0146303 |                              nan |                        1 |                 11 |            0 |                          0 | 100 |
| nqueens | tms_enabled |             0.0184766 |                              nan |                        1 |                 11 |            0 |                          0 | 100 |

## Detalles de Experimentos Individuales

| problem_family   | solver_name   |   time_taken |   solutions_found |   nodes_explored |   backtracks |   constraints_checked | solution_valid   |
|:-----------------|:--------------|-------------:|------------------:|-----------------:|-------------:|----------------------:|:-----------------|
| nqueens          | default       |    0.0146303 |                 1 |               11 |            0 |                     0 | True             |
| nqueens          | tms_enabled   |    0.0184766 |                 1 |               11 |            0 |                     0 | True             |


---

*Generado automáticamente por LatticeWeaver Benchmark Report Generator*
