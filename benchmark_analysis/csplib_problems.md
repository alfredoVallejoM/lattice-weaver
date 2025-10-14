# Problemas de Benchmark CSP de CSPLib

## Problemas Relevantes para LatticeWeaver

### prob054: N-Queens
- **Descripción**: Problema clásico de colocar N reinas en un tablero de ajedrez NxN sin que se ataquen entre sí.
- **Tipo**: CSP con restricciones binarias.
- **Complejidad**: Exponencial en N.
- **Uso en benchmarks**: Problema estándar para evaluar solvers CSP.

### prob057: Killer Sudoku
- **Descripción**: Variante de Sudoku con restricciones adicionales de suma.
- **Tipo**: CSP con restricciones globales y aritméticas.
- **Complejidad**: Alta, especialmente para instancias difíciles.

### prob067: Quasigroup Completion
- **Descripción**: Completar una tabla latina parcialmente llena.
- **Tipo**: CSP con restricciones de diferencia.
- **Complejidad**: Variable según el tamaño y el número de celdas pre-llenadas.

### prob074: Maximum Clique
- **Descripción**: Encontrar el clique máximo en un grafo.
- **Tipo**: Problema de optimización relacionado con graph coloring.
- **Complejidad**: NP-completo.

### prob079: n-Queens Completion Problem
- **Descripción**: Variante de N-Queens donde algunas reinas ya están colocadas.
- **Tipo**: CSP con restricciones binarias y asignaciones parciales.
- **Complejidad**: Similar a N-Queens pero con espacio de búsqueda reducido.

### prob080: Blocked n-Queens Problem
- **Descripción**: N-Queens con algunas posiciones bloqueadas.
- **Tipo**: CSP con restricciones binarias y restricciones de dominio.
- **Complejidad**: Similar a N-Queens pero con dominios restringidos.

## Problemas Relacionados con Scheduling

### prob061: Resource-Constrained Project Scheduling Problem (RCPSP)
- **Descripción**: Problema de scheduling con recursos limitados.
- **Tipo**: CSP con restricciones de precedencia y recursos.
- **Complejidad**: NP-completo.
- **Relevancia**: Similar a Job Shop Scheduling.

### prob026: Sports Tournament Scheduling
- **Descripción**: Programar torneos deportivos con restricciones de disponibilidad.
- **Tipo**: CSP con restricciones complejas de scheduling.
- **Complejidad**: Alta, especialmente para torneos grandes.

## Problemas de Graph Coloring

### Graph Coloring (no listado explícitamente en CSPLib)
- **Descripción**: Asignar colores a nodos de un grafo tal que nodos adyacentes tengan colores diferentes.
- **Tipo**: CSP con restricciones binarias.
- **Complejidad**: NP-completo.
- **Uso en benchmarks**: Problema estándar para evaluar solvers CSP.

## Observaciones

1. **CSPLib** contiene **96 problemas** en **16 categorías**, propuestos por **74 autores**.
2. Los problemas están organizados por **área temática** y **número de problema**.
3. CSPLib mantiene una lista de **lenguajes de restricciones** y herramientas para resolver problemas.
4. La motivación principal de CSPLib es **enfocar la investigación en problemas estructurados** en lugar de problemas puramente aleatorios.

## Problemas Usados en Nuestros Benchmarks

- **N-Queens**: prob054
- **Sudoku**: Relacionado con prob057 (Killer Sudoku)
- **Graph Coloring**: No listado explícitamente, pero es un problema estándar
- **Job Shop Scheduling**: Relacionado con prob061 (RCPSP)
- **Simple CSP**: Problemas aleatorios con restricciones binarias (no en CSPLib)

## Referencias

- CSPLib: https://www.csplib.org/
- CSPLib Git Repository: https://github.com/csplib/csplib
- XCSP3 Format: https://xcsp.org/
- XCSP Competitions: https://www.xcsp.org/competitions/

