# Diseño de Pruebas Exhaustivas y Benchmarking para Fibration Flow y CSPSolver

## 1. Objetivo

El objetivo de este documento es detallar el plan para realizar pruebas exhaustivas y benchmarking comparativo entre el `CSPSolver` tradicional (backtracking con forward checking) y el `CSPSolver` integrado con Fibration Flow. Se busca validar el rendimiento, identificar los escenarios donde Fibration Flow ofrece ventajas y documentar los resultados de manera clara y concisa.

## 2. Metodología de Pruebas

Se utilizará una metodología de pruebas basada en la variación de la complejidad de los problemas CSP, midiendo métricas clave de rendimiento y analizando los resultados.

### 2.1. Problemas de Prueba

Se seleccionarán problemas CSP representativos que permitan escalar en complejidad:

*   **Problema de N-Reinas:** Un problema clásico de satisfacción de restricciones que permite variar fácilmente el tamaño (N) y, por lo tanto, la complejidad del espacio de búsqueda. Se probarán diferentes valores de N (ej., 4, 8, 12, 16, etc.) para observar cómo escala el rendimiento.
*   **Problema de Coloreado de Grafos:** Otro problema combinatorio que puede variar en complejidad según el número de nodos, aristas y colores disponibles. Se generarán grafos aleatorios o predefinidos de diferentes tamaños.
*   **Problemas de Satisfacción de Restricciones Aleatorios:** Generación de CSPs aleatorios con diferentes densidades de restricciones y tamaños de dominio para explorar una gama más amplia de escenarios.

### 2.2. Métricas de Rendimiento

Se registrarán las siguientes métricas para cada ejecución:

*   **Tiempo de Ejecución (segundos):** El tiempo total que tarda el solver en encontrar la primera solución o determinar la insatisfacibilidad.
*   **Nodos Explorados:** El número de nodos visitados en el árbol de búsqueda.
*   **Retrocesos (Backtracks):** El número de veces que el algoritmo tuvo que retroceder.
*   **Restricciones Verificadas:** El número total de veces que se evaluaron las restricciones.
*   **Número de Soluciones Encontradas:** (Si aplica, para búsqueda de todas las soluciones).

### 2.3. Escenarios de Prueba

Se ejecutarán los solvers en los siguientes escenarios:

*   **Búsqueda de la Primera Solución:** El objetivo es encontrar una solución válida lo más rápido posible.
*   **Búsqueda de Todas las Soluciones:** El objetivo es encontrar todas las soluciones posibles (para problemas pequeños/medianos).
*   **Problemas Insatisfacibles:** Evaluar la eficiencia para determinar la insatisfacibilidad.

### 2.4. Configuración de los Solvers

*   **CSPSolver Tradicional:** Se utilizará la implementación de backtracking con forward checking (`CSPSolver.solve`).
*   **CSPSolver con Fibration Flow:** Se utilizará la integración (`CSPSolver.solve_with_fibration_flow`).

## 3. Análisis de Resultados

Los resultados se analizarán para:

*   **Comparación Directa:** Comparar las métricas de rendimiento entre ambos solvers para cada problema y escenario.
*   **Puntos de Inflexión:** Identificar si existe un tamaño de problema a partir del cual Fibration Flow comienza a superar al solver tradicional, o viceversa.
*   **Ventajas de Fibration Flow:** Determinar en qué tipos de problemas o complejidades Fibration Flow ofrece una mejora significativa.
*   **Robustez:** Evaluar la estabilidad del rendimiento de ambos solvers ante diferentes configuraciones de problemas.

## 4. Documentación de Resultados

Los resultados se documentarán en un informe final (Fase 9) que incluirá:

*   Tablas comparativas de métricas de rendimiento.
*   Gráficos de escalabilidad (tiempo vs. tamaño del problema, nodos explorados vs. tamaño del problema).
*   Análisis cualitativo de los hallazgos.
*   Conclusiones y recomendaciones.

## 5. Implementación de Pruebas

Se creará un script de pruebas (`benchmarking_script.py`) que automatice la ejecución de los solvers en los diferentes problemas y recoja las métricas. Este script se ejecutará y sus resultados se utilizarán para el informe final.
