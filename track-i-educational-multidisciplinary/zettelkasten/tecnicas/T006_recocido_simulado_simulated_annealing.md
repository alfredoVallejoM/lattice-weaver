---
id: T006
tipo: tecnica
titulo: Recocido Simulado (Simulated Annealing)
dominio_origen: fisica_estadistica,informatica,optimizacion
categorias_aplicables: [C003, C004]
tags: [optimizacion, metaheuristica, fisica_estadistica, algoritmos_probabilisticos, busqueda_local, termodinamica]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
implementado: false  # true | false
---

# Técnica: Recocido Simulado (Simulated Annealing)

## Descripción

El **Recocido Simulado (Simulated Annealing - SA)** es una metaheurística probabilística para la optimización global, inspirada en el proceso de recocido en metalurgia. El recocido es una técnica que implica calentar y luego enfriar lentamente un material para aumentar el tamaño de sus cristales y reducir sus defectos, lo que lo hace más fuerte y maleable. El algoritmo SA imita este proceso para explorar el espacio de soluciones de un problema de optimización, permitiendo ocasionalmente movimientos a soluciones peores para escapar de óptimos locales y encontrar un óptimo global.

## Origen

**Dominio de origen:** [[D004]] - Física Estadística, [[D003]] - Informática (Optimización)
**Año de desarrollo:** 1983
**Desarrolladores:** Scott Kirkpatrick, C. Daniel Gelatt y Mario P. Vecchi.
**Contexto:** Fue propuesto como una forma de resolver problemas de optimización combinatoria, como el problema del viajante de comercio, que son notoriamente difíciles para los métodos de optimización tradicionales. La inspiración provino del algoritmo de Metropolis-Hastings, utilizado en física estadística para simular el comportamiento de sistemas a diferentes temperaturas.

## Formulación

### Entrada

-   **Función de costo (E(s)):** Una función que asigna un valor numérico (energía) a cada estado `s` del sistema. El objetivo es minimizar esta función.
-   **Espacio de estados (S):** El conjunto de todas las posibles soluciones candidatas.
-   **Función de generación de vecinos (neighbor(s)):** Una función que, dado un estado `s`, genera un estado `s'`
- [[T004]]
- [[T005]]
- [[T003]]
- [[K003]]
- [[K007]]
- [[K008]]

## Conexiones
- [[T006]] - Conexión inversa con Técnica.
- [[D003]] - Conexión inversa con Dominio.
- [[D004]] - Conexión inversa con Dominio.
