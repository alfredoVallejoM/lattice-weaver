---
id: K008
tipo: concepto
titulo: Complejidad Computacional
dominio_origen: informatica,matematicas
categorias_aplicables: [C003, C006]
tags: [teoria_de_la_computacion, algoritmos, limites_computacionales, P_vs_NP]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
---

# Concepto: Complejidad Computacional

## Descripción

La **Complejidad Computacional** es una rama de la informática teórica y las matemáticas que estudia los recursos (principalmente tiempo y espacio de memoria) necesarios para resolver problemas computacionales. Su objetivo es clasificar los problemas según su dificultad inherente, independientemente del algoritmo específico o la tecnología de hardware utilizada. Permite entender los límites fundamentales de lo que se puede computar de manera eficiente y proporciona un marco para comparar la eficiencia de diferentes algoritmos. Es crucial para determinar si un problema es tratable (resoluble en tiempo razonable) o intratable (requiere un tiempo prohibitivo para resolverse).

## Origen

**Dominio de origen:** [[D003]] - Informática, [[D005]] - Matemáticas
**Año de desarrollo:** Década de 1960
**Desarrolladores:** Juris Hartmanis, Richard Stearns, Alan Cobham, Jack Edmonds.
**Contexto:** La teoría de la complejidad computacional surgió de los trabajos fundacionales de Alan Turing sobre la computabilidad en la década de 1930. Sin embargo, el estudio formal de los recursos necesarios para la computación comenzó en la década de 1960 con trabajos como el de Hartmanis y Stearns, quienes introdujeron las clases de complejidad basadas en el tiempo y el espacio de las máquinas de Turing. El concepto de problemas NP-completos, introducido por Stephen Cook y Richard Karp en la década de 1970, fue un hito fundamental que unificó la comprensión de la dificultad de muchos problemas importantes.

## Formulación

### Medidas de Complejidad

1.  **Complejidad Temporal:** Mide el número de pasos elementales que un algoritmo necesita para resolver un problema en función del tamaño de la entrada. Se expresa típicamente usando la notación de la Gran O (O-grande).
    -   Ejemplo: `O(n)` (lineal), `O(n log n)`, `O(n^2)` (cuadrático), `O(2^n)` (exponencial).
2.  **Complejidad Espacial:** Mide la cantidad de memoria que un algoritmo necesita para resolver un problema en función del tamaño de la entrada.

### Clases de Complejidad Fundamentales

-   **P (Tiempo Polinomial):** La clase de problemas de decisión que pueden ser resueltos por un algoritmo determinista en tiempo polinomial. Se consideran problemas "tratables" o "eficientes".
-   **NP (Tiempo Polinomial No Determinista):** La clase de problemas de decisión para los cuales una solución propuesta puede ser *verificada* por un algoritmo determinista en tiempo polinomial. No implica que puedan ser *resueltos* en tiempo polinomial.
-   **NP-Completo:** Un problema en NP al que cualquier otro problema en NP puede ser reducido en tiempo polinomial. Son los problemas más difíciles en NP. El [[K003]] - NP-Completitud es el concepto clave aquí.
-   **NP-Difícil:** Un problema al que cualquier problema en NP puede ser reducido en tiempo polinomial, pero que no necesariamente está en NP.

### El Problema P vs NP

La pregunta de si `P = NP` es uno de los problemas abiertos más importantes en la informática y las matemáticas. Si `P = NP`, significaría que cualquier problema cuya solución pueda ser verificada rápidamente también puede ser resuelto rápidamente. La mayoría de los científicos creen que `P ≠ NP`.

## Análisis

### Propiedades

1.  **Independencia del modelo:** La clasificación de problemas en clases de complejidad es robusta a cambios en el modelo de computación (ej. máquina de Turing, RAM) siempre que sean equivalentes en tiempo polinomial.
2.  **Guía para el diseño de algoritmos:** La teoría de la complejidad ayuda a los diseñadores de algoritmos a saber cuándo buscar un algoritmo exacto eficiente y cuándo conformarse con heurísticas o algoritmos de aproximación.
3.  **Límites fundamentales:** Establece límites teóricos sobre lo que se puede lograr computacionalmente.

### Limitaciones

1.  **Análisis del peor caso:** La complejidad se suele definir en términos del peor caso, que puede no reflejar el rendimiento típico de un algoritmo en la práctica.
2.  **Constantes ocultas:** La notación de la Gran O ignora las constantes multiplicativas y los términos de orden inferior, que pueden ser importantes para entradas de tamaño pequeño o moderado.
3.  **No considera la implementación:** No tiene en cuenta factores prácticos como la arquitectura del hardware, la eficiencia del compilador o la calidad del código.

## Aplicabilidad

### Categorías Estructurales Aplicables

1.  [[C003]] - Optimización con Restricciones
    -   **Por qué funciona:** Muchos problemas de optimización son NP-difíciles o NP-completos. La complejidad computacional nos informa sobre la intratabilidad de encontrar soluciones óptimas y justifica el uso de heurísticas, metaheurísticas (ej. [[T005]] - Algoritmo Genético, [[T006]] - Recocido Simulado) o algoritmos de aproximación.
    -   **Limitaciones:** La teoría no proporciona soluciones directas, sino una comprensión de la dificultad inherente del problema.

2.  [[C006]] - Satisfacibilidad Lógica
    -   **Por qué funciona:** El problema de satisfacibilidad booleana (SAT) es el problema canónico NP-completo. La complejidad computacional es fundamental para entender por qué los *solvers* SAT (ej. [[T004]] - DPLL) son tan importantes y cómo se diseñan para manejar la intratabilidad del peor caso.
    -   **Limitaciones:** Aunque SAT es NP-completo, los *solvers* modernos son sorprendentemente eficientes en la práctica para muchas instancias del mundo real, lo que muestra una brecha entre la teoría del peor caso y el rendimiento promedio.

### Fenómenos Donde Se Ha Aplicado

-   [[F005]] - Algoritmo de Dijkstra / Caminos mínimos: Aunque encontrar el camino más corto es un problema P, la complejidad computacional ayuda a analizar la eficiencia de diferentes implementaciones del algoritmo.
-   [[F006]] - Coloreo de grafos: Un problema NP-completo que ilustra la dificultad de la asignación óptima de recursos.
-   [[F007]] - Satisfacibilidad booleana (SAT): El problema fundacional de la NP-completitud, con aplicaciones en verificación de hardware y software.

## Conexiones
#- [[K008]] - Conexión inversa con Concepto.
- [[D003]] - Conexión inversa con Dominio.
- [[D005]] - Conexión inversa con Dominio.

## Técnicas Relacionadas

-   [[T002]] - Algoritmo A*: La complejidad de búsqueda de este algoritmo depende de la calidad de la heurística y del tamaño del espacio de búsqueda, conceptos centrales en complejidad computacional.
-   [[T004]] - DPLL: Un algoritmo que, aunque exponencial en el peor caso, es un ejemplo de cómo se abordan problemas NP-completos en la práctica.

#- [[K008]] - Conexión inversa con Concepto.
## Conceptos Fundamentales Relacionados

-   [[K003]] - NP-Completitud: El concepto más importante y directamente relacionado, que define la clase de problemas más difíciles en NP.
-   [[K006]] - Teoría de Grafos: Muchos problemas fundamentales en teoría de grafos tienen una complejidad computacional bien estudiada, y muchos son NP-completos.
-   [[K007]] - Transiciones de Fase: En algunos problemas computacionales, la dificultad de resolverlos puede exhibir transiciones de fase, donde un pequeño cambio en los parámetros del problema puede llevar a un salto abrupto en la complejidad computacional.

## Historia y Evolución

### Desarrollo Histórico

-   **1936:** Alan Turing introduce la máquina de Turing, el modelo formal de computación.
-   **1960s:** Juris Hartmanis y Richard Stearns establecen las bases de la teoría de la complejidad, introduciendo las clases de tiempo y espacio.
-   **1971:** Stephen Cook demuestra que SAT es NP-completo.
-   **1972:** Richard Karp publica una lista de 21 problemas NP-completos, demostrando la ubicuidad de esta clase de problemas.
-   **2000:** El Clay Mathematics Institute ofrece un premio de un millón de dólares por la solución al problema P vs NP.

### Impacto

La teoría de la complejidad computacional ha tenido un impacto transformador en la informática, las matemáticas y la ciencia en general. Ha proporcionado una comprensión profunda de los límites de la computación y ha guiado el desarrollo de algoritmos y heurísticas para problemas difíciles. Es esencial para áreas como la criptografía (donde la dificultad de ciertos problemas es una ventaja), la inteligencia artificial, la optimización y el diseño de sistemas complejos. Su influencia se extiende a la filosofía de la ciencia, al cuestionar la naturaleza de la "resolubilidad" de los problemas.

**Citaciones:** Los trabajos de Cook, Karp, Hartmanis y Stearns son fundamentales en la informática teórica.
**Adopción:** Esencial en ciencias de la computación, matemáticas discretas, investigación operativa, inteligencia artificial, criptografía y bioinformática.

---

**Última actualización:** 2025-10-13  
**Responsable:** Manus AI  
**Validado por:** [Pendiente de revisión por experto humano]
- [[I007]]
- [[T002]]
- [[T004]]
- [[K003]]
- [[C003]]
- [[C006]]
- [[F005]]
- [[F006]]
- [[F007]]
- [[T005]]
- [[T006]]
- [[K006]]
- [[K007]]
