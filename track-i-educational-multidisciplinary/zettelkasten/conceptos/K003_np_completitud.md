---
id: K003
tipo: concepto
titulo: NP-Completitud
dominio_origen: informatica,matematicas
categorias_aplicables: [C003, C006]
tags: [teoria_de_la_computacion, complejidad_computacional, problemas_dificiles, P_vs_NP]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
---

# Concepto: NP-Completitud

## Descripción

La **NP-Completitud** es una propiedad fundamental en la teoría de la complejidad computacional que clasifica un conjunto de problemas como los más difíciles dentro de la clase de problemas NP (tiempo polinomial no determinista). Un problema es NP-completo si cumple dos condiciones: 1) pertenece a la clase NP, lo que significa que una solución propuesta puede ser verificada en tiempo polinomial por una máquina de Turing determinista; y 2) es NP-difícil, lo que implica que cualquier otro problema en NP puede ser reducido a él en tiempo polinomial. Esto significa que si se encontrara un algoritmo de tiempo polinomial para resolver un problema NP-completo, entonces todos los problemas en NP podrían resolverse en tiempo polinomial, lo que resolvería el famoso problema P vs NP.

## Origen

**Dominio de origen:** [[D003]] - Informática (Teoría de la Computación)
**Año de desarrollo:** 1971
**Desarrolladores:** Stephen Cook y Leonid Levin.
**Contexto:** Stephen Cook, en su paper de 1971 "The Complexity of Theorem Proving Procedures", introdujo el concepto de NP-completitud y demostró que el problema de satisfacibilidad booleana (SAT) es NP-completo. Poco después, Leonid Levin demostró independientemente un resultado similar. Este descubrimiento unificó una gran cantidad de problemas aparentemente no relacionados que eran difíciles de resolver, mostrando que todos eran, en esencia, igual de difíciles.

## Formulación

### Clases de Complejidad

1.  **Clase P:** Problemas que pueden ser resueltos por una máquina de Turing determinista en tiempo polinomial.
2.  **Clase NP:** Problemas cuyas soluciones pueden ser *verificadas* por una máquina de Turing determinista en tiempo polinomial. No implica que puedan ser *resueltos* en tiempo polinomial.

### Definición de NP-Completitud

Un problema de decisión `L` es **NP-completo** si:

1.  `L ∈ NP` (L pertenece a la clase NP).
2.  Para cualquier otro problema `L'` en NP, `L'` es reducible a `L` en tiempo polinomial (`L' ≤p L`). Esto significa que existe una función `f` computable en tiempo polinomial tal que `x ∈ L'` si y solo si `f(x) ∈ L`.

### Problema SAT (Satisfiability)

El problema de satisfacibilidad booleana (SAT) fue el primer problema demostrado ser NP-completo. Dada una fórmula booleana en forma normal conjuntiva (CNF), ¿existe una asignación de valores de verdad a sus variables que haga que la fórmula sea verdadera?

## Análisis

### Implicaciones

1.  **Problema P vs NP:** La existencia de problemas NP-completos es central para la pregunta abierta más importante en informática teórica: ¿P = NP? Si P = NP, entonces todos los problemas NP-completos (y por lo tanto todos los problemas en NP) podrían resolverse en tiempo polinomial. Si P ≠ NP, entonces no existe tal algoritmo.
2.  **Equivalencia de dificultad:** Todos los problemas NP-completos son equivalentes en dificultad en el sentido de que si uno puede resolverse eficientemente, todos pueden.

### Limitaciones

1.  **No hay algoritmos eficientes conocidos:** A pesar de décadas de investigación, no se ha encontrado ningún algoritmo de tiempo polinomial para ningún problema NP-completo.
2.  **Dificultad inherente:** La NP-completitud sugiere que estos problemas son inherentemente difíciles de resolver de manera óptima en el peor de los casos.

## Aplicabilidad

### Categorías Estructurales Aplicables

1.  [[C003]] - Optimización con Restricciones
    -   **Por qué funciona:** Muchos problemas de optimización con restricciones (ej. [[F006]] - Coloreo de grafos, [[F007]] - Satisfacibilidad booleana (SAT)) son NP-completos o NP-difíciles. La NP-completitud nos informa sobre la dificultad inherente de encontrar soluciones óptimas y motiva el uso de heurísticas o algoritmos de aproximación.
    -   **Limitaciones:** La NP-completitud no ofrece soluciones, sino que clasifica la dificultad. Para problemas NP-completos, se suelen usar algoritmos exactos con tiempo exponencial o heurísticas/metaheurísticas que no garantizan la optimalidad.

2.  [[C006]] - Satisfacibilidad Lógica
    -   **Por qué funciona:** La NP-completitud se originó con el problema SAT, que es el problema canónico de esta clase. Otros problemas de satisfacibilidad lógica a menudo son NP-completos o NP-difíciles.
    -   **Limitaciones:** A pesar de la dificultad teórica, los *solvers* SAT modernos (ej. [[T004]] - DPLL) son sorprendentemente eficientes en la práctica para muchas instancias de problemas del mundo real.

### Fenómenos Donde Se Ha Aplicado

-   [[F007]] - Satisfacibilidad booleana (SAT): El problema prototípico NP-completo.
-   [[F006]] - Coloreo de grafos: Determinar si un grafo puede colorearse con `k` colores es NP-completo para `k ≥ 3`.
-   [[F005]] - Algoritmo de Dijkstra / Caminos mínimos: Encontrar el camino más corto es P, pero encontrar el camino más largo simple es NP-completo.

## Conexiones
#- [[K003]] - Conexión inversa con Concepto.
- [[K003]] - Conexión inversa con Concepto.
- [[K003]] - Conexión inversa con Concepto.
- [[D003]] - Conexión inversa con Dominio.

## Técnicas Relacionadas

-   [[T004]] - DPLL: Un algoritmo exacto para resolver SAT, que aunque es exponencial en el peor caso, es muy eficiente en la práctica para muchas instancias.
-   [[T005]] - Algoritmo Genético: Una metaheurística que puede usarse para encontrar soluciones aproximadas a problemas NP-completos cuando no se requiere la optimalidad o cuando el tiempo de cómputo es limitado.
-   [[T006]] - Recocido Simulado (Simulated Annealing): Otra metaheurística para problemas de optimización NP-difíciles.

#- [[K003]] - Conexión inversa con Concepto.
## Conceptos Fundamentales Relacionados

-   [[K008]] - Complejidad Computacional: La NP-completitud es un concepto central dentro de esta área, que estudia los recursos (tiempo, espacio) necesarios para resolver problemas computacionales.
-   [[K006]] - Teoría de Grafos: Muchos problemas NP-completos se formulan en términos de grafos (ej. problema del viajante de comercio, clique máximo, conjunto independiente).

## Historia y Evolución

### Desarrollo Histórico

-   **1936:** Alan Turing introduce la máquina de Turing, base de la computación moderna.
-   **1960s:** Se desarrollan las bases de la teoría de la complejidad computacional.
-   **1971:** Stephen Cook demuestra que SAT es NP-completo.
-   **1972:** Richard Karp publica una lista de 21 problemas NP-completos, demostrando la ubicuidad de esta clase de problemas.
-   **Desde entonces:** Se han identificado miles de problemas NP-completos en diversas áreas.

### Impacto

El concepto de NP-completitud ha tenido un impacto inmenso en la informática y la ciencia en general. Ha proporcionado una comprensión profunda de los límites fundamentales de la computación eficiente y ha guiado la investigación hacia el desarrollo de algoritmos de aproximación, heurísticas y técnicas de búsqueda local para problemas intratables. Es una herramienta conceptual esencial para cualquier científico o ingeniero que trabaje con problemas computacionales complejos.

**Citaciones:** El paper de Cook de 1971 es uno de los más citados en informática.
**Adopción:** Fundamental en algoritmos, inteligencia artificial, investigación operativa, bioinformática, criptografía y muchas otras áreas.

---

**Última actualización:** 2025-10-13  
**Responsable:** Manus AI  
**Validado por:** [Pendiente de revisión por experto humano]
- [[I005]]
- [[I007]]
- [[T004]]
- [[K006]]
- [[K008]]
- [[C003]]
- [[C006]]
- [[F005]]
- [[F006]]
- [[F007]]
- [[T005]]
- [[T006]]
