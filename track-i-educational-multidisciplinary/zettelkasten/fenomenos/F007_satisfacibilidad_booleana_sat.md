---
id: F007
tipo: fenomeno
titulo: Satisfacibilidad booleana (SAT)
dominios: [informatica, matematicas, logica, inteligencia_artificial, ingenieria_electronica]
categorias: [C003, C006]
tags: [logica_proposicional, NP_completo, satisfacibilidad, resolucion_problemas, verificacion, planificacion]
fecha_creacion: 2025-10-12
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
prioridad: alta  # maxima | alta | media | baja
---

# Satisfacibilidad booleana (SAT)

## Descripción

El **Problema de Satisfacibilidad Booleana (SAT)** es un problema fundamental en la lógica matemática y la informática teórica. Dada una fórmula booleana, el problema consiste en determinar si existe una asignación de valores de verdad (verdadero o falso) a sus variables que haga que la fórmula sea verdadera (satisfacible). Si tal asignación existe, la fórmula es satisfacible; de lo contrario, es insatisfacible. SAT es el primer problema que se demostró ser NP-completo, un hito establecido por Stephen Cook en 1971, lo que significa que es uno de los problemas computacionalmente más difíciles en la clase NP.

A pesar de su complejidad teórica, los avances en los "solvers SAT" (programas que resuelven instancias de SAT) han sido extraordinarios en las últimas décadas. Estos solvers son ahora capaces de resolver instancias con millones de variables y cláusulas, lo que ha permitido que SAT se convierta en una herramienta práctica y poderosa para resolver una amplia gama de problemas en campos como la verificación de hardware y software, la planificación automática, la inteligencia artificial, la criptografía y la bioinformática. La formulación de un problema como una instancia SAT es un enfoque común para aprovechar la eficiencia de estos solvers.

## Componentes Clave

### Variables
-   **Variables Booleanas (x_1, x_2, ..., x_n):** Símbolos que pueden tomar uno de dos valores de verdad.
-   **Literales:** Una variable booleana o su negación (ej. `x_i` o `¬x_i`).
-   **Cláusulas:** Una disyunción de literales (ej. `x_1 ∨ ¬x_2 ∨ x_3`).
-   **Fórmula en Forma Normal Conjuntiva (FNC/CNF):** Una conjunción de cláusulas (ej. `(x_1 ∨ ¬x_2) ∧ (¬x_1 ∨ x_3)`).

### Dominios
-   **Dominio de Variables Booleanas:** {Verdadero, Falso} o {1, 0}.

### Restricciones/Relaciones
-   **Restricciones de Cláusulas:** Cada cláusula en la FNC debe ser satisfecha (evaluarse como verdadera) por la asignación de valores de verdad a las variables.

### Función Objetivo (si aplica)
-   En el problema SAT clásico, no hay una función objetivo a optimizar; el objetivo es la **satisfacción** (encontrar *cualquier* asignación que haga la fórmula verdadera) o la **insatisfacción** (demostrar que ninguna asignación existe).
-   Para el problema **Maximum SAT (MaxSAT)**, la función objetivo es maximizar el número de cláusulas satisfechas.

## Mapeo a Formalismos

### CSP (Constraint Satisfaction Problem)
-   **Variables:** Las variables booleanas `x_1, ..., x_n` de la fórmula SAT.
-   **Dominios:** {Verdadero, Falso} para cada variable.
-   **Restricciones:** Cada cláusula de la fórmula FNC se traduce directamente en una restricción. Por ejemplo, la cláusula `(x_1 ∨ ¬x_2)` es una restricción que prohíbe la asignación `x_1=Falso, x_2=Verdadero`.
-   **Tipo:** Satisfacción (encontrar una asignación que satisfaga todas las restricciones).

### Lógica Proposicional
-   SAT es la encarnación computacional del problema de determinar la validez de una fórmula en lógica proposicional. Una fórmula es válida si su negación es insatisfacible.

## Ejemplos Concretos

### Ejemplo 1: Verificación de Circuitos Digitales
**Contexto:** Verificar si un diseño de circuito digital (ej. un chip) se comporta como se espera, o si dos diseños son funcionalmente equivalentes.

**Mapeo:**
-   Variables Booleanas = Entradas, salidas y estados internos de las puertas lógicas del circuito.
-   Fórmula FNC = Representación de la lógica del circuito y las propiedades a verificar (ej. `(entrada_A AND entrada_B) XOR salida_C`).

**Solución esperada:** Si la fórmula que representa la no equivalencia de dos circuitos es satisfacible, significa que hay un caso donde se comportan diferente, revelando un error. Si es insatisfacible, son equivalentes.

**Referencias:** Biere, A., & Fröhlich, A. (2015). "Bounded Model Checking". In *Handbook of Satisfiability* (pp. 195-231). IOS Press.

### Ejemplo 2: Planificación Automática
**Contexto:** Encontrar una secuencia de acciones para alcanzar un objetivo dado un estado inicial y un conjunto de acciones posibles (ej. un robot que debe moverse de un punto A a un punto B en un entorno con obstáculos).

**Mapeo:**
-   Variables Booleanas = Representan si una acción se realiza en un momento dado, o si una condición es verdadera en un estado particular.
-   Fórmula FNC = Codifica las precondiciones y efectos de las acciones, y el objetivo a alcanzar.

**Solución esperada:** Una asignación satisfacible proporciona un plan (secuencia de acciones) que logra el objetivo.

**Referencias:** Kautz, H., & Selman, B. (1992). "Planning as Satisfiability". *Proceedings of the European Conference on Artificial Intelligence*.

### Ejemplo 3: Resolución de Sudoku
**Contexto:** Como se mencionó en [[F006]] - Coloreo de grafos, el Sudoku puede ser modelado como un problema de coloreo. También puede ser directamente modelado como SAT.

**Mapeo:**
-   Variables Booleanas `x_ijk`: Verdadero si la celda `(i,j)` contiene el número `k`.
-   Fórmula FNC = Conjunto de cláusulas que codifican las reglas del Sudoku:
    -   Cada celda `(i,j)` contiene exactamente un número `k`.
    -   Cada número `k` aparece exactamente una vez en cada fila `i`.
    -   Cada número `k` aparece exactamente una vez en cada columna `j`.
    -   Cada número `k` aparece exactamente una vez en cada bloque de 3x3.

**Solución esperada:** Una asignación satisfacible de las variables `x_ijk` proporciona la solución al Sudoku.

**Referencias:** Marques-Silva, J. P., & Sakallah, K. A. (1999). "GRASP: A New Search Algorithm for Satisfiability". *IEEE Transactions on Computers*, 48(5), 506-521.

## Conexiones

### Categoría Estructural
-   [[C003]] - Optimización con Restricciones: SAT es un caso particular de problema de satisfacción de restricciones, donde las variables son booleanas y las restricciones son cláusulas.
-   [[C006]] - Satisfacibilidad Lógica: SAT es el problema canónico de satisfacibilidad en lógica proposicional.

### Isomorfismos
-   [[I005]] - SAT ≅ 3-SAT (Teorema de Cook-Levin, 3-SAT es también NP-completo y cualquier instancia SAT puede reducirse a 3-SAT).
-   [[I007]] - SAT ≅ Coloreo de Grafos (como se vio en [[F006]], el coloreo de grafos puede reducirse a SAT).
-   [[I008]] - SAT ≅ Problema de la Mochila (Knapsack Problem) (muchos problemas NP-completos son inter-reducibles).

### Instancias en Otros Dominios
-   [[F006]] - Coloreo de grafos: Puede ser formulado como SAT.
-   [[F002]] - Redes de Regulación Génica: Las redes booleanas son un caso especial de SAT.
-   [[F011]] - Lógica y Argumentación (Filosofía)

### Técnicas Aplicables
-   [[T004]] - Algoritmos DPLL (Davis-Putnam-Logemann-Loveland): Base de la mayoría de los solvers SAT modernos.
-   [[T006]] - Búsqueda Local (WalkSAT, GSAT): Algoritmos heurísticos para encontrar soluciones satisfacibles.
-   [[T003]] - Resolución de Cláusulas (Clause Learning): Técnica clave para mejorar la eficiencia de los solvers DPLL.

### Conceptos Fundamentales
-   [[K003]] - NP-Completitud
-   [[K006]] - Lógica Proposicional
-   [[K007]] - Forma Normal Conjuntiva (FNC/CNF)
-   [[K008]] - Reducción (entre problemas)

### Prerequisitos
-   [[K006]] - Lógica Booleana Básica
-   [[K003]] - Conceptos de Complejidad Computacional

### Conexiones Inversas
- [[C003]] - Optimización con Restricciones
- [[C006]] - Satisfacibilidad Lógica
- [[F006]] - Coloreo de grafos
- [[F011]] - Lógica y Argumentación (Filosofía)
- [[I007]] - Coloreo de grafos ≅ Problema de Satisfacibilidad Booleana (SAT)
- [[T004]] - DPLL

## Propiedades Matemáticas

### Complejidad Computacional
-   **Decisión:** SAT es NP-completo. Esto significa que no se conoce un algoritmo de tiempo polinomial para resolverlo, y si se encontrara uno, implicaría P=NP.
-   **Optimización (MaxSAT):** NP-hard.

### Propiedades Estructurales
-   **Teorema de Cook-Levin:** Demuestra que SAT es NP-completo, estableciendo la base de la teoría de la complejidad computacional.
-   **Fase de Transición:** Para fórmulas SAT generadas aleatoriamente, existe una fase de transición aguda donde la probabilidad de satisfacibilidad cambia de casi 1 a casi 0 a medida que la relación entre el número de cláusulas y el número de variables aumenta.

### Teoremas Relevantes
-   **Teorema de Cook-Levin (1971):** El problema de satisfacibilidad booleana es NP-completo.

## Visualización

### Tipos de Visualización Aplicables
1.  **Grafo de Implicación:** Representar las variables y sus relaciones lógicas como un grafo dirigido.
2.  **Árbol de Búsqueda:** Visualizar el espacio de búsqueda explorado por un solver SAT (ej. DPLL), mostrando las decisiones y las propagaciones de unitarias.
3.  **Fórmula FNC:** Representación gráfica de las cláusulas y literales.

### Componentes Reutilizables
-   Componentes de visualización de grafos.
-   Componentes para representar árboles de búsqueda.
-   Visualizadores de fórmulas lógicas.

## Recursos

### Literatura Clave
1.  Cook, S. A. (1971). "The complexity of theorem-proving procedures". *Proceedings of the third annual ACM symposium on Theory of computing*.
2.  Biere, A., Heule, M., Maaren, H. v., & Walsh, T. (Eds.). (2009). *Handbook of Satisfiability*. IOS Press.
3.  Knuth, D. E. (2015). *The Art of Computer Programming, Volume 4A: Combinatorial Algorithms, Part 1*. Addison-Wesley Professional.

### Datasets
-   **Benchmarks SAT:** Colecciones de instancias SAT de problemas reales y sintéticos (ej. SATLIB, SAT Competition).

### Implementaciones Existentes
-   **MiniSat:** Uno de los solvers SAT más influyentes y ampliamente utilizado.
-   **Glucose, CaDiCaL:** Solvers SAT de alto rendimiento.

### Código en LatticeWeaver
-   **Módulo:** `lattice_weaver/phenomena/boolean_satisfiability/`
-   **Tests:** `tests/phenomena/test_boolean_satisfiability.py`
-   **Documentación:** `docs/phenomena/boolean_satisfiability.md`

## Estado de Implementación

### Fase 1: Investigación
-   [x] Revisión bibliográfica completada
-   [x] Ejemplos concretos identificados
-   [x] Datasets recopilados (referenciados)
-   [ ] Documento de investigación creado (integrado aquí)

### Fase 2: Diseño
-   [x] Mapeo a CSP diseñado
-   [x] Mapeo a otros formalismos (Lógica Proposicional)
-   [ ] Arquitectura de código planificada
-   [ ] Visualizaciones diseñadas

### Fase 3: Implementación
-   [ ] Clases base implementadas
-   [ ] Algoritmos implementados
-   [ ] Tests unitarios escritos
-   [ ] Tests de integración escritos

### Fase 4: Visualización
-   [ ] Componentes de visualización implementados
-   [ ] Visualizaciones interactivas creadas
-   [ ] Exportación de visualizaciones

### Fase 5: Documentación
-   [ ] Documentación de API
-   [ ] Tutorial paso a paso
-   [ ] Ejemplos de uso
-   [ ] Casos de estudio

### Fase 6: Validación
-   [ ] Revisión por pares
-   [ ] Validación con expertos del dominio
-   [ ] Refinamiento basado en feedback

## Estimaciones

-   **Tiempo de investigación:** 25 horas
-   **Tiempo de diseño:** 12 horas
-   **Tiempo de implementación:** 45 horas
-   **Tiempo de visualización:** 20 horas
-   **Tiempo de documentación:** 15 horas
-   **TOTAL:** 117 horas

## Notas Adicionales

### Ideas para Expansión
-   Explorar la relación con el problema de Satisfacibilidad Módulos de Teorías (SMT).
-   Aplicaciones en la generación automática de tests.
-   Estudio de la fase de transición para instancias aleatorias de SAT.

### Preguntas Abiertas
-   ¿Cuáles son los límites prácticos de los solvers SAT actuales?
-   ¿Cómo se pueden integrar los solvers SAT con otras técnicas de IA para problemas más complejos?

### Observaciones
-   La evolución de los solvers SAT es un testimonio del poder de la investigación teórica y la ingeniería práctica para superar barreras de complejidad computacional.

---

**Última actualización:** 2025-10-13
**Responsable:** Agente Autónomo de Análisis

