---
id: T005
tipo: tecnica
titulo: Algoritmo Genético
dominio_origen: biologia,informatica,optimizacion
categorias_aplicables: [C003]
tags: [optimizacion, evolucion, metaheuristica, inteligencia_artificial, busqueda_global, computacion_evolutiva]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
implementado: false  # true | false
---

# Técnica: Algoritmo Genético

## Descripción

Los **Algoritmos Genéticos (AG)** son una clase de algoritmos de búsqueda heurística y optimización inspirados en el proceso de selección natural y la genética biológica. Operan sobre una población de soluciones candidatas (llamadas *individuos* o *cromosomas*) que evolucionan a lo largo de generaciones. Cada individuo representa una posible solución al problema y se evalúa mediante una *función de aptitud* (fitness function). Los individuos más aptos tienen una mayor probabilidad de ser seleccionados para reproducirse, combinando su material genético (cruce o *crossover*) y mutando aleatoriamente para generar nuevas soluciones. Este proceso iterativo permite explorar el espacio de búsqueda de manera eficiente, convergiendo hacia soluciones óptimas o casi óptimas.

## Origen

**Dominio de origen:** [[D001]] - Biología (Genética, Evolución), [[D003]] - Informática (Inteligencia Artificial, Optimización)
**Año de desarrollo:** 1975
**Desarrolladores:** John Holland.
**Contexto:** John Holland, de la Universidad de Michigan, desarrolló los algoritmos genéticos en la década de 1960 y los formalizó en su libro de 1975, "Adaptation in Natural and Artificial Systems". Su objetivo era entender el poder de la adaptación natural y aplicar esos principios a sistemas artificiales para resolver problemas complejos de optimización y aprendizaje. Los AG se basan en la idea de que la evolución natural es un proceso de optimización muy efectivo.

## Formulación

### Entrada

-   **Función de aptitud (fitness_function):** Una función que evalúa la calidad de una solución candidata (individuo). Cuanto mayor sea el valor, mejor es la solución.
-   **Espacio de búsqueda:** El conjunto de todas las posibles soluciones al problema, codificadas como individuos.
-   **Parámetros del AG:** Tamaño de la población, número de generaciones, tasas de cruce y mutación.

### Salida

-   **Mejor individuo encontrado:** La solución con la mayor aptitud (o menor costo) encontrada durante la evolución.
-   **Historial de aptitud:** La aptitud promedio y máxima de la población a lo largo de las generaciones.

### Parámetros

| Parámetro          | Tipo    | Rango     | Descripción                                                                 | Valor por defecto |
|--------------------|---------|-----------|-----------------------------------------------------------------------------|-------------------|
| `population_size`  | Entero  | `> 0`     | Número de individuos en cada generación.                                    | 100               |
| `generations`      | Entero  | `> 0`     | Número de iteraciones del proceso evolutivo.                                | 500               |
| `crossover_rate`   | Flotante| `[0, 1]`  | Probabilidad de que dos individuos se crucen.                               | 0.8               |
| `mutation_rate`    | Flotante| `[0, 1]`  | Probabilidad de que un gen de un individuo mute.                            | 0.01              |
| `selection_method` | String  | `['roulette', 'tournament']` | Método para seleccionar padres para la reproducción. | `'tournament'`    |

## Algoritmo

### Pseudocódigo

```
ALGORITMO AlgoritmoGenetico(fitness_function, params)
    ENTRADA: Función de aptitud, parámetros del AG
    SALIDA: Mejor individuo encontrado
    
    1. Inicializar_Poblacion(population_size)
    
    2. PARA g DESDE 1 HASTA generations HACER
        2.1. Evaluar_Aptitud(Poblacion, fitness_function)
        2.2. Seleccionar_Padres(Poblacion, selection_method)
        2.3. Crear_Nueva_Generacion(Padres, crossover_rate, mutation_rate)
        2.4. Reemplazar_Poblacion(Nueva_Generacion)
    FIN PARA
    
    3. RETORNAR Mejor_Individuo(Poblacion)
FIN ALGORITMO

// Funciones auxiliares
// Inicializar_Poblacion(): Crea individuos aleatorios.
// Evaluar_Aptitud(): Calcula el fitness de cada individuo.
// Seleccionar_Padres(): Elige individuos para la reproducción.
// Crear_Nueva_Generacion(): Aplica cruce y mutación.
// Reemplazar_Poblacion(): Actualiza la población.
// Mejor_Individuo(): Encuentra el individuo con mayor fitness.
```

### Descripción Paso a Paso

1.  **Inicialización:** Se crea una población inicial de soluciones de forma aleatoria. Cada solución (individuo) se codifica de alguna manera (ej. cadena binaria, lista de números).
2.  **Evaluación:** Cada individuo de la población se evalúa utilizando la función de aptitud. Esta función asigna un valor numérico que indica cuán buena es la solución que representa el individuo.
3.  **Selección:** Se seleccionan individuos de la población actual para ser padres de la siguiente generación. Los individuos con mayor aptitud tienen una mayor probabilidad de ser seleccionados (ej. selección por ruleta, selección por torneo).
4.  **Cruce (Crossover):** Los padres seleccionados se combinan para crear nuevos individuos (descendencia). Esto simula la recombinación genética, donde partes de las soluciones de los padres se intercambian.
5.  **Mutación:** Se introducen pequeños cambios aleatorios en la descendencia. Esto asegura la diversidad genética en la población y ayuda a explorar nuevas regiones del espacio de búsqueda, evitando la convergencia prematura a óptimos locales.
6.  **Reemplazo:** La nueva generación de individuos reemplaza a la población actual. El proceso se repite desde el paso 2 hasta que se alcanza un criterio de terminación (ej. número máximo de generaciones, aptitud deseada).

### Invariantes

1.  **Preservación de la aptitud:** Aunque la aptitud de individuos específicos puede fluctuar, la aptitud máxima o promedio de la población tiende a mejorar o mantenerse estable a lo largo de las generaciones (principio de selección natural).
2.  **Diversidad genética:** La mutación y el cruce aseguran que la población mantenga una cierta diversidad, evitando que todas las soluciones converjan a un único punto prematuramente.

## Análisis

### Complejidad Temporal

-   **Mejor caso:** Difícil de definir, ya que es un algoritmo heurístico. Depende de la función de aptitud y el espacio de búsqueda.
-   **Caso promedio:** O(G * P * C_eval + G * P * C_op), donde G es el número de generaciones, P es el tamaño de la población, C_eval es el costo de evaluar la aptitud y C_op es el costo de las operaciones genéticas (selección, cruce, mutación).
-   **Peor caso:** Puede ser muy largo si el espacio de búsqueda es vasto y la función de aptitud es compleja, sin garantía de encontrar el óptimo global en un tiempo finito.

**Justificación:** La complejidad es dominada por el número de generaciones y el tamaño de la población, ya que cada individuo se evalúa y manipula en cada generación.

### Complejidad Espacial

-   **Espacio auxiliar:** O(P * L), donde P es el tamaño de la población y L es la longitud de la codificación de un individuo.
-   **Espacio total:** O(P * L).

**Justificación:** Se necesita almacenar la población actual y, a veces, la población de la generación anterior.

### Corrección

Los Algoritmos Genéticos son metaheurísticas, no algoritmos exactos. No garantizan encontrar la solución óptima global, ni la completitud (no pueden probar que no existe una solución si no la encuentran). Sin embargo, son *probabilísticamente completos*, lo que significa que la probabilidad de encontrar el óptimo global se acerca a 1 a medida que el tiempo de ejecución tiende a infinito.

**Teorema:** Teorema del Esquema (Schema Theorem) de Holland: Proporciona una base teórica para entender por qué los AG funcionan, indicando que los "esquemas" (patrones de bits) de alta aptitud y corta longitud se propagan exponencialmente en la población.

### Optimalidad

Los AG no son óptimos en el sentido de encontrar la mejor solución garantizada en un tiempo polinomial. Son algoritmos de búsqueda global que buscan un buen compromiso entre la exploración (buscar nuevas regiones del espacio de búsqueda) y la explotación (refinar las soluciones prometedoras). Son especialmente útiles para problemas donde el espacio de búsqueda es grande, complejo, no lineal o tiene múltiples óptimos locales.

## Aplicabilidad

### Categorías Estructurales Aplicables

1.  [[C003]] - Optimización con Restricciones
    -   **Por qué funciona:** Los AG son excelentes para problemas de optimización combinatoria y global, donde las restricciones pueden incorporarse en la función de aptitud o en la generación de individuos válidos.
    -   **Limitaciones:** La codificación del problema y el diseño de la función de aptitud son cruciales y pueden ser complejos. No garantizan la optimalidad.

2.  [[C002]] - Asignación Óptima
    -   **Por qué funciona:** Pueden encontrar asignaciones eficientes en problemas como el problema de la mochila, el problema de asignación de tareas o el problema del viajante de comercio, donde se busca la mejor combinación de elementos.
    -   **Limitaciones:** La convergencia puede ser lenta para problemas muy grandes o con funciones de aptitud muy planas.

### Fenómenos Donde Se Ha Aplicado

#### En Dominio Original (Biología/Informática)

-   **Diseño de Circuitos Electrónicos:** Optimización de la topología y componentes de circuitos para cumplir con especificaciones de rendimiento.
    -   **Resultado:** Generación de diseños de circuitos innovadores y eficientes que superan los diseños manuales.
    -   **Referencias:** Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*.

#### Transferencias a Otros Dominios

-   **Ingeniería (Diseño de Estructuras):** Optimización de la forma y materiales de estructuras (ej. puentes, alas de avión) para maximizar la resistencia y minimizar el peso.
    -   **Adaptaciones necesarias:** La codificación del individuo representa el diseño de la estructura, y la función de aptitud evalúa su resistencia y peso mediante simulaciones de elementos finitos.
    -   **Resultado:** Diseños estructurales más ligeros y resistentes, con ahorro de material.

-   **Finanzas (Optimización de Carteras):** Selección de activos para una cartera de inversión que maximice el retorno esperado para un nivel de riesgo dado.
    -   **Adaptaciones necesarias:** Los individuos representan diferentes combinaciones de activos, y la función de aptitud evalúa el rendimiento y el riesgo de la cartera.
    -   **Resultado:** Carteras de inversión optimizadas que se adaptan a los objetivos de riesgo/retorno del inversor.

### Prerequisitos

1.  **Codificación del problema:** El problema debe poder ser codificado como un "cromosoma" o "individuo" que pueda ser manipulado por operadores genéticos.
2.  **Función de aptitud:** Debe existir una forma de evaluar la calidad de cada solución candidata.
3.  **Espacio de búsqueda:** El espacio de búsqueda debe ser lo suficientemente grande y complejo como para justificar el uso de una metaheurística.

### Contraindicaciones

1.  **Problemas con soluciones exactas conocidas:** Si un problema puede resolverse de forma exacta y eficiente con otros algoritmos, los AG pueden ser excesivamente lentos y no garantizar la optimalidad.
2.  **Función de aptitud costosa:** Si la evaluación de la función de aptitud es computacionalmente muy cara, el AG puede ser inviable debido al gran número de evaluaciones requeridas.
3.  **Problemas con pocas variables o espacio de búsqueda pequeño:** Para estos casos, la búsqueda exhaustiva o métodos más simples pueden ser más eficientes.

## Variantes

### Variante 1: Programación Genética (Genetic Programming - GP)

**Modificación:** En lugar de optimizar un conjunto de parámetros, GP evoluciona programas o funciones completas. Los individuos son árboles de sintaxis abstracta que representan programas.
**Ventaja:** Puede descubrir soluciones algorítmicas o funcionales en lugar de solo valores de parámetros.
**Desventaja:** Mayor complejidad de representación y de los operadores genéticos (cruce y mutación).
**Cuándo usar:** Para problemas de aprendizaje automático, diseño de algoritmos o síntesis de programas.

### Variante 2: Algoritmos Meméticos (Memetic Algorithms - MAs)

**Modificación:** Combinan los AG con técnicas de búsqueda local (como el Recocido Simulado o la búsqueda de gradiente). Después de las operaciones genéticas, cada individuo se refina localmente.
**Ventaja:** Mejora la capacidad de explotación, permitiendo que la población converja más rápidamente a óptimos locales de alta calidad, y luego los AG exploran el espacio global.
**Desventaja:** Mayor complejidad computacional por la adición de la búsqueda local.
**Cuándo usar:** Para problemas donde se desea un equilibrio entre exploración global y refinamiento local de soluciones.

## Comparación con Técnicas Alternativas

### Técnica Alternativa 1: [[T006]] - Recocido Simulado (Simulated Annealing)

| Criterio              | Algoritmo Genético | Recocido Simulado |
|-----------------------|--------------------|-------------------|
| Complejidad temporal  | Heurística, variable | Heurística, variable |
| Complejidad espacial  | O(P*L)             | Baja              |
| Facilidad de implementación | Media              | Media             |
| Calidad de solución   | Aproximada, probabilística | Aproximada, probabilística |
| Aplicabilidad         | Optimización global, búsqueda en espacios complejos | Optimización combinatoria global |

**Cuándo preferir esta técnica (AG):** Cuando se necesita mantener una diversidad de soluciones y explorar múltiples regiones del espacio de búsqueda simultáneamente. Es más robusto frente a múltiples óptimos locales.
**Cuándo preferir la alternativa (SA):** Cuando se busca una solución de alta calidad en un espacio de búsqueda complejo, pero con un enfoque más en la búsqueda local y la capacidad de escapar de óptimos locales mediante movimientos probabilísticos.

### Técnica Alternativa 2: [[T004]] - DPLL

| Criterio              | Algoritmo Genético | DPLL                |
|-----------------------|--------------------|---------------------|
| Complejidad temporal  | Heurística, variable | Exponencial (peor caso) |
| Complejidad espacial  | O(P*L)             | Polinomial          |
| Facilidad de implementación | Media              | Media               |
| Calidad de solución   | Aproximada, probabilística | Exacta, completa    |
| Aplicabilidad         | Optimización global, búsqueda en espacios complejos | Problemas SAT, lógica |

**Cuándo preferir esta técnica (AG):** Para problemas de optimización donde no se requiere una solución exacta, el espacio de búsqueda es muy grande o complejo, y se aceptan soluciones de buena calidad pero no necesariamente óptimas.
**Cuándo preferir la alternativa (DPLL):** Cuando se requiere una solución exacta y se puede codificar el problema como SAT. Es completo y garantiza encontrar una solución si existe.

## Ejemplos de Uso

### Ejemplo 1: Problema del Viajante de Comercio (TSP)

**Contexto:** Encontrar la ruta más corta que visita un conjunto de ciudades exactamente una vez y regresa a la ciudad de origen.

**Entrada:**
-   Codificación del individuo: Una permutación de las ciudades (ej. `[1, 3, 2, 4]` para 4 ciudades).
-   Función de aptitud: La inversa de la longitud total de la ruta (minimizar la longitud es maximizar la aptitud).

**Ejecución:**
1.  Se inicializa una población de rutas aleatorias.
2.  Las rutas se evalúan por su longitud.
3.  Las rutas más cortas tienen más posibilidades de ser seleccionadas.
4.  Se aplican operadores de cruce (ej. cruce de orden) y mutación (ej. intercambio de dos ciudades) para generar nuevas rutas.
5.  El proceso se repite hasta que se encuentra una ruta satisfactoria o se agotan las generaciones.

**Salida:** Una ruta aproximada que es una solución de alta calidad al TSP.

**Análisis:** Los AG son muy efectivos para encontrar soluciones de buena calidad para el TSP, un problema NP-hard, aunque no garantizan la optimalidad.

## Implementación

### En LatticeWeaver

**Módulo:** `lattice_weaver/algorithms/evolutionary_computation/genetic_algorithm.py`

**Interfaz:**
```python
from typing import List, Callable, Any, Tuple
import random

def genetic_algorithm(
    fitness_function: Callable[[List[Any]], float],
    individual_representation: Callable[[], List[Any]],
    population_size: int = 100,
    generations: int = 500,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.01,
    selection_method: str = 'tournament',
    tournament_size: int = 3
) -> Tuple[List[Any], float]:
    """
    Implementación genérica de un Algoritmo Genético.
    
    Args:
        fitness_function: Función que evalúa la aptitud de un individuo.
        individual_representation: Función que genera un individuo aleatorio.
        population_size: Tamaño de la población.
        generations: Número de generaciones a evolucionar.
        crossover_rate: Probabilidad de cruce.
        mutation_rate: Probabilidad de mutación.
        selection_method: Método de selección ('roulette' o 'tournament').
        tournament_size: Tamaño del torneo si se usa selección por torneo.
    
    Returns:
        Una tupla con el mejor individuo encontrado y su aptitud.
    
    Examples:
        >>> # Ejemplo simple: encontrar una cadena binaria de '1's
        >>> def binary_fitness(individual: List[int]) -> float:
        >>>     return sum(individual)
        >>> def binary_individual(length: int = 10) -> List[int]:
        >>>     return [random.randint(0, 1) for _ in range(length)]
        >>> 
        >>> best_individual, best_fitness = genetic_algorithm(
        >>>     fitness_function=binary_fitness,
        >>>     individual_representation=lambda: binary_individual(10),
        >>>     generations=50,
        >>>     population_size=50
        >>> )
        >>> print(f"Mejor individuo: {best_individual}, Aptitud: {best_fitness}")
    """
    # Implementación detallada de los pasos del AG
    # (Inicialización, Evaluación, Selección, Cruce, Mutación, Reemplazo)
    pass
```

### Dependencias

-   `random` (módulo estándar de Python) - Para la generación de números aleatorios y la selección probabilística.

### Tests

**Ubicación:** `tests/algorithms/evolutionary_computation/test_genetic_algorithm.py`

**Casos de test:**
1.  Test de problemas de optimización conocidos (ej. Problema de la Mochila, TSP para N pequeño).
2.  Test de convergencia de la aptitud a lo largo de las generaciones.
3.  Test de robustez con diferentes tasas de cruce y mutación.
4.  Test de casos borde (población pequeña, pocas generaciones).

## Visualización

### Visualización de la Ejecución

Una visualización de la evolución de la población a lo largo de las generaciones, mostrando la distribución de la aptitud, la diversidad de los individuos y cómo las soluciones convergen. Para problemas 2D, se puede mostrar la posición de los individuos en el espacio de búsqueda.

**Tipo de visualización:** Gráficos de aptitud (máxima, promedio), histogramas de diversidad, gráficos de dispersión 2D.

**Componentes:**
-   `matplotlib` o `plotly` para gráficos dinámicos.

### Visualización de Resultados

Mostrar el mejor individuo encontrado y cómo se compara con soluciones conocidas (si aplica). Para problemas como el TSP, visualizar la ruta óptima encontrada.

## Recursos

### Literatura Clave

#### Paper Original
-   Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.

#### Análisis y Mejoras
1.  Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
2.  Mitchell, M. (1996). *An Introduction to Genetic Algorithms*. MIT Press.

#### Aplicaciones
1.  Davis, L. (Ed.). (1991). *Handbook of Genetic Algorithms*. Van Nostrand Reinhold.

### Implementaciones Existentes

-   **DEAP (Distributed Evolutionary Algorithms in Python):** [https://deap.readthedocs.io/en/master/](https://deap.readthedocs.io/en/master/)
    -   **Lenguaje:** Python
    -   **Licencia:** LGPL
    -   **Notas:** Un *framework* completo para computación evolutiva, incluyendo AG, GP, etc.
-   **PyGAD:** [https://pygad.readthedocs.io/en/latest/](https://pygad.readthedocs.io/en/latest/)
    -   **Lenguaje:** Python
    -   **Licencia:** MIT
    -   **Notas:** Una biblioteca fácil de usar para implementar AG.

### Tutoriales y Recursos Educativos

-   **The Nature of Code - Genetic Algorithms:** [https://natureofcode.com/book/chapter-9-the-genetic-algorithm/](https://natureofcode.com/book/chapter-9-the-genetic-algorithm/) - Introducción interactiva y visual.
-   **Wikipedia - Genetic algorithm:** [https://en.wikipedia.org/wiki/Genetic_algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) - Descripción completa y referencias.

## Conexiones
#- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[T005]] - Conexión inversa con Técnica.
- [[D001]] - Conexión inversa con Dominio.
- [[D003]] - Conexión inversa con Dominio.

## Técnicas Relacionadas

-   [[T006]] - Recocido Simulado: Ambas son metaheurísticas de optimización global, pero los AG operan sobre una población y SA sobre un único individuo.
-   [[T003]] - Algoritmos de Monte Carlo: Los AG utilizan la aleatoriedad en sus operadores de mutación y selección, similar a los métodos de Monte Carlo.

### Conceptos Fundamentales

-   [[K009]] - Autoorganización: La emergencia de soluciones complejas a partir de interacciones simples en la población es un ejemplo de autoorganización.
-   [[K010]] - Emergencia: Las propiedades de la población y la convergencia hacia soluciones óptimas son propiedades emergentes del proceso evolutivo.

### Fenómenos Aplicables

-   [[F001]] - Teoría de Juegos Evolutiva: Los AG pueden modelar la evolución de estrategias en juegos repetidos.
-   [[F002]] - Redes de Regulación Génica: Pueden usarse para inferir la estructura de redes génicas o para optimizar su comportamiento.

## Historia y Evolución

### Desarrollo Histórico

-   **1960s:** John Holland desarrolla los fundamentos teóricos de los AG.
-   **1975:** Publicación de "Adaptation in Natural and Artificial Systems".
-   **1980s:** David Goldberg populariza los AG con su libro y aplicaciones a problemas de ingeniería.
-   **1990s en adelante:** Expansión a diversas áreas, desarrollo de variantes (GP, MAs) y aplicaciones en aprendizaje automático.

### Impacto

Los Algoritmos Genéticos han tenido un impacto profundo en el campo de la optimización y la inteligencia artificial, ofreciendo una poderosa herramienta para resolver problemas complejos que son intratables para los métodos tradicionales. Han demostrado ser particularmente útiles en diseño de ingeniería, finanzas, bioinformática y aprendizaje automático, donde la inspiración biológica ha proporcionado una heurística robusta y adaptable.

**Citaciones:** El trabajo de Holland y Goldberg es seminal en el campo de la computación evolutiva.
**Adopción:** Ampliamente adoptado en la academia y la industria para problemas de optimización, diseño y planificación.

## Estado de Implementación

-   [x] Pseudocódigo documentado
-   [x] Análisis de complejidad completado
-   [ ] Implementación en Python (sección de interfaz ya creada)
-   [ ] Tests unitarios
-   [ ] Tests de performance
-   [ ] Documentación de API
-   [ ] Ejemplos de uso
-   [ ] Visualización de ejecución
-   [ ] Tutorial

## Notas Adicionales

### Ideas para Mejora

-   Explorar diferentes esquemas de codificación para individuos (ej. codificación real, árboles de sintaxis).
-   Implementar operadores de cruce y mutación específicos para diferentes tipos de problemas.
-   Integrar mecanismos de paralelización para acelerar la evaluación de la población.

### Preguntas Abiertas

-   ¿Cómo se puede garantizar una diversidad adecuada en la población para evitar la convergencia prematura?
-   ¿Cuál es la relación óptima entre las tasas de cruce y mutación para diferentes problemas?

### Observaciones

La belleza de los Algoritmos Genéticos reside en su capacidad para resolver problemas complejos imitando un proceso tan fundamental y exitoso como la evolución biológica, demostrando la profunda conexión entre la naturaleza y la computación.

---

**Última actualización:** 2025-10-13  
**Responsable:** Manus AI  
**Validado por:** [Pendiente de revisión por experto humano]
- [[I004]]
- [[I007]]
- [[I008]]
- [[T002]]
- [[T004]]
- [[T003]]
- [[T006]]
- [[T007]]
- [[K002]]
- [[K003]]
- [[K008]]
- [[K009]]
- [[K010]]
- [[C002]]
- [[C003]]
- [[F001]]
- [[F002]]
