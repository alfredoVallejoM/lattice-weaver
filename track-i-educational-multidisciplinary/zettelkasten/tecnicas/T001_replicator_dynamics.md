---
id: T001
tipo: tecnica
titulo: Replicator Dynamics
dominio_origen: biologia,economia
categorias_aplicables: [C004]
tags: [sistemas_dinamicos, teoria_de_juegos_evolutiva, modelado_poblacional, evolucion, seleccion_natural, estabilidad]
fecha_creacion: 2025-10-13
fecha_modificacion: 2025-10-13
estado: completo  # borrador | en_revision | completo
implementado: false  # true | false
---

# Técnica: Replicator Dynamics

## Descripción

La **Dinámica del Replicador (Replicator Dynamics)** es un conjunto de ecuaciones diferenciales (o de diferencia, en tiempo discreto) que describen cómo las proporciones de diferentes estrategias (o fenotipos) en una población cambian con el tiempo. Se utiliza principalmente en la Teoría de Juegos Evolutiva para modelar la selección natural o social, donde las estrategias más exitosas (aquellas con mayor *fitness* o pago promedio) se replican más rápidamente y, por lo tanto, aumentan su proporción en la población. Es una herramienta fundamental para analizar la estabilidad de las estrategias y la evolución de los sistemas complejos.

## Origen

**Dominio de origen:** [[D001]] - Biología (Teoría de Juegos Evolutiva), [[D002]] - Economía (Teoría de Juegos)
**Año de desarrollo:** Finales de los 70 y principios de los 80.
**Desarrolladores:** John Maynard Smith, George R. Price, Peter Schuster, Karl Sigmund, entre otros.
**Contexto:** Desarrollada para formalizar la evolución de estrategias en poblaciones biológicas, extendiendo los conceptos de la teoría de juegos clásica a escenarios donde los agentes no son necesariamente racionales, sino que sus estrategias se propagan en función de su éxito relativo. Posteriormente, se aplicó a la economía y las ciencias sociales para modelar la evolución de normas y comportamientos.

## Formulación

### Entrada

-   **Matriz de pagos (A):** Una matriz que define los pagos que obtiene una estrategia al interactuar con otra. Para un juego simétrico de dos jugadores, si hay `n` estrategias, `A` es una matriz `n x n` donde `A_ij` es el pago que obtiene la estrategia `i` al interactuar con la estrategia `j`.
-   **Vector de proporciones poblacionales (x):** Un vector `n`-dimensional donde `x_i` es la proporción de la estrategia `i` en la población, y `sum(x_i) = 1`.

### Salida

-   **Vector de tasas de cambio (dx/dt o x_t+1):** Describe cómo cambian las proporciones de cada estrategia en la población a lo largo del tiempo.
-   **Puntos de equilibrio:** Configuraciones de proporciones poblacionales donde el sistema se estabiliza (dx/dt = 0).

### Parámetros

| Parámetro | Tipo    | Rango     | Descripción                                    | Valor por defecto |
|-----------|---------|-----------|------------------------------------------------|-------------------|
| `A`       | Matriz  | `n x n`   | Matriz de pagos del juego                      | N/A               |
| `x0`      | Vector  | `sum(x_i)=1` | Proporciones iniciales de las estrategias     | N/A               |
| `dt`      | Flotante | `> 0`     | Tamaño del paso de tiempo (para discreto)      | 0.01              |
| `T`       | Entero  | `> 0`     | Número de iteraciones/tiempo total (para discreto) | 1000              |

## Algoritmo

### Pseudocódigo (Tiempo Continuo)

```
ALGORITMO ReplicatorDynamicsContinuo(matriz_pagos A, proporciones_iniciales x0, tiempo_total T)
    ENTRADA: Matriz de pagos A, vector de proporciones x0, tiempo total T
    SALIDA: Evolución de las proporciones x(t)
    
    x = x0
    PARA t DESDE 0 HASTA T CON PASO dt HACER
        pago_promedio_estrategia = A * x  // Vector de pagos promedio para cada estrategia
        pago_promedio_poblacion = x . pago_promedio_estrategia // Producto escalar
        
        PARA cada estrategia i HACER
            dx_i/dt = x_i * (pago_promedio_estrategia[i] - pago_promedio_poblacion)
        FIN PARA
        
        x = x + (dx/dt) * dt // Actualizar proporciones
        Normalizar x para que sum(x_i) = 1
    FIN PARA
    RETORNAR historial de x(t)
FIN ALGORITMO
```

### Pseudocódigo (Tiempo Discreto)

```
ALGORITMO ReplicatorDynamicsDiscreto(matriz_pagos A, proporciones_iniciales x0, num_iteraciones N)
    ENTRADA: Matriz de pagos A, vector de proporciones x0, número de iteraciones N
    SALIDA: Evolución de las proporciones x_t
    
    x = x0
    PARA t DESDE 0 HASTA N-1 HACER
        pago_promedio_estrategia = A * x
        pago_promedio_poblacion = x . pago_promedio_estrategia
        
        PARA cada estrategia i HACER
            x_i_siguiente = x_i * (pago_promedio_estrategia[i] / pago_promedio_poblacion)
        FIN PARA
        
        x = x_siguiente
        Normalizar x para que sum(x_i) = 1
    FIN PARA
    RETORNAR historial de x_t
FIN ALGORITMO
```

### Descripción Paso a Paso

1.  **Calcular el pago promedio de cada estrategia:** Para cada estrategia `i`, se calcula su pago promedio al interactuar con la población actual. Esto se hace multiplicando la fila `i` de la matriz de pagos `A` por el vector de proporciones `x`.
2.  **Calcular el pago promedio de la población:** Se calcula el pago promedio de toda la población, que es el producto escalar del vector de proporciones `x` por el vector de pagos promedio de cada estrategia.
3.  **Actualizar las proporciones:** La tasa de cambio de la proporción de una estrategia `i` es proporcional a su proporción actual y a la diferencia entre su pago promedio y el pago promedio de la población. Si una estrategia tiene un pago superior al promedio, su proporción aumenta; si es inferior, disminuye.
4.  **Normalizar:** Las proporciones se normalizan para asegurar que su suma siempre sea 1.
5.  **Iterar:** Los pasos se repiten hasta alcanzar un tiempo total o un número de iteraciones predefinido, o hasta que el sistema converge a un equilibrio.

### Invariantes

1.  **Suma de proporciones:** La suma de las proporciones de todas las estrategias siempre es 1 (`sum(x_i) = 1`).
2.  **No negatividad:** Las proporciones de las estrategias siempre son no negativas (`x_i >= 0`).
3.  **Puntos de equilibrio:** Los puntos donde el pago promedio de cada estrategia es igual al pago promedio de la población son puntos fijos de la dinámica.

## Análisis

### Complejidad Temporal

-   **Mejor caso:** O(N * n^2) para `N` iteraciones y `n` estrategias (dominado por la multiplicación matriz-vector).
-   **Caso promedio:** O(N * n^2)
-   **Peor caso:** O(N * n^2)

**Justificación:** En cada paso de tiempo, la operación dominante es la multiplicación de la matriz de pagos `n x n` por el vector de proporciones `n x 1`, que es O(n^2). Esto se repite `N` veces.

### Complejidad Espacial

-   **Espacio auxiliar:** O(n) (para almacenar vectores temporales de pagos).
-   **Espacio total:** O(n^2) (para almacenar la matriz de pagos `A` y el vector de proporciones `x`).

**Justificación:** La matriz de pagos `A` requiere O(n^2) espacio. Los vectores de proporciones y pagos requieren O(n) espacio.

### Corrección

**Teorema:** La Dinámica del Replicador modela la selección natural/social donde las estrategias con fitness superior al promedio aumentan su proporción en la población.
**Demostración:** La formulación matemática se deriva directamente de la suposición de que la tasa de crecimiento de una estrategia es proporcional a su fitness relativo. Los puntos fijos de la dinámica corresponden a los Equilibrios de Nash simétricos y las Estrategias Evolutivamente Estables (ESS) bajo ciertas condiciones.

### Optimalidad

La Dinámica del Replicador no busca una solución óptima en el sentido de maximizar una función objetivo global, sino que describe un proceso evolutivo. Sin embargo, los puntos de equilibrio de la dinámica a menudo corresponden a soluciones estables o óptimas localmente en el contexto de la teoría de juegos evolutiva (ej. ESS).

## Aplicabilidad

### Categorías Estructurales Aplicables

1.  [[C004]] - Sistemas Dinámicos
    -   **Por qué funciona:** La Dinámica del Replicador es intrínsecamente un sistema dinámico que describe la evolución de estados (proporciones de estrategias) a lo largo del tiempo.
    -   **Limitaciones:** Asume poblaciones bien mezcladas o interacciones aleatorias, y no siempre captura la complejidad de las interacciones en redes estructuradas.

2.  [[C001]] - Redes de Interacción
    -   **Por qué funciona:** Puede extenderse para modelar la evolución de estrategias en redes, donde el fitness de un individuo depende de las interacciones con sus vecinos.
    -   **Limitaciones:** La formulación estándar no considera explícitamente la topología de la red, requiriendo extensiones como la Dinámica del Replicador en grafos.

### Fenómenos Donde Se Ha Aplicado

#### En Dominio Original (Biología/Economía)

-   [[F001]] - Teoría de Juegos Evolutiva
    -   **Resultado:** Análisis de la evolución de la cooperación (ej. Dilema del Prisionero), la dinámica de poblaciones de halcones y palomas, y la emergencia de estrategias estables.
    -   **Referencias:** Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life*.

#### Transferencias a Otros Dominios

-   [[F009]] - Modelo de Votantes
    -   **Adaptaciones necesarias:** Los "pagos" se interpretan como la influencia o la probabilidad de ser imitado. Las "estrategias" son las opiniones.
    -   **Resultado:** Modelado de la difusión de opiniones y la polarización social, donde las opiniones más "exitosas" (más influyentes) se propagan.
    -   **Referencias:** Castellano, C., Fortunato, S., & Loreto, V. (2009). Statistical physics of social dynamics.

-   [[F010]] - Segregación urbana (Schelling)
    -   **Adaptaciones necesarias:** Los agentes adoptan una "estrategia" (ubicación/vecindario) en función de la satisfacción con sus vecinos, lo que puede ser mapeado a un fitness.
    -   **Resultado:** Modelado de cómo las preferencias individuales por la similitud pueden llevar a patrones de segregación a nivel macroscópico.

### Prerequisitos

1.  **Juego bien definido:** Debe existir una matriz de pagos que describa las interacciones entre las estrategias.
2.  **Población grande:** La formulación en tiempo continuo asume una población infinita o muy grande. Para poblaciones finitas, se usan variantes estocásticas.
3.  **Interacciones aleatorias/bien mezcladas:** La versión estándar asume que los individuos interactúan aleatoriamente con cualquier otro en la población.

### Contraindicaciones

1.  **Interacciones estructuradas complejas:** Si la topología de la red de interacción es crucial y no puede ser aproximada por un modelo bien mezclado, se necesitan extensiones de la dinámica del replicador.
2.  **Agentes racionales:** Si los agentes toman decisiones estratégicas conscientes y racionales, la teoría de juegos clásica o la teoría de juegos evolutiva con aprendizaje explícito pueden ser más apropiadas.
3.  **Cambios rápidos en la matriz de pagos:** Si la matriz de pagos cambia dinámicamente en función de las proporciones de estrategias, la dinámica puede volverse más compleja.

## Variantes

### Variante 1: Dinámica del Replicador en Grafos

**Modificación:** En lugar de interacciones aleatorias en una población bien mezclada, los individuos interactúan solo con sus vecinos en una red. El fitness de un individuo se calcula en función de las interacciones con sus vecinos, y la replicación ocurre localmente.
**Ventaja:** Más realista para muchos sistemas biológicos y sociales donde las interacciones son locales.
**Desventaja:** El análisis matemático es más complejo y a menudo requiere simulaciones.
**Cuándo usar:** Cuando la estructura de la red de interacción es un factor clave en la evolución de las estrategias.

### Variante 2: Dinámica del Replicador Estocástica

**Modificación:** Incorpora ruido o aleatoriedad en el proceso de replicación, adecuado para poblaciones finitas. A menudo se modela como un proceso de Moran o un proceso de Wright-Fisher.
**Ventaja:** Captura la influencia de la deriva genética o el azar en poblaciones pequeñas.
**Desventaja:** Pierde la simplicidad analítica de la dinámica determinista.
**Cuándo usar:** Para modelar la evolución en poblaciones pequeñas o cuando el ruido es un factor importante.

## Comparación con Técnicas Alternativas

### Técnica Alternativa 1: [[T007]] - Simulación Basada en Agentes (SBA)

| Criterio              | Replicator Dynamics | Simulación Basada en Agentes |
|-----------------------|---------------------|------------------------------|
| Complejidad temporal  | O(N * n^2)          | O(N * M * k) (M agentes, k interacciones) |
| Complejidad espacial  | O(n^2)              | O(M)                         |
| Facilidad de implementación | Media               | Media-Alta                   |
| Calidad de solución   | Analítica, determinista | Estocástica, emergente       |
| Aplicabilidad         | Poblaciones bien mezcladas, análisis de equilibrio | Interacciones locales, heterogeneidad, redes complejas |

**Cuándo preferir esta técnica (Replicator Dynamics):** Para un análisis rápido y determinista de la evolución de estrategias en poblaciones grandes y bien mezcladas, o para encontrar equilibrios estables.
**Cuándo preferir la alternativa (SBA):** Cuando la heterogeneidad de los agentes, las interacciones locales en redes complejas o el comportamiento emergente son cruciales y no pueden ser capturados por un modelo determinista.

### Técnica Alternativa 2: [[T003]] - Algoritmos de Monte Carlo

| Criterio              | Replicator Dynamics | Algoritmos de Monte Carlo |
|-----------------------|---------------------|---------------------------|
| Complejidad temporal  | O(N * n^2)          | Variable, depende de la simulación |
| Complejidad espacial  | O(n^2)              | Variable                  |
| Facilidad de implementación | Media               | Media-Alta                |
| Calidad de solución   | Analítica, determinista | Estocástica, aproximada   |
| Aplicabilidad         | Dinámica determinista | Sistemas estocásticos, muestreo de espacios de estados |

**Cuándo preferir esta técnica (Replicator Dynamics):** Para obtener una visión general de la dinámica determinista y los puntos de equilibrio.
**Cuándo preferir la alternativa (Monte Carlo):** Para simular sistemas estocásticos, explorar el espacio de estados de sistemas complejos o cuando no hay una formulación analítica sencilla.

## Ejemplos de Uso

### Ejemplo 1: Juego Halcón-Paloma

**Contexto:** Un juego simétrico donde dos estrategias, Halcón (H) y Paloma (P), compiten por un recurso. Los Halcones siempre luchan, las Palomas siempre ceden. Si dos Halcones luchan, ambos sufren un costo. Si un Halcón y una Paloma interactúan, el Halcón gana el recurso y la Paloma no sufre costo. Si dos Palomas interactúan, comparten el recurso.

**Matriz de Pagos (A):**
```
     H    P
H  (V-C)/2  V
P    0    V/2
```
Donde V es el valor del recurso y C es el costo de la lucha (V < C).

**Entrada:** `A = [[(V-C)/2, V], [0, V/2]]`, `x0 = [0.5, 0.5]` (50% Halcones, 50% Palomas)

**Ejecución (simulación):** La Dinámica del Replicador mostrará cómo la proporción de Halcones y Palomas evoluciona. Si `V > C`, los Halcones dominarán. Si `V < C`, el sistema convergerá a una mezcla estable de Halcones y Palomas (ESS de estrategia mixta).

**Salida:** Un gráfico de `x_H(t)` y `x_P(t)` que converge a un punto de equilibrio.

**Análisis:** Este ejemplo ilustra cómo la Dinámica del Replicador puede predecir la emergencia de una Estrategia Evolutivamente Estable (ESS) de estrategia mixta, donde ninguna estrategia pura puede invadir a la otra.

## Implementación

### En LatticeWeaver

**Módulo:** `lattice_weaver/algorithms/game_theory/replicator_dynamics.py`

**Interfaz:**
```python
import numpy as np

def replicator_dynamics(
    payoff_matrix: np.ndarray,
    initial_proportions: np.ndarray,
    time_steps: int,
    dt: float = 0.01
) -> np.ndarray:
    """
    Simula la Dinámica del Replicador para un juego simétrico.
    
    Args:
        payoff_matrix: Matriz de pagos (n x n) donde A[i,j] es el pago de la estrategia i contra la estrategia j.
        initial_proportions: Vector inicial de proporciones de estrategias (suma a 1).
        time_steps: Número de pasos de tiempo para la simulación.
        dt: Tamaño del paso de tiempo para la integración.
    
    Returns:
        Un array 2D con la evolución de las proporciones de estrategias a lo largo del tiempo.
    
    Raises:
        ValueError: Si la matriz de pagos no es cuadrada o las proporciones iniciales no suman 1.
    
    Examples:
        >>> # Juego Halcón-Paloma (V=2, C=4)
        >>> A = np.array([[-1, 2], [0, 1]])
        >>> x0 = np.array([0.5, 0.5])
        >>> history = replicator_dynamics(A, x0, 1000, dt=0.01)
        >>> print(history[-1]) # Proporciones finales
    """
    num_strategies = payoff_matrix.shape[0]
    if payoff_matrix.shape[1] != num_strategies:
        raise ValueError("La matriz de pagos debe ser cuadrada.")
    if not np.isclose(np.sum(initial_proportions), 1.0):
        raise ValueError("Las proporciones iniciales deben sumar 1.")

    history = np.zeros((time_steps, num_strategies))
    x = initial_proportions.copy()
    history[0] = x

    for t in range(1, time_steps):
        # Pago promedio de cada estrategia
        fitness_i = np.dot(payoff_matrix, x)
        # Pago promedio de la población
        avg_fitness = np.dot(x, fitness_i)
        
        # Ecuaciones del replicador
        dx_dt = x * (fitness_i - avg_fitness)
        
        # Actualizar proporciones
        x = x + dx_dt * dt
        
        # Asegurar que las proporciones no sean negativas y sumen 1
        x[x < 0] = 0
        x = x / np.sum(x)
        history[t] = x
        
    return history
```

### Dependencias

-   `numpy` - Para operaciones numéricas eficientes con matrices y vectores.

### Tests

**Ubicación:** `tests/algorithms/game_theory/test_replicator_dynamics.py`

**Casos de test:**
1.  Test de corrección con juego Halcón-Paloma (convergencia a ESS de estrategia mixta).
2.  Test de juego con una estrategia dominante (convergencia a estrategia pura).
3.  Test de casos borde (proporciones iniciales muy sesgadas).
4.  Test de validación de invariantes (suma de proporciones = 1, no negatividad).
5.  Test de errores (matriz no cuadrada, proporciones no válidas).

## Visualización

### Visualización de la Ejecución

Una animación que muestre la evolución de las proporciones de las estrategias en un simplex (para 3 estrategias) o un gráfico de líneas (para cualquier número de estrategias) a lo largo del tiempo. Se pueden resaltar los puntos de equilibrio.

**Tipo de visualización:** Gráfico de líneas o Simplex dinámico.

**Componentes:**
-   `matplotlib` o `plotly` para gráficos.
-   `scipy.integrate` para soluciones más precisas de ODEs si es necesario.

### Visualización de Resultados

Gráficos de la trayectoria de las proporciones de estrategias y un diagrama de fase (para 2 o 3 estrategias) que muestre los atractores y repulsores.

## Recursos

### Literatura Clave

#### Paper Original
-   Taylor, P. D., & Jonker, L. B. (1978). Evolutionary stable strategies and game dynamics. *Mathematical Biosciences*, 40(1-2), 145-156.
-   Schuster, P., & Sigmund, K. (1983). Replicator dynamics. *Journal of Theoretical Biology*, 100(3), 533-538.

#### Análisis y Mejoras
1.  Hofbauer, J., & Sigmund, K. (1998). *Evolutionary Games and Population Dynamics*. Cambridge University Press.
2.  Weibull, J. W. (1995). *Evolutionary Game Theory*. MIT Press.

#### Aplicaciones
1.  Gintis, H. (2009). *Game Theory Evolving: A Problem-Centered Introduction to Modeling Strategic Behavior*. Princeton University Press. (Aplicaciones en economía y ciencias sociales).

### Implementaciones Existentes

-   **Axelrod's Library (Python):** [https://github.com/Axelrod-Python/Axelrod](https://github.com/Axelrod-Python/Axelrod)
    -   **Lenguaje:** Python
    -   **Licencia:** MIT
    -   **Notas:** Implementa varios juegos evolutivos y dinámicas, incluyendo variantes del replicador.

### Tutoriales y Recursos Educativos

-   **Stanford Encyclopedia of Philosophy - Evolutionary Game Theory:** [https://plato.stanford.edu/entries/game-evolutionary/](https://plato.stanford.edu/entries/game-evolutionary/) - Excelente introducción a la TJE y la Dinámica del Replicador.
-   **
- [[F001]]
- [[I003]]
- [[I006]]
- [[T003]]
- [[T007]]
- [[K001]]
- [[K002]]
- [[K004]]
- [[K005]]
- [[C001]]
- [[C004]]
- [[F009]]
- [[F010]]

## Conexiones
- [[T001]] - Conexión inversa con Técnica.
- [[D001]] - Conexión inversa con Dominio.
- [[D002]] - Conexión inversa con Dominio.
