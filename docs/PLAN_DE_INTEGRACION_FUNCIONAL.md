# Plan de Integración Funcional: Tracks B y C

**Proyecto:** LatticeWeaver  
**Fecha:** 15 de Octubre, 2025  
**Autor:** Manus AI  
**Propósito:** Proponer estrategias para la **integración funcional** de las capacidades de los Tracks B y C en el flujo de resolución de problemas existente.

---

## 1. Introducción

Tras la confirmación de que los Tracks B (Locales y Frames) y C (Problem Families) están completamente integrados a nivel de código en la rama `main`, este documento aborda el siguiente paso crítico: la **integración funcional**. 

El objetivo no es solo tener el código disponible, sino **aprovechar activamente estas nuevas capacidades para enriquecer y optimizar el motor de resolución de problemas (ACE)**, creando nuevas vías para analizar y resolver CSPs de manera más eficiente e inteligente.

Se proponen tres estrategias principales de integración, diseñadas para ser implementadas de forma incremental.

---

## 2. Análisis de la Arquitectura Actual

El motor de resolución de problemas (solver) actual se encuentra en `lattice_weaver/core/csp_engine/solver.py`. Su arquitectura se basa en un algoritmo de **backtracking con forward checking**.

**Componentes Clave:**

- **`CSP` (en `core/csp_problem.py`):** Estructura de datos que define el problema (variables, dominios, restricciones).
- **`CSPSolver`:** Clase principal que orquesta la resolución.
- **`_backtrack()`:** Implementación recursiva del algoritmo de búsqueda.
- **`_is_consistent()`:** Verifica la consistencia de una asignación parcial.
- **`_select_unassigned_variable()`:** Heurística de selección de variable (actualmente, la más simple: la primera que encuentra).
- **`_forward_check()`:** Poda los dominios de variables vecinas tras una asignación.
- **`enforce_arc_consistency()`:** Implementación del algoritmo AC-3 para pre-procesamiento.

**Puntos de Mejora Identificados:**

1.  **Heurísticas de Búsqueda:** La selección de variables y valores es muy básica. No aprovecha la estructura del problema.
2.  **Análisis del Problema:** El solver no realiza un análisis estructural previo del CSP; simplemente ejecuta AC-3 y empieza la búsqueda.
3.  **Propagación de Restricciones:** El forward checking es una técnica de propagación de bajo nivel. Podrían explorarse métodos más potentes.

---

## 3. Estrategias de Integración Funcional

### Estrategia 1: Análisis Topológico del Espacio de Búsqueda (Integración de Track B)

**Concepto:** Utilizar el módulo `topology_new` para realizar un análisis cualitativo del espacio de búsqueda de un CSP **antes** de iniciar el proceso de resolución. Esto permite entender la "forma" del problema y tomar decisiones más informadas.

**Implementación:**

1.  **Crear un `MetaSolver` o `AnalysisEngine`** que envuelva al `CSPSolver`.
2.  Antes de llamar a `solve()`, este motor utilizará el `ACELocaleBridge` para convertir el CSP en un `Locale`.
3.  Se invocarán las funciones de análisis del bridge, como `analyze_consistency_topology()`.

**Puntos de Integración:**

- **Pre-procesamiento:** El análisis topológico se convierte en un paso de pre-procesamiento, complementario a AC-3.
- **Guía de Heurísticas:** Los resultados del análisis (ej. "densidad de soluciones", "conectividad del espacio") pueden usarse para seleccionar dinámicamente la mejor heurística de búsqueda (ver Estrategia 2).
- **Detección Temprana de Inconsistencias:** El análisis puede revelar que un problema no tiene solución (espacio vacío) sin necesidad de explorar el árbol de búsqueda.

**Beneficios:**

- **Visión Global:** Proporciona una comprensión estructural del problema que el backtracking por sí solo no tiene.
- **Búsqueda Inteligente:** Permite adaptar la estrategia de búsqueda a la naturaleza del problema.
- **Potencial de Optimización:** Identifica subproblemas aislados que podrían resolverse en paralelo.

**Ejemplo de Flujo:**

```python
from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.topology_new.ace_bridge import ACELocaleBridge
from lattice_weaver.core.csp_engine.solver import CSPSolver

# 1. Se recibe o genera un problema CSP
csp: CSP = ...

# 2. Análisis Topológico (NUEVO PASO)
bridge = ACELocaleBridge()
locale = bridge.csp_to_locale(csp)
analysis = bridge.analyze_consistency_topology(locale)

# 3. Selección de Estrategia (NUEVO PASO)
if analysis["is_trivial"]:
    # Resolver directamente
    ...
elif analysis["solution_density"] > 0.5:
    # Usar heurística para zonas densas
    heuristic = "least_constraining_value"
else:
    # Usar heurística para zonas dispersas
    heuristic = "minimum_remaining_values"

# 4. Resolución
solver = CSPSolver(csp, heuristic=heuristic)
solution = solver.solve()
```

---

### Estrategia 2: Heurísticas de Búsqueda Basadas en Familias de Problemas (Integración de Track C)

**Concepto:** Aprovechar el `ProblemCatalog` del Track C para identificar el tipo de problema que se está resolviendo y aplicar heurísticas de búsqueda especializadas y probadamente eficientes para esa familia.

**Implementación:**

1.  **Extender el `CSPSolver`** para que acepte funciones de heurística como parámetros en su constructor (ej. `variable_heuristic`, `value_heuristic`).
2.  **Crear un módulo de heurísticas** (`lattice_weaver/core/csp_engine/heuristics.py`) que contenga implementaciones de heurísticas comunes (MRV, Degree, LCV, etc.).
3.  **Modificar el `MetaSolver`** para que, al recibir un problema del `ProblemCatalog`, consulte sus metadatos (`problem.metadata['family']`) y seleccione la heurística adecuada del módulo de heurísticas.

**Puntos de Integración:**

- **`_select_unassigned_variable()`:** Reemplazar la implementación actual por una llamada a la función de heurística de variable seleccionada.
- **`_backtrack()`:** Modificar el bucle de valores para que siga el orden de la heurística de valor seleccionada.
- **`ProblemCatalog`:** El catálogo se convierte en la fuente de conocimiento para la selección de estrategias.

**Beneficios:**

- **Rendimiento Exponencialmente Mejor:** Usar la heurística correcta para un problema puede reducir el espacio de búsqueda en órdenes de magnitud.
- **Automatización del Conocimiento Experto:** Codifica el conocimiento de qué heurísticas funcionan mejor para qué problemas.
- **Extensibilidad:** Es fácil añadir nuevas familias y asociarles nuevas heurísticas.

**Ejemplo de Mapeo de Heurísticas:**

| Familia de Problema | Heurística de Variable Recomendada | Heurística de Valor Recomendada |
| :--- | :--- | :--- |
| Graph Coloring | `MRV` (Minimum Remaining Values) + `Degree` | `LCV` (Least Constraining Value) |
| N-Queens | `MRV` | `LCV` |
| Sudoku | `MRV` | `LCV` |
| Scheduling | `MostConstrained` | `FirstAvailable` |

---

### Estrategia 3: Propagación de Restricciones con Operadores Modales (Integración Avanzada de Track B)

**Concepto:** Ir más allá del `forward_checking` y utilizar los operadores modales (◇, □) del módulo `topology_new.operations` para realizar una propagación de restricciones más potente, razonando sobre "regiones" del espacio de búsqueda en lugar de valores individuales.

**Implementación (Conceptual):**

1.  Durante la búsqueda, en lugar de asignar un solo valor (`var = value`), se define una **región** (un subconjunto del espacio de soluciones).
2.  Se aplican los operadores modales para **propagar esta restricción topológica** a otras variables, podando regiones enteras del espacio de búsqueda que se sabe que son inconsistentes.
3.  Esto requiere una integración mucho más profunda con el bucle de `_backtrack`, potencialmente creando un nuevo tipo de solver (`TopologicalSolver`).

**Beneficios:**

- **Poda Más Agresiva:** Capaz de eliminar grandes áreas del espacio de búsqueda de una sola vez.
- **Razonamiento de Nivel Superior:** Permite un razonamiento más abstracto sobre la consistencia del problema.
- **Fundamento para Sheaves:** Sienta las bases para la futura implementación de la Teoría de Sheaves (Meseta 2), que razona sobre la consistencia local y global.

**Estado:** Esta es una estrategia avanzada que requiere más investigación. Se propone como una línea de desarrollo a largo plazo.

---

## 4. Plan de Implementación Incremental

Se propone el siguiente plan para implementar estas integraciones de forma progresiva:

### Fase 1: Implementación de Heurísticas (Track C)

1.  **Refactorizar `CSPSolver`:** Modificar el constructor para aceptar funciones de heurística personalizadas.
2.  **Crear Módulo de Heurísticas:** Implementar `mrv`, `degree`, `lcv` en `heuristics.py`.
3.  **Crear `MetaSolver`:** Implementar la lógica que lee los metadatos del problema y selecciona la heurística.
4.  **Tests:** Crear tests que verifiquen que se llama a la heurística correcta para cada familia de problema.

**Entregable:** Un solver significativamente más rápido para las familias de problemas conocidas.

### Fase 2: Implementación de Análisis Topológico (Track B)

1.  **Integrar `ACELocaleBridge`:** Añadir el paso de análisis topológico en el `MetaSolver`.
2.  **Conectar Análisis con Heurísticas:** Usar los resultados del análisis para tomar decisiones sobre qué heurística o estrategia de solver usar.
3.  **Tests:** Añadir tests que verifiquen que el análisis se realiza y que sus resultados influyen en la estrategia de resolución.

**Entregable:** Un solver que adapta su estrategia basándose en la estructura inherente del problema.

### Fase 3: Investigación de Propagación Modal (Track B Avanzado)

1.  **Prototipo:** Desarrollar un prototipo de `TopologicalSolver` que utilice operadores modales para la propagación.
2.  **Benchmarking:** Comparar el rendimiento del `TopologicalSolver` con el solver de backtracking tradicional en problemas seleccionados.
3.  **Documento de Diseño:** Redactar un documento técnico detallando la arquitectura y los algoritmos del `TopologicalSolver`.

**Entregable:** Un prototipo funcional y un documento de diseño para la siguiente generación del motor de resolución.

---

## 5. Conclusión

La integración funcional de los Tracks B y C transformará a LatticeWeaver de una colección de módulos potentes a un **framework de resolución de problemas verdaderamente inteligente y adaptable**. Al implementar estas estrategias, el motor de resolución podrá:

- **Seleccionar automáticamente la mejor herramienta** para el trabajo basándose en la familia del problema.
- **Analizar la estructura profunda** del espacio de búsqueda para guiar su estrategia.
- **Sentar las bases para técnicas de razonamiento de nivel superior** como la propagación modal y la teoría de Sheaves.

Este plan proporciona una hoja de ruta clara y pragmática para realizar esta visión, maximizando el valor de los desarrollos ya completados.
