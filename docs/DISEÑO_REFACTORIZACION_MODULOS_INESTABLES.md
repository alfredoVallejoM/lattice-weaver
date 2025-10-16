# Diseño en Profundidad para la Refactorización de Módulos Inestables

**Fecha:** 16 de Octubre, 2025  
**Autor:** Manus AI  
**Versión:** 1.0

---

## 1. Introducción

Este documento detalla el diseño en profundidad para la refactorización de los módulos identificados como inestables o con fallos preexistentes en el proyecto LatticeWeaver. El objetivo principal es restaurar la funcionalidad completa y la estabilidad de estas partes, asegurando su coherencia con la arquitectura actual del proyecto y los principios de diseño establecidos. Se priorizará la estabilidad del core existente y, en caso de conflicto entre la versión estable y la lógica de los módulos inestables, se reimplementará la lógica inestable desde cero, adaptándola a las APIs y estructuras de datos actuales.

El análisis previo (`ANALISIS_ORIGEN_FALLOS.md` y `ANALISIS_ESTABILIDAD_GLOBAL.md`) reveló que los fallos no fueron introducidos por la Fase 7.0, sino que eran problemas preexistentes. Este diseño abordará sistemáticamente cada categoría de fallo, proponiendo soluciones que minimicen el riesgo y aseguren la integración exitosa.

---

## 2. Meta-Principios de Diseño y Máximas

La refactorización se guiará por los meta-principios de diseño de LatticeWeaver, asegurando que las soluciones sean robustas, escalables y mantenibles. En caso de duda, se priorizará la simplicidad, la modularidad y la coherencia con el core estable del proyecto.

### 2.1. Principios Fundamentales

| Principio | Descripción | Aplicación en la Refactorización |
|:----------|:------------|:---------------------------------|
| **Dinamismo** | Adaptabilidad a cambios, clustering dinámico, renormalización. | Las nuevas implementaciones deben ser flexibles y permitir futuras extensiones sin romper la compatibilidad. Evitar soluciones rígidas. |
| **Distribución/Paralelización** | Escalabilidad horizontal, arquitectura Ray, actores distribuidos. | Aunque no es el enfoque principal de esta refactorización, las soluciones deben ser diseñadas pensando en la futura paralelización y no introducir cuellos de botella inherentes. |
| **No Redundancia/Canonicalización** | Evitar duplicidades, caché de isomorfismo, memoización, PEC. | Reimplementar solo lo necesario, reutilizando componentes existentes. Implementar mecanismos de caché cuando sea apropiado para evitar recálculos. |
| **Aprovechamiento de la Información** | Maximizar uso de datos, no-good learning, KnowledgeSheaf. | Asegurar que los módulos refactorizados puedan interactuar eficientemente con el `CSPSolver` y otros componentes para compartir y utilizar la información de manera óptima. |
| **Gestión de Memoria Eficiente** | Minimizar consumo, object pooling, poda. | Las reimplementaciones deben ser conscientes del uso de memoria, especialmente en estructuras de datos grandes o recursivas. |
| **Economía Computacional** | Optimizar el rendimiento y el uso de recursos. | Las soluciones deben ser eficientes, buscando el mejor equilibrio entre complejidad y rendimiento. |
| **Generalidad y Modularidad** | Uso de técnicas de ingeniería de software, código reutilizable y desacoplado. | Cada módulo refactorizado debe tener una responsabilidad clara y una interfaz bien definida, minimizando las dependencias directas. |
| **Automatización de Patrones** | Identificación y aplicación de patrones de diseño. | Utilizar patrones de diseño reconocidos para resolver problemas comunes, facilitando la comprensión y el mantenimiento. |

### 2.2. Máximas Operacionales

1.  **Prioridad a lo Estable:** En caso de conflicto, la lógica y la API de los módulos estables (Fibration, Arc Engine, CSPSolver) tienen prioridad. Los módulos inestables se adaptarán a estos.  
2.  **Reimplementación sobre Parches:** Si un módulo inestable tiene un diseño fundamentalmente defectuoso o una API incompatible, se optará por la reimplementación desde cero, conservando la lógica funcional pero adaptándola a la arquitectura actual.  
3.  **Tests Primero:** Cada refactorización o reimplementación estará acompañada de tests unitarios y de integración robustos para asegurar la corrección y prevenir futuras regresiones.  
4.  **Integración Incremental:** Las correcciones se realizarán en pequeñas fases, permitiendo una validación continua y una fácil reversión en caso de problemas.  
5.  **Documentación Clara:** Cada cambio importante estará acompañado de documentación actualizada que refleje la nueva lógica, APIs y decisiones de diseño.

---

## 3. Diseño de Refactorización por Módulo

### 3.1. Categoría: Core CSP y Generadores de Problemas

#### 3.1.1. Módulo: `problems/generators` (API de CSP)

*   **Problema Actual:** Los generadores de problemas (`graph_coloring.py`, `nqueens.py`, `sudoku.py`) fallan debido a un cambio en la API de la clase `CSP`. Anteriormente, se asignaban variables y dominios directamente a un diccionario `CSP.variables`. Ahora, `CSP.variables` es un `Set[str]` y los dominios se gestionan a través de `CSP.domains` y el método `CSP.add_variable(name, domain)`.  
*   **Funcionalidad Deseada:** Los generadores deben poder crear instancias de `CSP` válidas que puedan ser resueltas por el `CSPSolver`.  
*   **Impacto:** 40+ tests fallando, incapacidad para generar problemas estándar para benchmarking y pruebas.  
*   **Propuesta de Refactorización:**  
    *   **Lógica:** La lógica de generación de problemas (cómo se construyen los grafos, las restricciones de N-Reinas, etc.) es correcta y se mantendrá.  
    *   **Reimplementación:** Se modificará la sección donde se añaden variables y dominios al objeto `CSP` dentro de cada generador.  
    *   **Detalle:** Reemplazar todas las instancias de `engine.variables[var_name] = domain` por `engine.add_variable(var_name, domain)`.  
    *   **Coherencia:** Esto se alinea con la API actual de `CSP` y `CSPSolver`, que son módulos estables.  
*   **Dependencias:** `lattice_weaver/core/csp_problem.py`  
*   **Estimación de Esfuerzo:** 2-3 horas.  

### 3.2. Categoría: Módulos Formales (CSP-Cubical Integration)

#### 3.2.1. Módulo: `formal/cubical_types.py` y `formal/cubical_csp_type.py` (Jerarquía de `FiniteType`)

*   **Problema Actual:** La clase `FiniteType` en `cubical_csp_type.py` hereda de `CubicalFiniteType` en `cubical_types.py`. `CubicalFiniteType` espera un argumento `size` en su constructor, mientras que `FiniteType` define `name` y `values`. Esto causa un `TypeError` cuando se intenta instanciar `FiniteType` desde `CubicalCSPType.from_csp_problem()`.  
*   **Funcionalidad Deseada:** Representar dominios finitos de variables CSP como tipos cúbicos (`Fin(n)` o un conjunto explícito de valores) de manera coherente con la teoría de tipos.  
*   **Impacto:** 76 tests fallando, bloqueando la funcionalidad de integración CSP-Cubical.  
*   **Propuesta de Refactorización:**  
    *   **Lógica:** La idea de tener un `FiniteType` con un conjunto explícito de valores (`values`) es válida y útil para CSPs. La `CubicalFiniteType` base (`Fin(n)`) es más abstracta.  
    *   **Reimplementación:**  
        1.  Modificar `FiniteType` para que su constructor sea compatible con `CubicalFiniteType`, posiblemente pasando `len(values)` como `size` al constructor de la clase base.  
        2.  Alternativamente, si la `CubicalFiniteType` base es demasiado restrictiva, reevaluar la jerarquía de herencia o crear un adaptador.  
        3.  **Opción Recomendada:** Ajustar `FiniteType` para que acepte `name` y `values`, y que calcule `size = len(values)` para pasarlo a `super().__init__(size)`. Esto mantiene la semántica de `FiniteType` y satisface la jerarquía.  
    *   **Coherencia:** Mantener la representación de dominios como tipos finitos es crucial para la integración CSP-Cubical.  
*   **Dependencias:** `lattice_weaver/formal/cubical_types.py`, `lattice_weaver/formal/cubical_csp_type.py`  
*   **Estimación de Esfuerzo:** 1-2 horas.  

### 3.3. Categoría: Utilidades Auxiliares

#### 3.3.1. Módulo: `core/csp_engine/tms.py` (Truth Maintenance System - TMS)

*   **Problema Actual:** La implementación actual de `TruthMaintenanceSystem` es un "stub" que no implementa los métodos esperados por sus tests (`record_removal`, `explain_inconsistency`, `suggest_constraint_to_relax`, `get_restorable_values`).  
*   **Funcionalidad Deseada:** Un sistema de mantenimiento de la verdad que rastree las dependencias entre decisiones, explique inconsistencias y sugiera relajaciones de restricciones, integrado con el `CSPSolver`.  
*   **Impacto:** 7 tests fallando, funcionalidad de debugging y análisis avanzado limitada.  
*   **Propuesta de Refactorización:**  
    *   **Lógica:** La lógica de un TMS es compleja y su reimplementación completa desde cero es un esfuerzo significativo. Sin embargo, existen implementaciones en `arc_engine/tms.py` y `arc_engine/tms_enhanced.py`.  
    *   **Reimplementación/Adaptación:**  
        1.  **Prioridad:** Evaluar si `arc_engine/tms.py` o `arc_engine/tms_enhanced.py` ya implementan la API deseada por `tests/unit/test_tms.py`.  
        2.  Si es así, actualizar el import en `tests/unit/test_tms.py` para usar la implementación de `arc_engine`.  
        3.  Si no, o si la implementación de `arc_engine` es incompatible, se deberá implementar la API esperada en `core/csp_engine/tms.py`, adaptando la lógica de las versiones de `arc_engine` o reimplementando las funciones clave (`record_removal`, `explain_inconsistency`, etc.) para que se ajusten a la interfaz requerida.  
    *   **Coherencia:** El TMS debe integrarse con el `CSPSolver` para monitorizar las decisiones y propagaciones.  
*   **Dependencias:** `lattice_weaver/core/csp_engine/solver.py`, `lattice_weaver/arc_engine/tms.py`, `lattice_weaver/arc_engine/tms_enhanced.py`  
*   **Estimación de Esfuerzo:** 1-3 horas (dependiendo de la adaptación necesaria).

#### 3.3.2. Módulo: `path_finder.py` y `symmetry_extractor.py`

*   **Problema Actual:** Estos módulos fallan con `TypeError` en operaciones de caché o equivalencia. Anteriormente estaban bloqueados por el SyntaxError.  
*   **Funcionalidad Deseada:**  
    *   **Path Finder:** Encontrar caminos entre soluciones en el espacio de búsqueda.  
    *   **Symmetry Extractor:** Identificar y explotar simetrías en problemas CSP para reducir el espacio de búsqueda.  
*   **Impacto:** 35 tests fallando, utilidades de análisis de soluciones y optimización de búsqueda no operativas.  
*   **Propuesta de Refactorización:**  
    *   **Lógica:** Las ideas de path finding y extracción de simetrías son valiosas. La lógica subyacente (cálculo de distancia de Hamming, identificación de clases de equivalencia) es probablemente correcta.  
    *   **Reimplementación/Adaptación:**  
        1.  **Diagnóstico:** Ejecutar tests individuales y depurar para identificar la causa exacta del `TypeError`. Es probable que se deba a cambios en la estructura de las soluciones (`CSPSolution`) o en la API de las funciones de caché.  
        2.  **Adaptación:** Ajustar el código para que sea compatible con la estructura de soluciones actual del `CSPSolver` y las APIs de caché disponibles.  
        3.  **Reimplementación:** Si la lógica de caché o equivalencia es fundamentalmente incompatible, reimplementar las funciones afectadas desde cero, asegurando que se integren con las estructuras de datos actuales y los principios de no redundancia (memoización, PEC).  
    *   **Coherencia:** Estos módulos deben interactuar con las soluciones proporcionadas por el `CSPSolver`.  
*   **Dependencias:** `lattice_weaver/core/csp_engine/solver.py` (para `CSPSolution`), posibles módulos de caché.  
*   **Estimación de Esfuerzo:** 2-4 horas.  

#### 3.3.3. Módulo: `visualization.py`

*   **Problema Actual:** Fallos en la generación de reportes.  
*   **Funcionalidad Deseada:** Generar reportes visuales o textuales de los resultados del `CSPSolver` o de análisis de problemas.  
*   **Impacto:** 2 tests fallando, capacidad de visualización limitada.  
*   **Propuesta de Refactorización:**  
    *   **Lógica:** La lógica de lo que se debe reportar (estadísticas, soluciones, estructuras) es probablemente correcta.  
    *   **Reimplementación/Adaptación:**  
        1.  **Diagnóstico:** Depurar los tests para identificar la causa exacta del fallo. Podría ser un cambio en los paths de salida, permisos, o en la estructura de los datos que se intentan visualizar.  
        2.  **Adaptación:** Ajustar el código para que sea compatible con el entorno de ejecución y las estructuras de datos actuales.  
        3.  **Reimplementación:** Si la biblioteca de visualización o la forma de generar reportes ha cambiado drásticamente, reimplementar la generación de reportes utilizando bibliotecas estándar o las herramientas de visualización internas del proyecto.  
    *   **Coherencia:** El módulo debe recibir datos del `CSPSolver` o de otros módulos de análisis.  
*   **Dependencias:** `matplotlib`, `networkx`, `pandas` (posiblemente).  
*   **Estimación de Esfuerzo:** 1-2 horas.

---

## 4. Plan de Implementación Incremental

La refactorización de estos módulos se integrará en el plan general de Fase 7 como una fase adicional de "limpieza" o "estabilización", para no interrumpir el flujo de optimizaciones. Se propone el siguiente orden de prioridad, basándose en el impacto y la facilidad de corrección:

### Fase 7.0.1: Corrección de Generadores de Problemas (API de CSP)

*   **Objetivo:** Restaurar la funcionalidad de los generadores de problemas estándar.  
*   **Tareas:**  
    1.  Actualizar `lattice_weaver/problems/generators/graph_coloring.py` para usar `CSP.add_variable()`.  
    2.  Actualizar `lattice_weaver/problems/generators/nqueens.py` para usar `CSP.add_variable()`.  
    3.  Actualizar `lattice_weaver/problems/generators/sudoku.py` para usar `CSP.add_variable()`.  
    4.  Ejecutar tests de `tests/integration/problems/` y `tests/integration/regression/` para validar.  
*   **Estimación:** 2-3 horas.  

### Fase 7.0.2: Corrección de Jerarquía de `FiniteType`

*   **Objetivo:** Corregir el `TypeError` en la inicialización de `FiniteType` y habilitar los tests de módulos formales.  
*   **Tareas:**  
    1.  Modificar `FiniteType` en `cubical_csp_type.py` para que su constructor sea compatible con `CubicalFiniteType`.  
    2.  Ejecutar tests de `tests/unit/formal/` y `tests/integration/formal/` para validar.  
*   **Estimación:** 1-2 horas.  

### Fase 7.0.3: Adaptación del TMS

*   **Objetivo:** Restaurar la funcionalidad del Truth Maintenance System.  
*   **Tareas:**  
    1.  Evaluar `arc_engine/tms.py` y `arc_engine/tms_enhanced.py`.  
    2.  Si son compatibles, actualizar el import en `tests/unit/test_tms.py`.  
    3.  Si no, implementar los métodos faltantes (`record_removal`, etc.) en `core/csp_engine/tms.py` basándose en la lógica de las versiones de `arc_engine`.  
    4.  Ejecutar tests de `tests/unit/test_tms.py` para validar.  
*   **Estimación:** 1-3 horas.  

### Fase 7.0.4: Corrección de `Path Finder` y `Symmetry Extractor`

*   **Objetivo:** Restaurar la funcionalidad de las utilidades de análisis de soluciones.  
*   **Tareas:**  
    1.  Depurar `TypeError` en `path_finder.py` y `symmetry_extractor.py`.  
    2.  Adaptar el código a la estructura de `CSPSolution` y APIs de caché actuales.  
    3.  Ejecutar tests de `tests/unit/test_path_finder.py` y `tests/unit/test_symmetry_extractor.py` para validar.  
*   **Estimación:** 2-4 horas.  

### Fase 7.0.5: Corrección de `Visualization`

*   **Objetivo:** Restaurar la generación de reportes.  
*   **Tareas:**  
    1.  Depurar fallos en `visualization.py`.  
    2.  Adaptar el código a los paths y estructuras de datos actuales.  
    3.  Ejecutar tests de `tests/unit/test_visualization.py` para validar.  
*   **Estimación:** 1-2 horas.

**Tiempo total estimado para esta fase de refactorización:** 7-14 horas.

---

## 5. Estrategia de Testing

Para cada sub-fase de refactorización, se seguirá una estrategia de testing rigurosa:

1.  **Tests Unitarios:** Se ejecutarán los tests unitarios específicos del módulo refactorizado para asegurar que la nueva implementación cumple con la funcionalidad esperada.  
2.  **Tests de Integración:** Se ejecutarán los tests de integración relevantes que dependen del módulo refactorizado para verificar que no se han introducido regresiones en la interacción con otros componentes.  
3.  **Tests de Regresión:** Tras completar todas las sub-fases, se ejecutará la suite completa de tests (excluyendo benchmarks si son muy largos) para asegurar la estabilidad global del proyecto.  
4.  **Validación Manual (Opcional):** Para funcionalidades clave, se podrá realizar una validación manual simple para confirmar el comportamiento esperado.

**Criterio de Aceptación:** Una sub-fase se considerará completada cuando todos los tests relevantes pasen al 100% y no se detecten regresiones en el core del proyecto.

---

## 6. Conclusiones y Próximos Pasos

Este diseño en profundidad proporciona una hoja de ruta clara para abordar los módulos inestables de LatticeWeaver. Al seguir un enfoque incremental y priorizar la estabilidad del core, se puede restaurar la funcionalidad completa del proyecto sin introducir nuevos riesgos.

**Próximo Paso:** Una vez aprobado este diseño, se procederá con la implementación de la **Fase 7.0.1: Corrección de Generadores de Problemas (API de CSP)**, seguida por las demás sub-fases en el orden propuesto. Cada sub-fase culminará con la validación de tests y un commit documentado.

Este plan permitirá que, una vez finalizadas todas las fases de corrección, tengamos un código completamente funcional y estable, listo para las optimizaciones de Fase 7 y futuras extensiones.  

---

## Referencias

[1] `ANALISIS_ORIGEN_FALLOS.md` (Documento interno del proyecto LatticeWeaver)

[2] `ANALISIS_ESTABILIDAD_GLOBAL.md` (Documento interno del proyecto LatticeWeaver)

