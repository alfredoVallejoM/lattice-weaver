# Diseño de Refactorización de Fibration Flow: Coherencia con los Principios de `main`

**Versión:** 1.0  
**Fecha:** 15 de Octubre, 2025  
**Propósito:** Detallar la refactorización de los componentes de Fibration Flow (`EnergyLandscapeOptimized`, `HacificationEngine`, `FibrationSearchSolver`) para asegurar su coherencia con los `LatticeWeaver_Meta_Principios_Diseño.md`.

## 1. Análisis de Coherencia con los Principios de Diseño

### 1.1. `EnergyLandscapeOptimized`

| Principio de Diseño | Coherencia Actual | Acciones de Refactorización |
|---|---|---|
| **Economía Computacional** | **Alta**. El cálculo incremental y el caché de energías son ejemplos claros de este principio. | Mantener y potenciar. Explorar la posibilidad de un caché persistente para problemas recurrentes. |
| **Localidad** | **Media**. El `_var_to_constraints` es un buen ejemplo de localidad, pero el caché de energías es global. | Investigar si un caché distribuido o por niveles de abstracción sería más eficiente. |
| **Asincronía** | **Baja**. Todas las operaciones son síncronas. | No es una prioridad inmediata para este componente, pero se podría explorar la computación asíncrona de gradientes. |
| **Convergencia Emergente** | **N/A**. Este componente es un evaluador, no un sistema dinámico. | N/A |
| **Caché Agresivo** | **Alta**. El caché de energías es un pilar del diseño. | Mantener y optimizar. |
| **Evaluación Perezosa** | **Baja**. La energía se calcula bajo demanda, pero no hay una evaluación perezosa de componentes. | Explorar si los componentes de energía (local, pattern, global) pueden ser evaluados de forma perezosa. |
| **Inmutabilidad** | **Media**. Las asignaciones son diccionarios mutables. | Utilizar `frozenset` para las claves de caché y `dataclasses` inmutables para los componentes de energía. |

### 1.2. `HacificationEngine`

| Principio de Diseño | Coherencia Actual | Acciones de Refactorización |
|---|---|---|
| **Economía Computacional** | **Media**. El AC-3 es eficiente, pero se ejecuta en cada llamada a `filter_coherent_extensions`. | Implementar un sistema de propagación de consistencia más persistente que no reinicie el trabajo en cada paso. |
| **Localidad** | **Media**. El AC-3 opera sobre dominios locales, pero la información de poda no se propaga globalmente de forma eficiente. | Investigar algoritmos de consistencia de arco más avanzados (ej. AC-2001/3.1) que mantengan estructuras de datos para una propagación más eficiente. |
| **Asincronía** | **Baja**. El AC-3 es síncrono. | Explorar versiones paralelas o asíncronas de AC-3. |
| **Aprovechamiento de Información** | **Media**. El AC-3 es una forma de aprendizaje de no-goods implícito, pero no se almacenan explícitamente. | Añadir un `NoGoodLearner` explícito que registre las podas de dominio y las reutilice. |

### 1.3. `FibrationSearchSolver`

| Principio de Diseño | Coherencia Actual | Acciones de Refactorización |
|---|---|---|
| **Economía Computacional** | **Media**. El Branch & Bound y las heurísticas (MRV, LCV) son eficientes, pero la selección de variables y valores puede ser costosa. | Optimizar las heurísticas y explorar técnicas de poda más agresivas. |
| **Aprovechamiento de Información** | **Media**. El Branch & Bound es una forma de reutilización de información, pero no hay aprendizaje entre búsquedas. | Integrar el `NoGoodLearner` del `HacificationEngine` y explorar el caché de isomorfismos para subproblemas. |
| **Modularidad** | **Alta**. La separación de `EnergyLandscape`, `HacificationEngine` y `LandscapeModulator` es un buen ejemplo de composición. | Mantener y reforzar esta separación de responsabilidades. |

## 2. Plan de Refactorización

### 2.1. `EnergyLandscapeOptimized`

1.  **Refactorizar `EnergyComponents` a un `dataclass` inmutable (`frozen=True`)**.
2.  **Utilizar `frozenset(assignment.items())` como clave de caché** para asegurar la inmutabilidad.
3.  **Explorar la evaluación perezosa de los componentes de energía** utilizando propiedades (`@property`).

### 2.2. `HacificationEngine`

1.  **Refactorizar el AC-3** para que sea más persistente, manteniendo una cola de arcos a revisar entre llamadas.
2.  **Integrar un `NoGoodLearner`** que registre las podas de dominio y las utilice para una poda más rápida en futuras llamadas.

### 2.3. `FibrationSearchSolver`

1.  **Integrar el `NoGoodLearner`** del `HacificationEngine` en la lógica de poda.
2.  **Optimizar las heurísticas MRV y LCV** para reducir su costo computacional.

## 3. Implementación de Tests

Para cada componente refactorizado, se crearán o actualizarán los tests unitarios y de integración para asegurar:

*   **Correctitud**: El comportamiento funcional no ha cambiado.
*   **Rendimiento**: El rendimiento no ha empeorado y, preferiblemente, ha mejorado.
*   **Coherencia**: El código se adhiere a los principios de diseño.

## 4. Documentación

Se actualizará la documentación de cada componente para reflejar los cambios de la refactorización y las decisiones de diseño tomadas, siguiendo el `PROTOCOLO_AGENTES_LATTICEWEAVER.md`.

