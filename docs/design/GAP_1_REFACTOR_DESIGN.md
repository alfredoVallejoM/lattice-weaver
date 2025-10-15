# Diseño de Refactorización: Gap 1 - Puente CSP ↔ Tipos Cúbicos

**Proyecto:** LatticeWeaver  
**Versión:** 8.0-alpha  
**Fecha:** 15 de Octubre, 2025  
**Autor:** Manus AI  
**Propósito:** Refactorizar la implementación existente del puente CSP-Cúbico para alinearla con la nueva arquitectura de estrategias y tipos, y completar la funcionalidad pendiente.

---

## 1. Análisis de la Implementación Existente

Tras analizar `csp_cubical_bridge.py` y `cubical_csp_type.py`, se concluye:

-   **Existe una implementación funcional:** El código ya traduce un CSP a un `CubicalCSPType`, que es una forma de `SigmaType`.
-   **Falta de Abstracción:** Las clases `FiniteType` y `PropositionType` están mezcladas con la lógica del `CubicalCSPType` y no son tipos cúbicos genéricos.
-   **Acoplamiento Fuerte:** `CSPToCubicalBridge` está fuertemente acoplado a `CubicalCSPType`.
-   **Gap de Verificación:** El `TODO` en `verify_solution` confirma que la verificación formal con un motor HoTT no está implementada.

## 2. Objetivos de la Refactorización

1.  **Desacoplar y Abstraer:** Separar las definiciones de tipos cúbicos genéricos de la lógica específica del CSP.
2.  **Alinear con el Nuevo Diseño:** Adaptar el puente para que funcione como una `VerificationStrategy`.
3.  **Completar la Funcionalidad:** Implementar la traducción de restricciones complejas y la verificación formal real.

## 3. Plan de Refactorización Incremental

### Fase 1: Abstracción de Tipos (Semana 1)

1.  **Tarea 1.1:** Crear `cubical_types.py` con las clases genéricas (`CubicalType`, `CubicalFiniteType`, `CubicalSigmaType`, etc.) como se diseñó originalmente.
2.  **Tarea 1.2:** Refactorizar `cubical_csp_type.py` para que `FiniteType` y `PropositionType` **hereden** de las nuevas clases base o sean reemplazadas por ellas.
3.  **Tarea 1.3:** Modificar `CSPToCubicalBridge` para que utilice y retorne los nuevos tipos genéricos, en lugar de `CubicalCSPType`.

### Fase 2: Integración como Estrategia (Semana 2)

1.  **Tarea 2.1:** Crear la clase `CubicalVerificationStrategy` en `lattice_weaver/strategies/verification/cubical.py`.
2.  **Tarea 2.2:** Mover la lógica de `CSPToCubicalBridge.verify_solution` a `CubicalVerificationStrategy.verify_solution`.
3.  **Tarea 2.3:** La estrategia usará el `CSPToCubicalBridge` refactorizado para la traducción.

### Fase 3: Completar la Verificación (Semana 3-4)

1.  **Tarea 3.1:** Implementar la lógica para traducir los `CubicalPredicate` a algo que un motor HoTT pueda entender (este es el núcleo del gap).
2.  **Tarea 3.2:** Integrar con el `CubicalEngine` para realizar la verificación formal real.

---

Este plan preserva el código existente mientras lo alinea con la nueva arquitectura, permitiendo un progreso incremental y seguro.
