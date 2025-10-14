# Diseño de Refactorización: `ConstraintHierarchy`

**Fecha:** 15 de Octubre, 2025
**Autor:** Manus AI
**Propósito:** Detallar la planificación y el diseño en profundidad para la refactorización del módulo `ConstraintHierarchy` dentro de Fibration Flow, asegurando su robustez, modularidad, extensibilidad y alineación con los Meta-Principios de Diseño de LatticeWeaver y el `PROTOCOLO_AGENTES_LATTICEWEAVER.md`.

---

## 1. Planificación Detallada de la Tarea

### 1.1. Tarea Principal
Refactorizar el módulo `lattice_weaver/fibration/constraint_hierarchy.py` para mejorar su diseño, robustez y extensibilidad, preparándolo para una integración futura y flexible con la nueva arquitectura de `main`.

### 1.2. Subtareas
*   **1.2.1. Abstracción de `Constraint`**: Mejorar la clase `Constraint` para que sea más genérica y permita diferentes tipos de predicados y formas de evaluación.
*   **1.2.2. Flexibilidad de `ConstraintHierarchy`**: Permitir la adición de nuevos niveles de abstracción o tipos de restricciones sin modificar la estructura central.
*   **1.2.3. Manejo de Predicados**: Estandarizar la forma en que los predicados son definidos y evaluados, resolviendo la ambigüedad actual sobre si esperan `Dict[str, Any]` o `**kwargs`.
*   **1.2.4. Serialización/Deserialización**: Mejorar la serialización de `ConstraintHierarchy` para manejar predicados de forma segura y eficiente.
*   **1.2.5. Implementación de Tests**: Desarrollar tests unitarios exhaustivos para `Constraint` y `ConstraintHierarchy`.
*   **1.2.6. Documentación**: Actualizar docstrings y añadir comentarios explicativos.

### 1.3. Criterios de Éxito
*   El código de `constraint_hierarchy.py` es más modular y fácil de entender.
*   La clase `Constraint` soporta múltiples firmas de predicados o una firma estandarizada clara.
*   La `ConstraintHierarchy` permite la adición de nuevos niveles de restricción de forma dinámica.
*   La serialización/deserialización es robusta y segura para predicados.
*   Cobertura de tests unitarios >90% para `constraint_hierarchy.py`.
*   Docstrings y comentarios actualizados y claros.

## 2. Diseño Acorde a Principios

### 2.1. Meta-Principios de Diseño de LatticeWeaver Aplicados

*   **Modularidad y Generalidad**: El diseño de `Constraint` y `ConstraintHierarchy` debe ser lo suficientemente genérico para soportar diferentes tipos de restricciones y niveles de abstracción, facilitando su reutilización y extensión. Se buscará desacoplar la definición de la restricción de su evaluación.
*   **No Redundancia/Canonicalización**: Evitar la duplicación de lógica en la gestión de restricciones y en la evaluación de predicados. Estandarizar las interfaces.
*   **Verificabilidad**: El diseño debe facilitar la creación de tests unitarios, especialmente para la evaluación de restricciones y la gestión de la jerarquía.
*   **Economía Computacional**: Aunque no es el foco principal de esta refactorización, el diseño debe sentar las bases para futuras optimizaciones en la evaluación de restricciones.

### 2.2. Decisiones de Diseño Clave

*   **Estandarización de Predicados**: Se adoptará una convención estricta para la firma de los predicados. Todos los predicados deberán aceptar un único argumento de tipo `Dict[str, Any]` que contenga la asignación de variables. Esto simplificará la lógica de `Constraint.evaluate` y evitará errores de `TypeError`.
*   **`Constraint` como Data Class Enriquecida**: La clase `Constraint` se mantendrá como un `dataclass` pero se enriquecerá con métodos para la validación de predicados y una representación más informativa.
*   **`ConstraintHierarchy` Dinámica**: La `ConstraintHierarchy` se refactorizará para permitir la definición y adición de nuevos `ConstraintLevel`s de forma dinámica, posiblemente a través de un registro o una configuración, en lugar de una enumeración fija. Esto mejorará la extensibilidad.
*   **Serialización Segura de Predicados**: En lugar de intentar serializar `Callable`s directamente (lo cual es inseguro y complejo), se optará por serializar una referencia a los predicados (por ejemplo, su nombre calificado) y requerir un registro de predicados conocidos para la deserialización. Esto se implementará en una fase posterior si es necesario, por ahora se mantendrá un placeholder seguro.

### 2.3. Hoja de Ruta de Implementación (Subtareas Detalladas)

1.  **Modificar `Constraint.evaluate`**: Asegurar que solo acepta `assignment: Dict[str, Any]` y que los predicados se llaman con este diccionario.
2.  **Actualizar `add_hard_constraint` y `add_soft_constraint`**: Adaptar estos métodos para que construyan el objeto `Constraint` de acuerdo con la nueva convención de predicados.
3.  **Refactorizar `ConstraintHierarchy`**: 
    *   Permitir la adición de nuevos niveles de restricción de forma dinámica (e.g., `add_level(name: str)`).
    *   Asegurar que los métodos `add_local_constraint`, `add_unary_constraint`, `add_pattern_constraint`, `add_global_constraint` utilizan la nueva convención de predicados.
4.  **Crear Tests Unitarios**: 
    *   `test_constraint_evaluation`: Probar la evaluación de `Constraint` con diferentes predicados y asignaciones.
    *   `test_constraint_hierarchy_add_constraint`: Probar la adición de restricciones a diferentes niveles.
    *   `test_constraint_hierarchy_evaluate_solution`: Probar la evaluación de soluciones completas.
    *   `test_dynamic_levels`: Probar la adición dinámica de nuevos niveles a la jerarquía.
5.  **Actualizar Docstrings y Comentarios**: Asegurar que todo el código está bien documentado.

---

## 3. Análisis de Riesgos y Desafíos

*   **Compatibilidad con código existente**: La estandarización de la firma de predicados podría requerir ajustes en otros módulos que definen predicados para Fibration Flow. Esto se mitigará con una comunicación clara de la nueva convención.
*   **Serialización de predicados**: La serialización segura de `Callable`s es un problema conocido en Python. La solución propuesta (referencia por nombre) es un compromiso que requiere un registro global de predicados, lo cual añade complejidad. Se pospone la implementación completa de esto para cuando sea estrictamente necesario.

---

## 4. Checklist de Validación (Fase de Diseño)

*   [X] Planificación detallada de la tarea realizada.
*   [X] Diseño de la solución alineado con los Meta-Principios de Diseño de LatticeWeaver.
*   [X] Documento de diseño creado y justificado.
*   [ ] Se han identificado los trade-offs y se han documentado.

---

Este documento de diseño servirá como guía para la implementación de la refactorización de `ConstraintHierarchy`.
