# Diseño de Refactorización: `EnergyLandscape`

## 1. Introducción

Este documento detalla el plan para refactorizar la clase `EnergyLandscape` ubicada en `lattice_weaver/fibration/energy_landscape_optimized.py`. El objetivo principal es mejorar la modularidad, la legibilidad y la adherencia a los principios de diseño de LatticeWeaver, así como actualizar su API para una mayor claridad y facilidad de uso. Se pondrá especial énfasis en la robustez del código y la implementación de pruebas unitarias exhaustivas.

## 2. Análisis de la Implementación Actual

La implementación actual de `EnergyLandscape` es fundamental para el cálculo de la energía de una solución en Fibration Flow. Se han identificado las siguientes áreas de mejora:

*   **Dependencia de `ConstraintHierarchy`**: La `EnergyLandscape` interactúa directamente con la `ConstraintHierarchy` para obtener y evaluar restricciones. Es crucial que esta interacción sea clara y que la `EnergyLandscape` no asuma demasiada lógica interna de la jerarquía.
*   **Flexibilidad en la Función de Energía**: La forma en que se calcula la energía podría ser más flexible, permitiendo diferentes estrategias de agregación de violaciones de restricciones soft.
*   **Claridad en la API**: Asegurar que los métodos para añadir y evaluar restricciones soft sean intuitivos y consistentes.
*   **Serialización/Deserialización**: Al igual que `ConstraintHierarchy`, la serialización de `EnergyLandscape` (especialmente si contiene referencias a predicados o funciones de agregación) necesita ser robusta y segura.

## 3. Principios de Diseño Aplicados

La refactorización se guiará por los siguientes meta-principios de diseño de LatticeWeaver (ver `README.md`):

*   **Modularidad**: Separar claramente las responsabilidades. La `EnergyLandscape` debe centrarse en el cálculo de la energía, delegando la gestión de restricciones a `ConstraintHierarchy`.
*   **No Redundancia (DRY)**: Evitar la duplicación de lógica en el cálculo de energía y la evaluación de restricciones.
*   **Claridad y Legibilidad**: Simplificar la interfaz y el código interno para facilitar la comprensión y el mantenimiento.
*   **Extensibilidad**: Permitir la fácil incorporación de nuevas formas de calcular la energía o de agregar violaciones de restricciones.

## 4. Propuesta de Refactorización

### 4.1. Interacción con `ConstraintHierarchy`

La `EnergyLandscape` recibirá una instancia de `ConstraintHierarchy` en su constructor o a través de un método `set_constraint_hierarchy`. Esto permitirá que la `EnergyLandscape` consulte las restricciones de la jerarquía sin tener que gestionarlas directamente.

### 4.2. Función de Energía Flexible

Se introducirá un mecanismo para definir cómo se agregan las violaciones de las restricciones soft. Esto podría ser a través de una función de agregación configurable (e.g., suma, promedio, máximo) o permitiendo que la `EnergyLandscape` se configure con diferentes estrategias de cálculo de energía.

### 4.3. Serialización/Deserialización Robusta

Se implementará un mecanismo más robusto para `to_json` y `from_json`, similar al propuesto para `ConstraintHierarchy`, enfocándose en la serialización de referencias a funciones de agregación o estrategias de cálculo de energía.

### 4.4. Pruebas Unitarias

Se crearán o actualizarán pruebas unitarias en `tests/fibration/test_energy_landscape.py` para cubrir:

*   Cálculo de energía con diferentes combinaciones de restricciones hard y soft.
*   Manejo de la interacción con `ConstraintHierarchy`.
*   Serialización y deserialización.
*   Casos de borde (sin restricciones, todas hard, todas soft).

## 5. Plan de Implementación (Pasos)

1.  **Leer `energy_landscape_optimized.py`**: Entender la implementación actual.
2.  **Definir `EnergyLandscapeAPI`**: Si no existe, crear una interfaz abstracta para `EnergyLandscape`.
3.  **Refactorizar `EnergyLandscape`**: Implementar los cambios según la propuesta.
4.  **Desarrollar/Actualizar Pruebas Unitarias**: Escribir pruebas exhaustivas para la funcionalidad refactorizada.
5.  **Actualizar Documentación en Código**: Asegurar que todos los docstrings y comentarios reflejen los cambios.

## 6. Entregables

*   Archivo `lattice_weaver/fibration/energy_landscape_optimized.py` refactorizado.
*   Archivo `lattice_weaver/fibration/energy_landscape_api.py` (si es necesario).
*   Archivo `tests/fibration/test_energy_landscape.py` con pruebas unitarias actualizadas.
*   Este documento de diseño actualizado.

## 7. Impacto y Beneficios

Esta refactorización resultará en un código más limpio, modular y fácil de mantener para el cálculo de la energía. La API simplificada reducirá la probabilidad de errores y mejorará la experiencia del desarrollador. Una serialización más robusta permitirá la persistencia del paisaje de energía, un paso crucial para la comunicación entre agentes y la gestión del estado del problema.
