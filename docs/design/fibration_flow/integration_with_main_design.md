# Diseño de Integración de Fibration Flow como Capa de Abstracción Superior en `main`

## 1. Introducción
Este documento detalla el diseño para integrar Fibration Flow como una capa de abstracción superior dentro del proyecto `main` de Lattice Weaver. El objetivo es permitir que los componentes existentes de `main` puedan aprovechar las capacidades de Fibration Flow para la resolución de problemas complejos, la optimización y el razonamiento multiescala, al mismo tiempo que Fibration Flow pueda utilizar las estructuras de datos y funcionalidades de `main`.

## 2. Objetivos
*   Establecer puntos de integración claros entre Fibration Flow y los módulos clave de `main`.
*   Permitir que Fibration Flow actúe como un motor de resolución o un componente de optimización para `main`.
*   Asegurar la compatibilidad y el flujo de datos bidireccional entre las estructuras de datos de Fibration Flow y `main`.
*   Mantener la modularidad y la extensibilidad de ambos sistemas.
*   Proporcionar ejemplos de uso y documentación para la integración.

## 3. Puntos de Integración Clave

### 3.1. Integración con CSP (Constraint Satisfaction Problems)
Los adaptadores `CSPToConstraintHierarchyAdapter` y `ConstraintHierarchyToCSPAdapter` ya desarrollados serán fundamentales aquí. La idea es permitir que un `CSP` definido en `lattice_weaver/core/csp_problem.py` pueda ser convertido a una `ConstraintHierarchy`, resuelto por Fibration Flow, y su solución reconvertida a un formato `CSP`.

**Estrategia:**
*   **CSP a Fibration Flow**: Utilizar `CSPToConstraintHierarchyAdapter` para transformar un `CSP` en una `ConstraintHierarchy` y dominios compatibles con Fibration Flow. Luego, Fibration Flow puede aplicar sus algoritmos de búsqueda y optimización.
*   **Fibration Flow a CSP**: Utilizar `ConstraintHierarchyToCSPAdapter` para convertir una `ConstraintHierarchy` (o una parte de ella) en un `CSP` que pueda ser resuelto por solvers CSP tradicionales (ej. `simple_backtracking_solver`). Esto es útil para la validación o para aprovechar solvers existentes.
*   **Punto de Abstracción**: Se podría introducir una función o clase de utilidad en `lattice_weaver/core/csp_solver_fibration_integration.py` que orqueste esta conversión, resolución y descompilación, presentando una interfaz simple para los usuarios de `main`.

### 3.2. Integración con el Compilador Multiescala
El `SimpleMultiscaleCompiler` de Fibration Flow ya implementa la interfaz `MultiscaleCompilerAPI`. La integración aquí implica exponer esta funcionalidad a `main` para que los problemas puedan ser 
