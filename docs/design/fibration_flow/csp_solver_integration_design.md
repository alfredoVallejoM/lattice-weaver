# Integración de Fibration Flow con CSPSolver

## 1. Objetivo

Este documento describe cómo Fibration Flow puede integrarse con el `CSPSolver` de `lattice_weaver/core/csp_engine/solver.py`. La integración busca proporcionar una alternativa de resolución de CSPs que aproveche las capacidades de Fibration Flow, como la jerarquía de restricciones y la búsqueda basada en energía.

## 2. Contexto Actual

El `CSPSolver` implementa un algoritmo de backtracking con forward checking. Resuelve problemas CSP directamente utilizando las restricciones y dominios definidos en el objeto `CSP`.

Fibration Flow, por otro lado, opera sobre una `ConstraintHierarchy` y `fibration_domains`, y utiliza un `FibrationSearchSolver` para encontrar soluciones. Ya existe un adaptador `CSPToConstraintHierarchyAdapter` que convierte un `CSP` en la representación de Fibration Flow.

## 3. Estrategia de Integración

La estrategia de integración consiste en modificar la clase `CSPSolver` para que pueda utilizar Fibration Flow como un motor de resolución alternativo. Esto se logrará mediante la adición de un nuevo método o un parámetro en el método `solve` que permita seleccionar el motor de resolución.

### 3.1. Modificaciones en `CSPSolver`

Se añadirá un nuevo método `solve_with_fibration_flow` a la clase `CSPSolver`. Este método:

1.  Utilizará el `CSPToConstraintHierarchyAdapter` para convertir el `CSP` del solver en una `ConstraintHierarchy` y `fibration_domains`.
2.  Instanciará un `FibrationSearchSolver`.
3.  Llamará al método `solve` del `FibrationSearchSolver` con la jerarquía y los dominios convertidos.
4.  Convertirá la solución de Fibration Flow de nuevo al formato de solución de CSP.
5.  Devolverá un objeto `CSPSolutionStats` con la solución y las estadísticas (si están disponibles).

### 3.2. Flujo de Datos

El flujo de datos será el siguiente:

1.  El usuario crea una instancia de `CSPSolver` con un `CSP`.
2.  El usuario llama al método `solve_with_fibration_flow`.
3.  Internamente, el `CSPSolver` convierte el `CSP` a la representación de Fibration Flow.
4.  El `FibrationSearchSolver` resuelve el problema.
5.  La solución se convierte de nuevo al formato de CSP.
6.  Se devuelve la solución al usuario.

## 4. Beneficios de la Integración

*   **Flexibilidad:** Permite al usuario elegir entre el solver de backtracking tradicional y el solver de Fibration Flow sin cambiar la interfaz principal del `CSPSolver`.
*   **Abstracción:** Oculta la complejidad de la conversión entre CSP y Fibration Flow al usuario.
*   **Comparación:** Facilita la comparación del rendimiento y la calidad de las soluciones entre ambos solvers.

## 5. Próximos Pasos

1.  Implementar el método `solve_with_fibration_flow` en la clase `CSPSolver`.
2.  Añadir pruebas unitarias para verificar la correcta integración y el flujo de datos.
3.  Actualizar la documentación para reflejar la nueva funcionalidad.

