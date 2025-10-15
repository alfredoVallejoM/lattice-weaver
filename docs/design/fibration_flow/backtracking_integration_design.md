# Integración de Fibration Flow con Simple Backtracking Solver

## 1. Objetivo

El objetivo de este documento es describir cómo Fibration Flow puede integrarse o interactuar con el `simple_backtracking_solver.py` existente. La integración busca aprovechar las capacidades de Fibration Flow para mejorar la eficiencia o la capacidad de resolución del backtracking tradicional, o para proporcionar una capa de abstracción que permita la elección entre diferentes estrategias de resolución.

## 2. Contexto Actual

El `simple_backtracking_solver.py` implementa un algoritmo de backtracking básico con heurísticas de selección de variables (MRV y Degree). Resuelve problemas CSP directamente utilizando las restricciones y dominios definidos en el objeto `CSP`.

Fibration Flow, por otro lado, opera sobre una `ConstraintHierarchy` y `fibration_domains`, y utiliza un `FibrationSearchSolver` para encontrar soluciones. Ya existe un adaptador `CSPToConstraintHierarchyAdapter` que convierte un `CSP` en la representación de Fibration Flow.

## 3. Estrategias de Integración

Se proponen dos estrategias principales para la integración:

### 3.1. Estrategia 1: Reemplazo o Delegación Completa

En esta estrategia, el `simple_backtracking_solver` no se modifica directamente, sino que se crea una nueva función o un nuevo solver que, al recibir un `CSP`, delega la resolución completamente a Fibration Flow. Esto ya se ha implementado parcialmente con `solve_csp_with_fibration_flow`.

**Ventajas:**
*   Mantiene el `simple_backtracking_solver` intacto para propósitos de referencia o como fallback.
*   Proporciona una interfaz limpia para usar Fibration Flow como el motor de resolución principal.

**Desventajas:**
*   No hay una interacción directa o híbrida entre ambos solvers.

### 3.2. Estrategia 2: Backtracking Aumentado por Fibration Flow (Híbrido)

Esta estrategia implica modificar el `simple_backtracking_solver` para que utilice Fibration Flow en puntos estratégicos del algoritmo de backtracking. Por ejemplo:

*   **Pre-procesamiento:** Usar Fibration Flow para simplificar el CSP antes de iniciar el backtracking (e.g., eliminar valores de dominio inconsistentes, identificar variables fijas).
*   **Heurísticas de Selección:** Utilizar la información de la jerarquía de restricciones de Fibration Flow para mejorar las heurísticas de selección de variables o valores en el backtracking.
*   **Detección de Inconsistencias Temprana:** Si Fibration Flow puede determinar que un subproblema es insatisfacible más rápidamente que el backtracking, se podría podar el árbol de búsqueda.

**Ventajas:**
*   Combina las fortalezas de ambos enfoques.
*   Potencialmente más eficiente para ciertos tipos de CSPs.

**Desventajas:**
*   Mayor complejidad de implementación.
*   Requiere una cuidadosa gestión del estado entre ambos solvers.

## 4. Decisión de Implementación para la Fase Actual

Para esta fase (Fase 7: Integrar Fibration Flow como una capa de abstracción superior en `main`), la **Estrategia 1 (Reemplazo o Delegación Completa)** es la más adecuada y ya se ha iniciado con `csp_solver_fibration_integration.py`. El objetivo es establecer Fibration Flow como una alternativa de resolución de CSPs dentro del ecosistema `main`.

El `simple_backtracking_solver.py` se mantendrá como está, sirviendo como un solver básico y de referencia. La integración de Fibration Flow como una 
