# Diseño del Adaptador ConstraintHierarchy a CSP

## 1. Introducción
Este documento describe el diseño de un adaptador bidireccional entre la `ConstraintHierarchy` de Fibration Flow y la representación de Problemas de Satisfacción de Restricciones (CSP) utilizada en el módulo `lattice_weaver/core/csp_problem.py`. El objetivo es permitir que las `ConstraintHierarchy` puedan ser traducidas a problemas CSP para ser resueltos por solvers CSP tradicionales, y que las soluciones CSP puedan ser verificadas o descompiladas en el contexto de Fibration Flow.

## 2. Objetivos
*   Convertir un objeto `ConstraintHierarchy` en un objeto `CSP`.
*   Manejar la conversión de variables y dominios.
*   Mapear las restricciones de `ConstraintHierarchy` a restricciones de CSP, considerando niveles y durezas.
*   Proporcionar un mecanismo para reconstruir una solución de Fibration Flow a partir de una solución CSP.

## 3. Componentes Clave

### 3.1. `ConstraintHierarchyToCSPAdapter` Clase
Esta clase será responsable de la lógica de conversión.

#### Métodos Propuestos:

*   `__init__(self)`: Constructor.
*   `convert_hierarchy_to_csp(self, hierarchy: ConstraintHierarchy, variables_domains: Dict[str, List[Any]]) -> Tuple[CSP, Dict[str, Any]]`:
    *   **Entrada**: Un objeto `ConstraintHierarchy` y los dominios de las variables en formato de lista.
    *   **Salida**: Una tupla que contiene:
        *   Un objeto `CSP`.
        *   Metadatos de mapeo para la descompilación (`Dict[str, Any]`).
    *   **Lógica**: Iterará sobre las restricciones de la `ConstraintHierarchy`, creando las estructuras equivalentes en `CSP`.
        *   **Variables y Dominios**: Las variables de Fibration Flow se mapearán directamente a las variables de CSP. Los dominios de `List[Any]` se convertirán a `frozenset`.
        *   **Restricciones**: Cada `Constraint` de `ConstraintHierarchy` se convertirá en una `CSP.Constraint`.
            *   **Dureza**: Solo las restricciones `HARD` de Fibration Flow se convertirán a restricciones CSP. Las restricciones `SOFT` se ignorarán o se manejarán como un problema de optimización separado si el CSP lo soporta.
            *   **Predicado**: El `predicate` de la restricción de Fibration Flow se envolverá en una nueva función booleana que acepte los valores de las variables en el orden del `scope` y devuelva `True` o `False`.
*   `convert_csp_solution_to_hierarchy_solution(self, csp_solution: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]`:
    *   **Entrada**: Una solución generada por un solver CSP (`Dict[str, Any]`) y los metadatos de mapeo.
    *   **Salida**: Una solución en el formato `Dict[str, Any]` compatible con Fibration Flow.
    *   **Lógica**: Utilizará los metadatos para reconstruir la solución original, que en este caso será un mapeo directo ya que no hay compilación/descompilación compleja de variables.

## 4. Mapeo de Tipos y Estructuras

| Característica Fibration Flow | Característica CSP        | Notas                                                                 |
| :---------------------------- | :------------------------ | :-------------------------------------------------------------------- |
| `ConstraintHierarchy`         | `CSP`                     | La jerarquía completa se convierte en un único CSP.                   |
| `Dict[str, List[Any]]` (keys) | `CSP.variables` (Set[str]) | Las claves del diccionario de dominios de Fibration Flow.              |
| `Dict[str, List[Any]]`        | `CSP.domains` (Dict[str, FrozenSet[Any]]) | `List` se convierte a `FrozenSet` para dominios.                      |
| `ConstraintHierarchy.Constraint` | `CSP.Constraint`          | Solo restricciones `HARD` se convierten.                               |
| `Constraint.variables` (Tuple[str, ...]) | `CSP.Constraint.scope` (FrozenSet[str]) | `Tuple` se convierte a `FrozenSet`.                                    |
| `Constraint.predicate` (Callable[[Dict[str, Any]], Tuple[bool, float]]) | `CSP.Constraint.relation` (Callable[..., bool]) | Se envuelve la función `predicate` para que cumpla con la firma de `relation`. |
| `Constraint.hardness` (`Hardness.HARD`) | N/A                       | Solo restricciones `HARD` se consideran.                               |

## 5. Consideraciones Adicionales
*   **Restricciones Soft**: Las restricciones `SOFT` de Fibration Flow no tienen un equivalente directo en el modelo CSP básico. Se ignorarán en esta versión del adaptador. Si se requiere, se podría considerar una extensión para problemas de optimización con restricciones soft.
*   **Niveles de Restricción**: Los niveles de restricción de Fibration Flow (`GLOBAL`, `LOCAL`, `PATTERN`) se aplanarán en un único conjunto de restricciones para el CSP.
*   **Metadatos**: Los metadatos de las restricciones de Fibration Flow se pueden transferir a los metadatos de las restricciones CSP.

## 6. Pruebas Unitarias
Se crearán pruebas unitarias para:
*   Verificar la correcta conversión de una `ConstraintHierarchy` simple a CSP.
*   Asegurar que las restricciones convertidas se evalúan correctamente en el contexto CSP.
*   Probar la descompilación de una solución CSP a una solución de Fibration Flow.
*   Probar casos límite (jerarquía vacía, solo restricciones soft, etc.).

## 7. Implementación
El adaptador se implementará en un nuevo archivo `lattice_weaver/fibration/constraint_hierarchy_to_csp_adapter.py`.
