# Diseño del Adaptador CSP a ConstraintHierarchy

## 1. Introducción
Este documento describe el diseño de un adaptador bidireccional entre la representación de Problemas de Satisfacción de Restricciones (CSP) utilizada en el módulo `lattice_weaver/core/csp_problem.py` y la `ConstraintHierarchy` de Fibration Flow. El objetivo es permitir que los problemas CSP existentes puedan ser procesados por el motor de Fibration Flow y que las soluciones de Fibration Flow puedan ser verificadas o descompiladas en el contexto de CSP.

## 2. Objetivos
*   Convertir un objeto `CSP` en un objeto `ConstraintHierarchy`.
*   Manejar la conversión de variables y dominios.
*   Mapear las restricciones de CSP a restricciones de `ConstraintHierarchy`, asignando niveles y durezas apropiadas.
*   Proporcionar un mecanismo para reconstruir una solución CSP a partir de una solución de Fibration Flow.

## 3. Componentes Clave

### 3.1. `CSPToConstraintHierarchyAdapter` Clase
Esta clase será responsable de la lógica de conversión.

#### Métodos Propuestos:

*   `__init__(self)`: Constructor.
*   `convert_csp_to_hierarchy(self, csp: CSP) -> Tuple[ConstraintHierarchy, Dict[str, List[Any]], Dict[str, Any]]`:
    *   **Entrada**: Un objeto `CSP`.
    *   **Salida**: Una tupla que contiene:
        *   Un objeto `ConstraintHierarchy`.
        *   Un diccionario de dominios de variables compatible con Fibration Flow (`Dict[str, List[Any]]`).
        *   Metadatos de mapeo para la descompilación (`Dict[str, Any]`).
    *   **Lógica**: Iterará sobre las variables y restricciones del CSP, creando las estructuras equivalentes en `ConstraintHierarchy`.
        *   **Variables y Dominios**: Las variables de CSP se mapearán directamente a las variables de Fibration Flow. Los dominios de `frozenset` se convertirán a `List[Any]`.
        *   **Restricciones**: Cada restricción de CSP se convertirá en una `Constraint` de `ConstraintHierarchy`.
            *   **Nivel**: Por defecto, todas las restricciones de CSP se considerarán de nivel `LOCAL` en Fibration Flow, a menos que se especifique una lógica más compleja (ej. basada en el arity de la restricción o metadatos).
            *   **Dureza**: Todas las restricciones de CSP son inherentemente `HARD`. Por lo tanto, se asignará `Hardness.HARD`.
            *   **Predicado**: El `relation` de la restricción CSP se envolverá en un nuevo predicado que acepte un diccionario de asignaciones y devuelva `(bool, float)` como requiere `ConstraintHierarchy`.
*   `convert_hierarchy_solution_to_csp_solution(self, fibration_solution: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]`:
    *   **Entrada**: Una solución generada por Fibration Flow (`Dict[str, Any]`) y los metadatos de mapeo.
    *   **Salida**: Una solución en el formato `Dict[str, Any]` compatible con CSP.
    *   **Lógica**: Utilizará los metadatos para reconstruir la solución original, que en este caso será un mapeo directo ya que no hay compilación/descompilación compleja de variables.

## 4. Mapeo de Tipos y Estructuras

| Característica CSP        | Característica Fibration Flow | Notas                                                                 |
| :------------------------ | :---------------------------- | :-------------------------------------------------------------------- |
| `CSP.variables` (Set[str]) | `Dict[str, List[Any]]` (keys) | Las claves del diccionario de dominios de Fibration Flow.              |
| `CSP.domains` (Dict[str, FrozenSet[Any]]) | `Dict[str, List[Any]]`        | `FrozenSet` se convierte a `List` para dominios.                      |
| `CSP.constraints` (List[Constraint]) | `ConstraintHierarchy`         | Cada `CSP.Constraint` se convierte en una `ConstraintHierarchy.Constraint`. |
| `CSP.Constraint.scope` (FrozenSet[str]) | `Constraint.variables` (Tuple[str, ...]) | `FrozenSet` se convierte a `Tuple` para inmutabilidad.                |
| `CSP.Constraint.relation` (Callable[..., bool]) | `Constraint.predicate` (Callable[[Dict[str, Any]], Tuple[bool, float]]) | Se envuelve la función `relation` para que cumpla con la firma de `predicate`. |
| N/A                       | `Constraint.level`            | Por defecto `ConstraintLevel.LOCAL`.                                  |
| N/A                       | `Constraint.hardness`         | Por defecto `Hardness.HARD`.                                          |

## 5. Consideraciones Adicionales
*   **Flexibilidad de Niveles**: Aunque por defecto se usará `LOCAL`, el adaptador podría extenderse para inferir niveles de restricción basados en la aridad o el tipo de restricción CSP.
*   **Restricciones Soft en CSP**: El modelo CSP actual no soporta restricciones soft directamente. Si se introducen en el futuro, el adaptador deberá ser actualizado para mapearlas correctamente.
*   **Metadatos**: Los metadatos de las restricciones CSP se pueden transferir a los metadatos de las restricciones de Fibration Flow.

## 6. Pruebas Unitarias
Se crearán pruebas unitarias para:
*   Verificar la correcta conversión de un CSP simple a `ConstraintHierarchy`.
*   Asegurar que las restricciones convertidas se evalúan correctamente.
*   Probar la descompilación de una solución de Fibration Flow a una solución CSP.
*   Probar casos límite (CSP sin restricciones, CSP con una sola variable, etc.).

## 7. Implementación
El adaptador se implementará en un nuevo archivo `lattice_weaver/fibration/csp_adapter.py`.
