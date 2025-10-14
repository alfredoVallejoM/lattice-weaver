# Análisis y Propuestas de Optimización de Estructuras de Datos para el Flujo de Fibración

**Fecha:** 14 de Octubre de 2025  
**Autor:** Manus AI

## 1. Introducción

Este documento presenta un análisis de las estructuras de datos actualmente utilizadas en el core del solver de Flujo de Fibración y propone una serie de mejoras y refactorizaciones. El objetivo es alinear la implementación con los principios de diseño de LatticeWeaver, específicamente la **Gestión de Memoria Eficiente** y la **No Redundancia/Canonicalización**, para mejorar el rendimiento y la escalabilidad del solver.

## 2. Análisis de Estructuras de Datos Actuales

El análisis se centra en el fichero `lattice_weaver/fibration/constraint_hierarchy.py`, que define la representación de las restricciones y su jerarquía.

### 2.1. `Constraint` (dataclass)

La clase `Constraint` es un `dataclass` que representa una única restricción. Esta es una elección de diseño clara y adecuada. Sin embargo, se identifican dos áreas de posible optimización:

*   **Uso de memoria de los predicados:** El atributo `predicate` almacena una función de Python. Para problemas con un gran número de restricciones que comparten la misma lógica (p. ej., `a != b`), se crean y almacenan múltiples objetos de función idénticos, lo que aumenta el consumo de memoria.
*   **Almacenamiento de variables:** El atributo `variables` es una lista de strings. Para restricciones de alta aridad, esta lista puede ocupar una cantidad no trivial de memoria.

### 2.2. `ConstraintHierarchy`

La clase `ConstraintHierarchy` organiza las restricciones en una jerarquía de niveles. La estructura de datos principal es `self.constraints: Dict[ConstraintLevel, List[Constraint]]`, un diccionario que mapea cada nivel a una lista de restricciones.

El principal punto de ineficiencia se encuentra en el método `get_constraints_involving(self, variable: str)`, que realiza una búsqueda lineal a través de todas las restricciones cada vez que se invoca. En algoritmos como AC-3, donde esta operación es frecuente, este método se convierte en un cuello de botella computacional, violando el principio de **No Redundancia**.

## 3. Propuestas de Mejora

Se proponen las siguientes mejoras para optimizar las estructuras de datos del Flujo de Fibración:

### 3.1. Indexación de Restricciones por Variable

**Problema:** Búsqueda lineal ineficiente en `get_constraints_involving`.

**Propuesta:** Añadir un índice `self.constraints_by_variable: Dict[str, List[Constraint]]` a la clase `ConstraintHierarchy`. Este diccionario mapeará cada ID de variable a una lista de las restricciones en las que participa.

**Implementación:**

```python
class ConstraintHierarchy:
    def __init__(self):
        self.constraints: Dict[ConstraintLevel, List[Constraint]] = { ... }
        self.constraints_by_variable: Dict[str, List[Constraint]] = {}

    def add_constraint(self, constraint: Constraint):
        self.constraints[constraint.level].append(constraint)
        for var in constraint.variables:
            if var not in self.constraints_by_variable:
                self.constraints_by_variable[var] = []
            self.constraints_by_variable[var].append(constraint)

    def get_constraints_involving(self, variable: str) -> List[Constraint]:
        return self.constraints_by_variable.get(variable, [])
```

**Beneficios:**

*   **Rendimiento:** La complejidad de `get_constraints_involving` se reduce de O(N) a O(1) en promedio.
*   **No Redundancia:** Se elimina la necesidad de realizar búsquedas repetitivas, alineándose con los principios de diseño.

### 3.2. Canonicalización de Predicados

**Problema:** Almacenamiento redundante de funciones de predicado idénticas.

**Propuesta:** Implementar un sistema de canonicalización para los predicados. Se puede utilizar un diccionario para almacenar una única instancia de cada función de predicado, utilizando su código fuente o un identificador único como clave.

**Implementación (conceptual):**

```python
class PredicateFactory:
    def __init__(self):
        self.predicates: Dict[str, Callable] = {}

    def get_predicate(self, predicate_code: str) -> Callable:
        if predicate_code not in self.predicates:
            # Compilar y almacenar la nueva función
            self.predicates[predicate_code] = compile(predicate_code, 
'<string>', 'eval')
        return self.predicates[predicate_code]
```

**Beneficios:**

*   **Memoria:** Se reduce significativamente el consumo de memoria al almacenar una única instancia de cada predicado.
*   **Canonicalización:** Se asegura que las restricciones lógicamente equivalentes compartan la misma representación en memoria.

### 3.3. Estructuras de Datos Eficientes para Dominios

**Problema:** El uso de listas o conjuntos de Python para representar dominios de variables puede ser ineficiente para dominios grandes o con una estructura particular.

**Propuesta:** Introducir una clase `Domain` que encapsule diferentes representaciones de dominio y seleccione la más eficiente según el tipo de datos.

**Implementación (conceptual):**

```python
class Domain:
    def __init__(self, values):
        if all(isinstance(x, int) for x in values):
            # Usar un bitset para dominios de enteros
            self._representation = BitSet(values)
        elif is_continuous_range(values):
            # Usar un intervalo para rangos continuos
            self._representation = Interval(min(values), max(values))
        else:
            # Usar un set como fallback
            self._representation = set(values)

    # ... métodos para operaciones de dominio (p. ej., __contains__, __iter__)
```

**Beneficios:**

*   **Memoria:** El uso de `bitsets` o `intervals` puede reducir drásticamente el consumo de memoria en comparación con los `sets` de Python.
*   **Rendimiento:** Las operaciones sobre estas estructuras de datos especializadas (p. ej., intersección, unión) son significativamente más rápidas.

## 4. Conclusión

La implementación de estas propuestas de optimización de estructuras de datos mejorará significativamente la eficiencia y escalabilidad del solver de Flujo de Fibración. Se recomienda abordar en primer lugar la **indexación de restricciones por variable**, ya que es la mejora con el mayor impacto potencial en el rendimiento y la más sencilla de implementar. Las otras dos propuestas, aunque también valiosas, requieren un esfuerzo de refactorización mayor y pueden abordarse en una fase posterior.

