# Integración CSP ↔ Tipos Cúbicos

**Versión:** 1.1
**Fecha:** 15 de Octubre, 2025
**Estado:** ✅ IMPLEMENTADO

---

## Resumen Ejecutivo

Este documento especifica la integración profunda entre el **motor de resolución CSP** (`arc_engine`) y el **sistema de tipos cúbicos** (`formal/cubical_*`), permitiendo traducir problemas CSP directamente a tipos cúbicos y verificar soluciones mediante el type checker cúbico.

### Motivación

Anteriormente, la integración CSP-HoTT utilizaba tipos Sigma simples, sin aprovechar el sistema completo de tipos cúbicos. Esta nueva integración resuelve esa limitación, creando un puente directo entre CSP y tipos cúbicos, lo que permite:

- **Verificación formal** de soluciones CSP mediante el sistema de tipos cúbicos.
- **Representación de restricciones aritméticas** como `SumConstraint`.
- **Abstracción de la lógica de negocio** en tipos de datos como `CubicalArithmetic` y `CubicalComparison`.

---

## Arquitectura de la Integración

### Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    CSP Problem                              │
│  Variables: {x1, x2, ..., xn}                              │
│  Domains: {D1, D2, ..., Dn}                                │
│  Constraints: {C1, C2, ..., Ck}                            │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              CSPToCubicalBridge                             │
│  - translate_problem_to_cubical_type()                     │
│  - _translate_sum_constraint()                             │
│  - solution_to_cubical_term()                              │
│  - verify_solution_cubical()                               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Cubical Type System                         │
│  Context: Γ                                                │
│  Type: T = Σ(x1:D1)...Σ(xn:Dn). PathType(constraints)     │
│  Term: t = (v1, ..., vn, paths)                            │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Cubical Type Checker                           │
│  - type_check(Γ, t, T) → Bool                             │
│  - normalize(t) → t'                                       │
│  - check_path_coherence() → Bool                           │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                 Verification Result                         │
│  - Solution is valid: Yes/No                               │
│  - Proof term: t                                           │
│  - Equivalence class: [solutions]                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Diseño Detallado

### 1. Traducción de Problemas CSP a Tipos Cúbicos

#### Representación de Variables

Cada variable CSP `xi` con dominio `Di` se traduce a un tipo cúbico:

```
xi : Di_Type
```

Donde `Di_Type` es un tipo discreto (0-dimensional) que representa el dominio finito.

**Ejemplo:**
```python
# CSP
x = Variable("x", domain={1, 2, 3})

# Tipo Cúbico
x : Fin(3)  # Tipo finito con 3 elementos
```

---

#### Representación de Restricciones como Caminos

Con la introducción de `CubicalArithmetic` y `CubicalComparison`, las restricciones pueden ser expresadas de manera más rica y directamente traducidas a la lógica cúbica. `CubicalArithmetic` permite operaciones aritméticas sobre términos cúbicos, mientras que `CubicalComparison` maneja las relaciones de igualdad y desigualdad.

##### Traducción de SumConstraint

La `SumConstraint` en CSP, que asegura que la suma de un conjunto de variables sea igual a un `target_sum`, se traduce a un tipo de camino que utiliza `CubicalArithmetic` y `CubicalComparison` para expresar la relación. Por ejemplo, una restricción `x + y == 10` se representaría como un camino que afirma la igualdad aritmética de la suma de `x` e `y` con el valor `10`.

```python
# CSP: Restricción SumConstraint(scope={\'x\', \'y\'}, target_sum=10)
constraint = SumConstraint(scope=frozenset({"x", "y"}), target_sum=10)

# Tipo Cúbico (conceptual)
sum_path : (x : Fin(D_x)) → (y : Fin(D_y)) → PathType(CubicalArithmetic.add(x, y) == CubicalArithmetic.constant(10))
```

Esta traducción permite que las propiedades de la suma sean verificadas dentro del sistema de tipos cúbicos, aprovechando sus capacidades para razonar sobre igualdades y caminos.

Una restricción `C(xi, xj)` se traduce a un **tipo de caminos** (PathType) que conecta valores compatibles:

```
C_path : (a : Di) → (b : Dj) → PathType(C(a,b))
```

**Interpretación:**
- Si `C(a, b)` es verdadero, existe un camino (prueba) que conecta `a` y `b`
- Si `C(a, b)` es falso, NO existe tal camino

**Ejemplo:**
```python
# CSP: Restricción x ≠ y
constraint = lambda x, y: x != y

# Tipo Cúbico
neq_path : (a : Fin(3)) → (b : Fin(3)) → (a ≠ b) → PathType(a, b)
```

---

#### Tipo Completo del Problema

Un problema CSP completo se traduce al tipo:

```
CSP_Type = Σ(x1 : D1). Σ(x2 : D2). ... Σ(xn : Dn). 
           Path(C1) × Path(C2) × ... × Path(Ck)
```

**Interpretación:**
- **Σ (Sigma):** "Existe un valor para la variable"
- **Path:** "Existe un camino (prueba) que satisface la restricción"
- **×:** "Y todas las restricciones se satisfacen simultáneamente"

---

### 2. Traducción de Soluciones a Términos Cúbicos

Una solución CSP se traduce a un **término cúbico** que habita el tipo del problema:

```
solution_term : CSP_Type
solution_term = (v1, v2, ..., vn, p1, p2, ..., pk)
```

Donde:
- `vi` son los valores asignados a las variables
- `pi` son los caminos (pruebas) de las restricciones

**Ejemplo:**
```python
# CSP Solution
solution = {"x": 1, "y": 2, "z": 3}

# Término Cúbico
term = (val_x_1, val_y_2, val_z_3, 
        path_xy_1_2, path_yz_2_3, path_xz_1_3)
```

---

### 3. Verificación mediante Type Checking Cúbico

La verificación de una solución se reduce a **type checking**:

```
Γ ⊢ solution_term : CSP_Type
```

Si el término type-checks correctamente, la solución es válida.

**Ventajas:**
1. **Verificación formal** mediante teoría de tipos
2. **Garantías de correctitud** por construcción
3. **Equivalencia de soluciones** mediante caminos
4. **Optimizaciones** basadas en estructura cúbica

---

### 4. Equivalencia de Soluciones

Dos soluciones `s1` y `s2` son **equivalentes** si existe un camino entre sus términos:

```
equiv : PathType(s1, s2)
```

Esto permite:
- Identificar clases de equivalencia de soluciones
- Contar soluciones únicas (módulo equivalencia)
- Optimizar búsqueda evitando soluciones equivalentes

---

## Implementación

### Clase Principal: `CSPToCubicalBridge`

```python
# lattice_weaver/formal/csp_cubical_bridge_refactored.py

from lattice_weaver.core import CSP, SumConstraint
from lattice_weaver.formal.cubical_types import CubicalType, CubicalArithmetic, CubicalComparison

class CSPToCubicalBridge:
    """Puente entre CSP y sistema de tipos cúbicos."""

    def _translate_sum_constraint(self, constraint: SumConstraint) -> CubicalComparison:
        """Traduce una SumConstraint a una CubicalComparison."""
        # Implementación de la traducción
        # ...
```

### Nuevos Tipos Cúbicos

Se han añadido los siguientes tipos a `cubical_types.py` para soportar la nueva funcionalidad:

- **`CubicalArithmetic`**: Representa operaciones aritméticas sobre términos cúbicos.
- **`CubicalComparison`**: Representa comparaciones (igualdad, desigualdad) entre términos cúbicos.

Estos tipos proporcionan la base para traducir restricciones aritméticas complejas a la lógica cúbica.

---

## Conclusión

La integración de `SumConstraint` y los nuevos tipos cúbicos (`CubicalArithmetic`, `CubicalComparison`) fortalece significativamente el puente entre el motor CSP y el sistema de tipos cúbicos. Esto no solo permite una verificación formal más robusta, sino que también abre la puerta a futuras optimizaciones y a una representación más expresiva de los problemas CSP en el dominio de la teoría de tipos homotópica.

