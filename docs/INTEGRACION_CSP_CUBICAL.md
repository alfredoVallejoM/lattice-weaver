# Integración CSP ↔ Tipos Cúbicos

**Versión:** 1.0  
**Fecha:** 12 de Octubre, 2025  
**Estado:** 🚧 EN DISEÑO - IMPLEMENTACIÓN PENDIENTE

---

## Resumen Ejecutivo

Este documento especifica la integración profunda entre el **motor de resolución CSP** (`arc_engine`) y el **sistema de tipos cúbicos** (`formal/cubical_*`), permitiendo traducir problemas CSP directamente a tipos cúbicos y verificar soluciones mediante el type checker cúbico.

### Motivación

Actualmente, la integración CSP-HoTT utiliza tipos Sigma simples (`csp_integration_extended.py`), pero **NO aprovecha el sistema completo de tipos cúbicos** implementado en `formal/cubical_syntax.py`, `cubical_operations.py` y `cubical_engine.py`. Esto significa que:

- El sistema de tipos cúbicos (42 KB de código) está **aislado** del motor CSP
- No se pueden verificar **equivalencias de soluciones** mediante caminos cúbicos
- No se aprovecha la **estructura cúbica** para optimizaciones
- Hay **duplicación conceptual** entre la integración Sigma y el sistema cúbico

Esta integración profunda resolverá estos problemas, creando un puente directo entre CSP y tipos cúbicos.

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
│  - translate_constraint_to_path()                          │
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
# lattice_weaver/formal/csp_cubical_bridge.py

from lattice_weaver.arc_engine import CSPProblem, CSPSolution
from lattice_weaver.formal.cubical_syntax import (
    Type, Term, Context, SigmaType, PathType
)
from lattice_weaver.formal.cubical_engine import CubicalTypeChecker

class CSPToCubicalBridge:
    """Puente entre CSP y sistema de tipos cúbicos."""
    
    def __init__(self):
        self.type_checker = CubicalTypeChecker()
        self.context = Context()
    
    def translate_problem_to_cubical_type(
        self, 
        problem: CSPProblem
    ) -> Type:
        """Traduce un problema CSP a un tipo cúbico.
        
        Args:
            problem: Problema CSP a traducir
            
        Returns:
            Tipo cúbico que representa el problema
            
        Example:
            >>> problem = CSPProblem(
            ...     variables=['x', 'y'],
            ...     domains={'x': {1,2,3}, 'y': {1,2,3}},
            ...     constraints=[('x', 'y', lambda a,b: a != b)]
            ... )
            >>> bridge = CSPToCubicalBridge()
            >>> cub_type = bridge.translate_problem_to_cubical_type(problem)
            >>> print(cub_type)
            Σ(x:Fin(3)).Σ(y:Fin(3)).Path(x≠y)
        """
        # 1. Traducir dominios a tipos finitos
        domain_types = {}
        for var, domain in problem.domains.items():
            domain_types[var] = self._domain_to_finite_type(domain)
            self.context.add_variable(var, domain_types[var])
        
        # 2. Traducir restricciones a tipos de caminos
        constraint_types = []
        for (var1, var2, constraint) in problem.constraints:
            path_type = self._constraint_to_path_type(
                var1, var2, constraint
            )
            constraint_types.append(path_type)
        
        # 3. Construir tipo Sigma anidado
        result_type = self._build_sigma_type(
            domain_types, 
            constraint_types
        )
        
        return result_type
    
    def _domain_to_finite_type(self, domain: set) -> Type:
        """Convierte un dominio CSP a tipo finito cúbico."""
        n = len(domain)
        return Type.Finite(n, elements=list(domain))
    
    def _constraint_to_path_type(
        self, 
        var1: str, 
        var2: str, 
        constraint: callable
    ) -> PathType:
        """Convierte una restricción CSP a tipo de caminos."""
        # Crear proposición que representa la restricción
        prop = self._constraint_to_proposition(var1, var2, constraint)
        
        # Crear tipo de caminos condicionado a la proposición
        path_type = PathType(
            source=var1,
            target=var2,
            condition=prop
        )
        
        return path_type
    
    def _constraint_to_proposition(
        self, 
        var1: str, 
        var2: str, 
        constraint: callable
    ) -> Term:
        """Convierte una restricción a proposición lógica."""
        # Representar la restricción como función booleana
        # que retorna un tipo (Type para True, ⊥ para False)
        def prop_function(a, b):
            if constraint(a, b):
                return Type.Unit()  # Tipo habitado (True)
            else:
                return Type.Empty()  # Tipo vacío (False)
        
        return Term.Lambda(var1, Term.Lambda(var2, prop_function))
    
    def _build_sigma_type(
        self, 
        domain_types: dict, 
        constraint_types: list
    ) -> Type:
        """Construye tipo Sigma anidado."""
        # Comenzar con el tipo de restricciones
        constraints_type = Type.Product(*constraint_types)
        
        # Anidar tipos Sigma para cada variable (de derecha a izquierda)
        result = constraints_type
        for var in reversed(list(domain_types.keys())):
            result = SigmaType(var, domain_types[var], result)
        
        return result
    
    def solution_to_cubical_term(
        self, 
        solution: CSPSolution, 
        problem: CSPProblem
    ) -> Term:
        """Convierte una solución CSP a término cúbico.
        
        Args:
            solution: Solución CSP
            problem: Problema CSP original
            
        Returns:
            Término cúbico que representa la solución
            
        Example:
            >>> solution = CSPSolution({'x': 1, 'y': 2})
            >>> term = bridge.solution_to_cubical_term(solution, problem)
        """
        # 1. Extraer valores de variables
        values = []
        for var in problem.variables:
            val = solution.assignment[var]
            values.append(Term.Value(val))
        
        # 2. Construir caminos para restricciones
        paths = []
        for (var1, var2, constraint) in problem.constraints:
            val1 = solution.assignment[var1]
            val2 = solution.assignment[var2]
            
            if constraint(val1, val2):
                # Construir camino que prueba la restricción
                path = self._build_constraint_path(
                    var1, val1, var2, val2, constraint
                )
                paths.append(path)
            else:
                # Solución inválida
                raise ValueError(
                    f"Constraint {var1}-{var2} violated: "
                    f"{val1}, {val2}"
                )
        
        # 3. Construir término Sigma anidado
        term = Term.Pair(*values, *paths)
        
        return term
    
    def _build_constraint_path(
        self, 
        var1: str, 
        val1, 
        var2: str, 
        val2, 
        constraint: callable
    ) -> Term:
        """Construye un camino que prueba una restricción."""
        # El camino es una prueba de que constraint(val1, val2) = True
        # En el sistema cúbico, esto es un camino trivial (reflexividad)
        # si la restricción se satisface
        
        return Term.ReflPath(
            source=Term.Value(val1),
            target=Term.Value(val2),
            proof=Term.ConstraintProof(constraint, val1, val2)
        )
    
    def verify_solution_cubical(
        self, 
        solution: CSPSolution, 
        problem: CSPProblem
    ) -> bool:
        """Verifica una solución usando type checking cúbico.
        
        Args:
            solution: Solución a verificar
            problem: Problema CSP original
            
        Returns:
            True si la solución es válida, False en caso contrario
            
        Example:
            >>> is_valid = bridge.verify_solution_cubical(solution, problem)
            >>> print(f"Solution valid: {is_valid}")
        """
        try:
            # 1. Traducir problema a tipo
            problem_type = self.translate_problem_to_cubical_type(problem)
            
            # 2. Traducir solución a término
            solution_term = self.solution_to_cubical_term(solution, problem)
            
            # 3. Type check
            is_valid = self.type_checker.type_check(
                self.context,
                solution_term,
                problem_type
            )
            
            return is_valid
            
        except Exception as e:
            # Cualquier error en la traducción o type checking
            # indica solución inválida
            return False
    
    def find_equivalent_solutions(
        self, 
        solutions: list[CSPSolution], 
        problem: CSPProblem
    ) -> list[list[CSPSolution]]:
        """Agrupa soluciones en clases de equivalencia.
        
        Args:
            solutions: Lista de soluciones CSP
            problem: Problema CSP original
            
        Returns:
            Lista de clases de equivalencia (listas de soluciones)
            
        Example:
            >>> equiv_classes = bridge.find_equivalent_solutions(
            ...     all_solutions, problem
            ... )
            >>> print(f"Found {len(equiv_classes)} equivalence classes")
        """
        equiv_classes = []
        
        for solution in solutions:
            # Convertir a término cúbico
            term = self.solution_to_cubical_term(solution, problem)
            
            # Buscar clase de equivalencia existente
            found_class = False
            for equiv_class in equiv_classes:
                # Verificar si existe camino a algún término de la clase
                representative = equiv_class[0]
                repr_term = self.solution_to_cubical_term(
                    representative, problem
                )
                
                if self._exists_path(term, repr_term):
                    equiv_class.append(solution)
                    found_class = True
                    break
            
            if not found_class:
                # Crear nueva clase de equivalencia
                equiv_classes.append([solution])
        
        return equiv_classes
    
    def _exists_path(self, term1: Term, term2: Term) -> bool:
        """Verifica si existe un camino entre dos términos."""
        # Intentar construir un camino
        try:
            path = self.type_checker.construct_path(term1, term2)
            return path is not None
        except:
            return False
    
    def get_statistics(self) -> dict:
        """Obtiene estadísticas de la traducción."""
        return {
            'context_size': len(self.context.variables),
            'type_checks_performed': self.type_checker.checks_count,
            'paths_constructed': self.type_checker.paths_count,
            'normalizations': self.type_checker.normalizations_count
        }
```

---

## Ejemplos de Uso

### Ejemplo 1: Coloración de Grafos

```python
from lattice_weaver.arc_engine import CSPProblem, CSPSolution
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge

# Definir problema: colorear triángulo con 3 colores
problem = CSPProblem(
    variables=['n1', 'n2', 'n3'],
    domains={
        'n1': {'red', 'blue', 'green'},
        'n2': {'red', 'blue', 'green'},
        'n3': {'red', 'blue', 'green'}
    },
    constraints=[
        ('n1', 'n2', lambda a, b: a != b),
        ('n2', 'n3', lambda a, b: a != b),
        ('n1', 'n3', lambda a, b: a != b)
    ]
)

# Crear puente
bridge = CSPToCubicalBridge()

# Traducir a tipo cúbico
cubical_type = bridge.translate_problem_to_cubical_type(problem)
print(f"Cubical type: {cubical_type}")

# Verificar solución
solution = CSPSolution({'n1': 'red', 'n2': 'blue', 'n3': 'green'})
is_valid = bridge.verify_solution_cubical(solution, problem)
print(f"Solution valid: {is_valid}")

# Estadísticas
stats = bridge.get_statistics()
print(f"Type checks: {stats['type_checks_performed']}")
```

---

### Ejemplo 2: Equivalencia de Soluciones

```python
# Encontrar todas las soluciones
from lattice_weaver.arc_engine import solve_csp

all_solutions = solve_csp(problem, return_all=True)
print(f"Total solutions: {len(all_solutions)}")

# Agrupar por equivalencia
equiv_classes = bridge.find_equivalent_solutions(all_solutions, problem)
print(f"Equivalence classes: {len(equiv_classes)}")

for i, equiv_class in enumerate(equiv_classes):
    print(f"\nClass {i+1}: {len(equiv_class)} solutions")
    for sol in equiv_class[:3]:  # Mostrar primeras 3
        print(f"  {sol.assignment}")
```

---

## Tests

### Tests Unitarios

```python
# tests/unit/test_csp_cubical_bridge.py

import pytest
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge
from lattice_weaver.arc_engine import CSPProblem, CSPSolution

class TestCSPToCubicalBridge:
    
    def test_translate_simple_problem(self):
        """Test traducción de problema simple."""
        problem = CSPProblem(
            variables=['x', 'y'],
            domains={'x': {1, 2}, 'y': {1, 2}},
            constraints=[('x', 'y', lambda a, b: a != b)]
        )
        
        bridge = CSPToCubicalBridge()
        cubical_type = bridge.translate_problem_to_cubical_type(problem)
        
        assert cubical_type is not None
        assert isinstance(cubical_type, SigmaType)
    
    def test_solution_to_term(self):
        """Test conversión de solución a término."""
        problem = CSPProblem(
            variables=['x', 'y'],
            domains={'x': {1, 2}, 'y': {1, 2}},
            constraints=[('x', 'y', lambda a, b: a != b)]
        )
        
        solution = CSPSolution({'x': 1, 'y': 2})
        
        bridge = CSPToCubicalBridge()
        term = bridge.solution_to_cubical_term(solution, problem)
        
        assert term is not None
    
    def test_verify_valid_solution(self):
        """Test verificación de solución válida."""
        problem = CSPProblem(
            variables=['x', 'y'],
            domains={'x': {1, 2}, 'y': {1, 2}},
            constraints=[('x', 'y', lambda a, b: a != b)]
        )
        
        solution = CSPSolution({'x': 1, 'y': 2})
        
        bridge = CSPToCubicalBridge()
        is_valid = bridge.verify_solution_cubical(solution, problem)
        
        assert is_valid == True
    
    def test_verify_invalid_solution(self):
        """Test verificación de solución inválida."""
        problem = CSPProblem(
            variables=['x', 'y'],
            domains={'x': {1, 2}, 'y': {1, 2}},
            constraints=[('x', 'y', lambda a, b: a != b)]
        )
        
        solution = CSPSolution({'x': 1, 'y': 1})  # Viola restricción
        
        bridge = CSPToCubicalBridge()
        is_valid = bridge.verify_solution_cubical(solution, problem)
        
        assert is_valid == False
```

---

### Tests de Integración

```python
# tests/integration/test_csp_to_cubical_flow.py

def test_full_csp_to_cubical_verification():
    """Test flujo completo CSP → Cubical → Verificación."""
    # 1. Definir problema CSP
    problem = create_graph_coloring_problem(n_nodes=4, n_colors=3)
    
    # 2. Resolver con motor CSP
    solutions = solve_csp(problem, return_all=True)
    
    # 3. Traducir a tipos cúbicos
    bridge = CSPToCubicalBridge()
    cubical_type = bridge.translate_problem_to_cubical_type(problem)
    
    # 4. Verificar cada solución con type checker cúbico
    for solution in solutions:
        is_valid = bridge.verify_solution_cubical(solution, problem)
        assert is_valid == True
    
    # 5. Agrupar por equivalencia
    equiv_classes = bridge.find_equivalent_solutions(solutions, problem)
    
    # 6. Verificar que hay menos clases que soluciones
    assert len(equiv_classes) <= len(solutions)
```

---

## Optimizaciones

### 1. Caching de Type Checking

```python
class CSPToCubicalBridge:
    def __init__(self):
        self.type_check_cache = {}
    
    def verify_solution_cubical(self, solution, problem):
        # Usar hash de la solución como clave
        cache_key = hash(frozenset(solution.assignment.items()))
        
        if cache_key in self.type_check_cache:
            return self.type_check_cache[cache_key]
        
        result = self._verify_solution_cubical_uncached(solution, problem)
        self.type_check_cache[cache_key] = result
        
        return result
```

### 2. Normalización Lazy

```python
class CubicalTypeChecker:
    def type_check(self, context, term, type):
        # Solo normalizar si es necesario
        if self._needs_normalization(term):
            term = self.normalize(term)
        
        return self._type_check_normalized(context, term, type)
```

### 3. Fast Path para Restricciones Simples

```python
def _constraint_to_path_type(self, var1, var2, constraint):
    # Detectar restricciones simples (!=, <, >, etc.)
    if self._is_simple_constraint(constraint):
        return self._simple_constraint_path(var1, var2, constraint)
    else:
        return self._general_constraint_path(var1, var2, constraint)
```

---

## Benchmarks

### Métricas a Medir

1. **Tiempo de traducción** CSP → Tipo Cúbico
2. **Tiempo de verificación** con type checker cúbico
3. **Memoria utilizada** para términos cúbicos
4. **Speedup** vs integración Sigma

### Casos de Prueba

- N-Queens (4, 8, 12, 16)
- Graph Coloring (10, 20, 50 nodos)
- Sudoku (4×4, 9×9)

---

## Roadmap de Implementación

### Fase 1: Diseño (Semanas 1-2)
- ✅ Especificación completa
- ✅ Diseño de API
- ✅ Documentación

### Fase 2: Implementación Core (Semanas 3-4)
- Implementar `CSPToCubicalBridge`
- Traducción de problemas
- Traducción de soluciones

### Fase 3: Verificación (Semanas 5-6)
- Integrar con type checker cúbico
- Implementar verificación
- Equivalencia de soluciones

### Fase 4: Tests y Optimización (Semanas 7-8)
- Tests unitarios completos
- Tests de integración
- Optimizaciones
- Benchmarks

---

## Conclusión

La integración CSP ↔ Tipos Cúbicos es un componente **crítico** que conecta el motor de resolución CSP con el sistema formal de tipos cúbicos, permitiendo:

- ✅ Verificación formal de soluciones
- ✅ Identificación de equivalencias
- ✅ Optimizaciones basadas en estructura cúbica
- ✅ Fundamentos teóricos sólidos

**Estimación total:** 8 semanas de desarrollo  
**Prioridad:** CRÍTICA  
**Dependencias:** `arc_engine`, `formal/cubical_*`

