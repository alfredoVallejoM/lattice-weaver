# Diseño de Integración Incremental para LatticeWeaver

**Versión**: 1.0  
**Fecha**: 15 de Octubre, 2025  
**Autor**: Manus AI  
**Estado**: Propuesta de Diseño

---

## 1. Objetivo

Diseñar e implementar un plan de integración incremental que conecte progresivamente las 332 clases disponibles en el repositorio con el solver principal (`CSPSolver`), mejorando iterativamente su rendimiento y capacidades mientras se mantiene la estabilidad y se previenen regresiones.

---

## 2. Principios de Diseño

Este diseño se adhiere estrictamente a los **Meta-Principios de LatticeWeaver**:

### 2.1 Dinamismo
El solver debe adaptarse dinámicamente a las características del problema, seleccionando estrategias óptimas según el contexto.

### 2.2 Distribución y Paralelización
Las mejoras deben aprovechar capacidades de paralelización donde sea posible, preparando el terreno para escalabilidad horizontal futura.

### 2.3 No Redundancia y Canonicalización
Se evitará duplicar funcionalidad existente. Se reutilizará código probado de `simple_backtracking_solver.py` y otros módulos.

### 2.4 Aprovechamiento de la Información
Cada decisión, fallo y éxito debe ser capturado y aprovechado mediante no-good learning, memoización y análisis estructural.

### 2.5 Gestión de Memoria Eficiente
Las integraciones deben minimizar el overhead de memoria mediante object pooling, lazy evaluation y paging cuando sea necesario.

### 2.6 Economía Computacional
Cada integración debe justificar su costo computacional con mejoras medibles en rendimiento o capacidades.

---

## 3. Estrategia de Integración Incremental

La integración se realizará en **6 fases incrementales**, cada una construyendo sobre la anterior. Cada fase incluye:

1. **Diseño detallado** de los cambios
2. **Implementación** del código
3. **Tests exhaustivos** (>90% cobertura)
4. **Benchmarking** antes/después
5. **Documentación** actualizada
6. **Validación** del usuario

### Filosofía: "Mejora Continua Validada"

Cada fase debe demostrar mejoras medibles antes de avanzar a la siguiente. Si una fase introduce regresiones, se revierte y se rediseña.

---

## 4. Fase 1: Integración de Heurísticas Básicas (MRV/Degree/LCV)

### 4.1 Objetivo

Mejorar la selección de variables y valores en `CSPSolver` integrando heurísticas probadas de `simple_backtracking_solver.py`.

### 4.2 Justificación

El `CSPSolver` actual usa selección simplista ("primera variable no asignada"). Las heurísticas MRV y Degree ya están implementadas y probadas en `simple_backtracking_solver.py`, reduciendo nodos explorados en 50-90% según la literatura.

### 4.3 Cambios Necesarios

#### A. Modificar `CSPSolver._select_unassigned_variable()`

**Archivo**: `lattice_weaver/core/csp_engine/solver.py`

**Código Actual**:
```python
def _select_unassigned_variable(self, current_domains: Dict[str, List[Any]]) -> Optional[str]:
    # Implementación simple: seleccionar la primera variable no asignada
    for var in self.csp.variables:
        if var not in self.assignment:
            return var
    return None
```

**Código Propuesto**:
```python
def _select_unassigned_variable(self, current_domains: Dict[str, List[Any]]) -> Optional[str]:
    """
    Selecciona la siguiente variable a asignar usando heurísticas MRV y Degree.
    
    MRV (Minimum Remaining Values): Selecciona la variable con menos valores legales.
    Degree: Como desempate, selecciona la variable involucrada en más restricciones
            con variables no asignadas.
    
    Returns:
        Variable a asignar, o None si todas están asignadas
    """
    unassigned_vars = [v for v in self.csp.variables if v not in self.assignment]
    if not unassigned_vars:
        return None
    
    # Calcular degree para cada variable no asignada
    degrees = {}
    for var in unassigned_vars:
        degree = 0
        for constraint in self.csp.constraints:
            if var in constraint.scope:
                # Contar cuántas otras variables no asignadas están en la misma restricción
                for other_var in constraint.scope:
                    if other_var != var and other_var in unassigned_vars:
                        degree += 1
        degrees[var] = degree
    
    # Combinar MRV y Degree: priorizar MRV, luego Degree (mayor degree primero)
    return min(unassigned_vars, key=lambda var: (len(current_domains[var]), -degrees[var]))
```

**Impacto Esperado**: Reducción de 50-90% en nodos explorados

**Riesgo**: Muy bajo (código ya probado en `simple_backtracking_solver.py`)

#### B. Implementar LCV (Least Constraining Value)

**Nuevo Método**:
```python
def _order_domain_values(self, var: str, current_domains: Dict[str, List[Any]]) -> List[Any]:
    """
    Ordena los valores del dominio de una variable usando LCV.
    
    LCV (Least Constraining Value): Ordena valores para probar primero aquellos
    que eliminan menos opciones de las variables vecinas.
    
    Args:
        var: Variable cuyo dominio ordenar
        current_domains: Dominios actuales de todas las variables
    
    Returns:
        Lista de valores ordenados (menos restrictivos primero)
    """
    domain = current_domains[var]
    
    # Para cada valor, contar cuántos valores elimina de variables vecinas
    value_constraints = []
    for value in domain:
        eliminated_count = 0
        
        # Revisar cada restricción que involucra a var
        for constraint in self.csp.constraints:
            if var in constraint.scope and len(constraint.scope) == 2:
                other_var = next((v for v in constraint.scope if v != var), None)
                if other_var and other_var not in self.assignment:
                    # Contar cuántos valores de other_var son incompatibles con value
                    for other_value in current_domains[other_var]:
                        if var == list(constraint.scope)[0]:
                            if not constraint.relation(value, other_value):
                                eliminated_count += 1
                        else:
                            if not constraint.relation(other_value, value):
                                eliminated_count += 1
        
        value_constraints.append((value, eliminated_count))
    
    # Ordenar por número de valores eliminados (menos eliminados primero)
    value_constraints.sort(key=lambda x: x[1])
    return [value for value, _ in value_constraints]
```

**Modificar `_backtrack()` para usar LCV**:
```python
def _backtrack(self, current_domains: Dict[str, List[Any]], all_solutions: bool, max_solutions: int) -> bool:
    self.stats.nodes_explored += 1
    
    if len(self.assignment) == len(self.csp.variables):
        solution = CSPSolution(assignment=self.assignment.copy())
        self.stats.solutions.append(solution)
        return all_solutions
    
    var = self._select_unassigned_variable(current_domains)
    if var is None:
        return True
    
    # CAMBIO: Usar LCV para ordenar valores
    ordered_values = self._order_domain_values(var, current_domains)
    
    for value in ordered_values:  # Usar valores ordenados
        if self._is_consistent(var, value):
            self.assignment[var] = value
            if self.tracer and self.tracer.enabled:
                self.tracer.record_assignment(variable=var, value=value, depth=len(self.assignment))
            
            new_domains = {v: list(d) for v, d in current_domains.items()}
            pruned_values = self._forward_check(var, value, new_domains)
            
            if pruned_values is not None:
                if self._backtrack(new_domains, all_solutions, max_solutions):
                    if not all_solutions or len(self.stats.solutions) >= max_solutions:
                        return True
            else:
                self.stats.backtracks += 1
                if self.tracer and self.tracer.enabled:
                    self.tracer.record_backtrack(variable=var, value=value, depth=len(self.assignment))
        
        del self.assignment[var]
    
    return False
```

**Impacto Esperado**: Reducción adicional de backtracking

**Riesgo**: Bajo

### 4.4 Tests Requeridos

**Archivo**: `tests/unit/test_csp_solver_heuristics.py`

```python
import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.problems.generators.nqueens import NQueensProblem

class TestCSPSolverHeuristics:
    """Tests para heurísticas de CSPSolver"""
    
    def test_mrv_selects_most_constrained_variable(self):
        """MRV debe seleccionar la variable con menor dominio"""
        # Crear CSP simple donde una variable tiene dominio más pequeño
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2, 3]),
                'B': frozenset([1]),  # Dominio más pequeño
                'C': frozenset([1, 2])
            },
            constraints=[]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2, 3],
            'B': [1],
            'C': [1, 2]
        }
        
        selected = solver._select_unassigned_variable(current_domains)
        assert selected == 'B', "MRV debe seleccionar variable con menor dominio"
    
    def test_degree_heuristic_as_tiebreaker(self):
        """Degree debe usarse como desempate cuando MRV empata"""
        # Crear CSP donde dos variables tienen mismo tamaño de dominio
        # pero una está más conectada
        csp = CSP(
            variables=['A', 'B', 'C'],
            domains={
                'A': frozenset([1, 2]),
                'B': frozenset([1, 2]),
                'C': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('A', 'C'), relation=lambda a, c: a != c),
                Constraint(scope=('B', 'C'), relation=lambda b, c: b != c),
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
            ]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2],
            'B': [1, 2],
            'C': [1, 2, 3]
        }
        
        # A y B tienen mismo dominio, pero A tiene más conexiones
        selected = solver._select_unassigned_variable(current_domains)
        assert selected in ['A', 'B'], "Debe seleccionar A o B (mismo MRV)"
    
    def test_lcv_orders_least_constraining_first(self):
        """LCV debe ordenar valores menos restrictivos primero"""
        csp = CSP(
            variables=['A', 'B'],
            domains={
                'A': frozenset([1, 2]),
                'B': frozenset([1, 2, 3])
            },
            constraints=[
                Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
            ]
        )
        
        solver = CSPSolver(csp)
        current_domains = {
            'A': [1, 2],
            'B': [1, 2, 3]
        }
        
        # Para A, valor 1 elimina 1 de B, valor 2 elimina 1 de B
        # Ambos son igualmente restrictivos
        ordered = solver._order_domain_values('A', current_domains)
        assert len(ordered) == 2
        assert set(ordered) == {1, 2}
    
    def test_nqueens_8_with_heuristics(self):
        """Test de regresión: N-Queens 8x8 debe resolverse correctamente"""
        problem = NQueensProblem(n=8)
        csp = problem.to_csp()
        solver = CSPSolver(csp)
        
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        assert len(stats.solutions) == 1, "Debe encontrar al menos una solución"
        assert stats.solutions[0].is_consistent, "La solución debe ser consistente"
        
        # Verificar que usa menos nodos que sin heurísticas
        # (esto requeriría comparación con versión anterior)
    
    def test_heuristics_reduce_nodes_explored(self):
        """Las heurísticas deben reducir nodos explorados vs. selección simple"""
        # Este test compararía con una versión sin heurísticas
        # Por ahora, solo verificamos que funciona
        problem = NQueensProblem(n=6)
        csp = problem.to_csp()
        solver = CSPSolver(csp)
        
        stats = solver.solve(all_solutions=False, max_solutions=1)
        
        # Con heurísticas, N-Queens 6x6 debería resolver en <100 nodos
        assert stats.nodes_explored < 100, f"Exploró {stats.nodes_explored} nodos, esperaba <100"
```

**Cobertura Esperada**: >90%

### 4.5 Benchmarking

**Script**: `scripts/benchmark_phase1.py`

```python
#!/usr/bin/env python3
"""
Benchmark para Fase 1: Heurísticas MRV/Degree/LCV
"""

import time
from lattice_weaver.core.csp_engine.solver import CSPSolver
from lattice_weaver.problems.generators.nqueens import NQueensProblem
from lattice_weaver.problems.generators.sudoku import SudokuProblem
from lattice_weaver.problems.generators.graph_coloring import GraphColoringProblem

def benchmark_problem(problem, name):
    """Benchmark de un problema"""
    csp = problem.to_csp()
    solver = CSPSolver(csp)
    
    start = time.perf_counter()
    stats = solver.solve(all_solutions=False, max_solutions=1)
    elapsed = time.perf_counter() - start
    
    print(f"\n{name}:")
    print(f"  Tiempo: {elapsed:.4f}s")
    print(f"  Nodos explorados: {stats.nodes_explored}")
    print(f"  Backtracks: {stats.backtracks}")
    print(f"  Restricciones chequeadas: {stats.constraints_checked}")
    print(f"  Soluciones: {len(stats.solutions)}")
    
    return {
        'name': name,
        'time': elapsed,
        'nodes': stats.nodes_explored,
        'backtracks': stats.backtracks,
        'constraints': stats.constraints_checked,
        'solutions': len(stats.solutions)
    }

if __name__ == '__main__':
    print("="*80)
    print("BENCHMARK FASE 1: Heurísticas MRV/Degree/LCV")
    print("="*80)
    
    results = []
    
    # N-Queens
    for n in [4, 6, 8, 10]:
        problem = NQueensProblem(n=n)
        result = benchmark_problem(problem, f"N-Queens {n}x{n}")
        results.append(result)
    
    # Sudoku (si está disponible)
    # problem = SudokuProblem(difficulty='easy')
    # result = benchmark_problem(problem, "Sudoku Easy")
    # results.append(result)
    
    # Graph Coloring
    # problem = GraphColoringProblem(nodes=10, edges=20, colors=3)
    # result = benchmark_problem(problem, "Graph Coloring 10 nodes")
    # results.append(result)
    
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    for r in results:
        print(f"{r['name']:30s} | {r['time']:8.4f}s | {r['nodes']:6d} nodos | {r['backtracks']:6d} backtracks")
```

### 4.6 Criterios de Éxito

- ✅ Todos los tests pasan (>90% cobertura)
- ✅ Reducción de nodos explorados en al menos 30% vs. versión anterior
- ✅ No regresiones en tiempo de ejecución para problemas pequeños
- ✅ Documentación actualizada

### 4.7 Estimación

- **Esfuerzo**: 4-6 horas
- **Riesgo**: Muy bajo
- **Impacto**: Alto

---

## 5. Fase 2: Sistema de Estrategias Modulares

### 5.1 Objetivo

Refactorizar `CSPSolver` para usar el sistema de estrategias modulares ya implementado en `lattice_weaver/strategies/`.

### 5.2 Justificación

El sistema de estrategias permite seleccionar dinámicamente algoritmos de selección de variables, ordenamiento de valores y propagación según las características del problema. Esto prepara el terreno para integración de ML y análisis estructural.

### 5.3 Arquitectura Propuesta

```
CSPSolver
  ├── variable_selection_strategy: VariableSelectionStrategy
  ├── value_ordering_strategy: ValueOrderingStrategy
  ├── propagation_strategy: PropagationStrategy
  └── analysis_strategy: AnalysisStrategy (opcional)
```

### 5.4 Interfaces de Estrategias

**Archivo**: `lattice_weaver/strategies/base.py` (ya existe)

Revisar y extender si es necesario:

```python
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SolverContext:
    """Contexto del solver para estrategias"""
    csp: 'CSP'
    assignment: Dict[str, Any]
    current_domains: Dict[str, List[Any]]
    stats: Any  # CSPSolutionStats

class VariableSelectionStrategy(ABC):
    """Estrategia de selección de variables"""
    
    @abstractmethod
    def select_variable(self, context: SolverContext) -> Optional[str]:
        """Selecciona la siguiente variable a asignar"""
        pass

class ValueOrderingStrategy(ABC):
    """Estrategia de ordenamiento de valores"""
    
    @abstractmethod
    def order_values(self, var: str, context: SolverContext) -> List[Any]:
        """Ordena los valores del dominio de una variable"""
        pass

class PropagationStrategy(ABC):
    """Estrategia de propagación de restricciones"""
    
    @abstractmethod
    def propagate(self, var: str, value: Any, context: SolverContext) -> Optional[Dict[str, List[Any]]]:
        """Propaga asignación y retorna nuevos dominios, o None si inconsistente"""
        pass
```

### 5.5 Implementaciones Concretas

**Archivo**: `lattice_weaver/strategies/heuristics/mrv_degree.py` (nuevo)

```python
from ..base import VariableSelectionStrategy, SolverContext
from typing import Optional

class MRVDegreeStrategy(VariableSelectionStrategy):
    """Estrategia MRV con Degree como desempate"""
    
    def select_variable(self, context: SolverContext) -> Optional[str]:
        unassigned_vars = [v for v in context.csp.variables if v not in context.assignment]
        if not unassigned_vars:
            return None
        
        # Calcular degrees
        degrees = {}
        for var in unassigned_vars:
            degree = 0
            for constraint in context.csp.constraints:
                if var in constraint.scope:
                    for other_var in constraint.scope:
                        if other_var != var and other_var in unassigned_vars:
                            degree += 1
            degrees[var] = degree
        
        # MRV + Degree
        return min(unassigned_vars, 
                  key=lambda var: (len(context.current_domains[var]), -degrees[var]))
```

**Archivo**: `lattice_weaver/strategies/heuristics/lcv.py` (nuevo)

```python
from ..base import ValueOrderingStrategy, SolverContext
from typing import List, Any

class LCVStrategy(ValueOrderingStrategy):
    """Estrategia Least Constraining Value"""
    
    def order_values(self, var: str, context: SolverContext) -> List[Any]:
        domain = context.current_domains[var]
        value_constraints = []
        
        for value in domain:
            eliminated_count = 0
            
            for constraint in context.csp.constraints:
                if var in constraint.scope and len(constraint.scope) == 2:
                    other_var = next((v for v in constraint.scope if v != var), None)
                    if other_var and other_var not in context.assignment:
                        for other_value in context.current_domains[other_var]:
                            if var == list(constraint.scope)[0]:
                                if not constraint.relation(value, other_value):
                                    eliminated_count += 1
                            else:
                                if not constraint.relation(other_value, value):
                                    eliminated_count += 1
            
            value_constraints.append((value, eliminated_count))
        
        value_constraints.sort(key=lambda x: x[1])
        return [value for value, _ in value_constraints]
```

**Archivo**: `lattice_weaver/strategies/propagation/forward_checking.py` (nuevo)

```python
from ..base import PropagationStrategy, SolverContext
from typing import Any, Dict, List, Optional

class ForwardCheckingStrategy(PropagationStrategy):
    """Estrategia de Forward Checking"""
    
    def propagate(self, var: str, value: Any, context: SolverContext) -> Optional[Dict[str, List[Any]]]:
        new_domains = {v: list(d) for v, d in context.current_domains.items()}
        
        for constraint in context.csp.constraints:
            if var in constraint.scope and len(constraint.scope) == 2:
                other_var = next((v for v in constraint.scope if v != var), None)
                if other_var and other_var not in context.assignment:
                    original_domain = list(new_domains[other_var])
                    new_domains[other_var] = [
                        other_value for other_value in original_domain
                        if (var == list(constraint.scope)[0] and constraint.relation(value, other_value)) or
                           (var == list(constraint.scope)[1] and constraint.relation(other_value, value))
                    ]
                    
                    if not new_domains[other_var]:
                        return None  # Inconsistencia
        
        return new_domains
```

### 5.6 Refactorización de CSPSolver

**Archivo**: `lattice_weaver/core/csp_engine/solver.py`

```python
from lattice_weaver.strategies.base import (
    VariableSelectionStrategy, 
    ValueOrderingStrategy, 
    PropagationStrategy,
    SolverContext
)
from lattice_weaver.strategies.heuristics.mrv_degree import MRVDegreeStrategy
from lattice_weaver.strategies.heuristics.lcv import LCVStrategy
from lattice_weaver.strategies.propagation.forward_checking import ForwardCheckingStrategy

class CSPSolver:
    """Solver CSP con estrategias modulares"""
    
    def __init__(
        self, 
        csp: CSP, 
        tracer: Optional[ExecutionTracer] = None,
        variable_strategy: Optional[VariableSelectionStrategy] = None,
        value_strategy: Optional[ValueOrderingStrategy] = None,
        propagation_strategy: Optional[PropagationStrategy] = None
    ):
        self.csp = csp
        self.assignment: Dict[str, Any] = {}
        self.stats = CSPSolutionStats()
        self.tracer = tracer
        
        # Estrategias (usar defaults si no se especifican)
        self.variable_strategy = variable_strategy or MRVDegreeStrategy()
        self.value_strategy = value_strategy or LCVStrategy()
        self.propagation_strategy = propagation_strategy or ForwardCheckingStrategy()
    
    def _get_context(self, current_domains: Dict[str, List[Any]]) -> SolverContext:
        """Crea contexto para estrategias"""
        return SolverContext(
            csp=self.csp,
            assignment=self.assignment,
            current_domains=current_domains,
            stats=self.stats
        )
    
    def _select_unassigned_variable(self, current_domains: Dict[str, List[Any]]) -> Optional[str]:
        """Delega a estrategia de selección de variables"""
        context = self._get_context(current_domains)
        return self.variable_strategy.select_variable(context)
    
    def _order_domain_values(self, var: str, current_domains: Dict[str, List[Any]]) -> List[Any]:
        """Delega a estrategia de ordenamiento de valores"""
        context = self._get_context(current_domains)
        return self.value_strategy.order_values(var, context)
    
    def _forward_check(self, var: str, value: Any, domains: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        """Delega a estrategia de propagación"""
        context = self._get_context(domains)
        return self.propagation_strategy.propagate(var, value, context)
    
    # Resto de métodos sin cambios...
```

### 5.7 Criterios de Éxito

- ✅ Refactorización completa sin regresiones
- ✅ Tests pasan (>90% cobertura)
- ✅ Benchmarks muestran mismo rendimiento o mejor
- ✅ Posibilidad de intercambiar estrategias dinámicamente

### 5.8 Estimación

- **Esfuerzo**: 8-12 horas
- **Riesgo**: Medio
- **Impacto**: Alto (prepara para ML y análisis)

---

## 6. Fase 3: Integración de FCA para Análisis Estructural

### 6.1 Objetivo

Integrar `LatticeBuilder` para detectar implicaciones y estructura en el CSP antes de resolver.

### 6.2 Justificación

FCA puede revelar implicaciones ocultas entre restricciones, simplificando el problema antes de la búsqueda.

### 6.3 Diseño

**Nuevo Módulo**: `lattice_weaver/strategies/analysis/fca.py`

```python
from lattice_weaver.strategies.base import AnalysisStrategy, AnalysisResult, SolverContext
from lattice_weaver.lattice_core.builder import LatticeBuilder
from lattice_weaver.lattice_core.context import FormalContext

class FCAAnalysisStrategy(AnalysisStrategy):
    """Estrategia de análisis usando FCA"""
    
    def analyze(self, context: SolverContext) -> AnalysisResult:
        """Analiza el CSP usando FCA"""
        
        # Construir contexto formal desde CSP
        formal_context = self._build_formal_context(context.csp)
        
        # Construir lattice de conceptos
        builder = LatticeBuilder(formal_context)
        concepts = builder.build_lattice()
        
        # Detectar implicaciones
        implications = self._extract_implications(concepts, formal_context)
        
        # Simplificar CSP usando implicaciones
        simplified_csp = self._apply_implications(context.csp, implications)
        
        return AnalysisResult(
            simplified_csp=simplified_csp,
            implications=implications,
            concepts=concepts,
            metadata={'num_concepts': len(concepts)}
        )
    
    def _build_formal_context(self, csp: 'CSP') -> FormalContext:
        """Construye contexto formal desde CSP"""
        # Objetos: pares (variable, valor)
        # Atributos: restricciones satisfechas
        
        objects = set()
        attributes = set()
        incidence = set()
        
        # Crear objetos (variable, valor)
        for var in csp.variables:
            for val in csp.domains[var]:
                objects.add((var, val))
        
        # Crear atributos (restricciones)
        for i, constraint in enumerate(csp.constraints):
            attributes.add(f"C{i}")
        
        # Crear incidencia: (var, val) tiene atributo Ci si satisface restricción i
        for obj in objects:
            var, val = obj
            for i, constraint in enumerate(csp.constraints):
                if var in constraint.scope:
                    # Verificar si (var, val) puede satisfacer la restricción
                    # (simplificado: solo restricciones unarias)
                    if len(constraint.scope) == 1:
                        if constraint.relation(val):
                            incidence.add((obj, f"C{i}"))
        
        return FormalContext(objects, attributes, incidence)
    
    def _extract_implications(self, concepts, formal_context):
        """Extrae implicaciones del lattice"""
        # Implementación simplificada
        implications = []
        # TODO: Implementar extracción de implicaciones
        return implications
    
    def _apply_implications(self, csp, implications):
        """Aplica implicaciones para simplificar CSP"""
        # Implementación simplificada
        # TODO: Reducir dominios usando implicaciones
        return csp
```

### 6.4 Integración en CSPSolver

```python
class CSPSolver:
    def __init__(
        self, 
        csp: CSP, 
        tracer: Optional[ExecutionTracer] = None,
        variable_strategy: Optional[VariableSelectionStrategy] = None,
        value_strategy: Optional[ValueOrderingStrategy] = None,
        propagation_strategy: Optional[PropagationStrategy] = None,
        analysis_strategy: Optional[AnalysisStrategy] = None,  # NUEVO
        use_analysis: bool = False  # NUEVO
    ):
        self.csp = csp
        self.assignment: Dict[str, Any] = {}
        self.stats = CSPSolutionStats()
        self.tracer = tracer
        
        self.variable_strategy = variable_strategy or MRVDegreeStrategy()
        self.value_strategy = value_strategy or LCVStrategy()
        self.propagation_strategy = propagation_strategy or ForwardCheckingStrategy()
        self.analysis_strategy = analysis_strategy  # NUEVO
        self.use_analysis = use_analysis  # NUEVO
    
    def solve(self, all_solutions: bool = False, max_solutions: int = 1) -> CSPSolutionStats:
        start_time = time.perf_counter()
        
        # NUEVO: Análisis estructural opcional
        if self.use_analysis and self.analysis_strategy:
            context = self._get_context({var: list(self.csp.domains[var]) for var in self.csp.variables})
            analysis_result = self.analysis_strategy.analyze(context)
            
            # Usar CSP simplificado si está disponible
            if analysis_result.simplified_csp:
                self.csp = analysis_result.simplified_csp
        
        initial_domains = {var: list(self.csp.domains[var]) for var in self.csp.variables}
        self._backtrack(initial_domains, all_solutions, max_solutions)
        end_time = time.perf_counter()
        self.stats.time_elapsed = end_time - start_time
        return self.stats
```

### 6.5 Criterios de Éxito

- ✅ FCA se ejecuta correctamente en CSPs de prueba
- ✅ Detección de al menos algunas implicaciones en problemas estructurados
- ✅ No regresiones en rendimiento cuando `use_analysis=False`
- ✅ Mejoras medibles en problemas con estructura (ej. Sudoku)

### 6.6 Estimación

- **Esfuerzo**: 12-16 horas
- **Riesgo**: Medio-Alto
- **Impacto**: Medio (depende del tipo de problema)

---

## 7. Fase 4: Adaptación de TopologyAnalyzer

### 7.1 Objetivo

Adaptar `TopologyAnalyzer` para trabajar directamente con `CSP` sin requerir `arc_engine`.

### 7.2 Problema Actual

`TopologyAnalyzer` espera un `arc_engine` que no existe. Necesitamos crear un adaptador.

### 7.3 Diseño

**Opción A: Adaptador CSP → ArcEngine Interface**

```python
class CSPToArcEngineAdapter:
    """Adaptador que presenta CSP como si fuera ArcEngine"""
    
    def __init__(self, csp: CSP):
        self.csp = csp
        self.variables = {
            var: MockVariable(var, csp.domains[var])
            for var in csp.variables
        }
        self.constraints = {
            i: MockConstraint(c)
            for i, c in enumerate(csp.constraints)
        }

class MockVariable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
    
    def get_values(self):
        return list(self.domain)

class MockConstraint:
    def __init__(self, constraint):
        self.constraint = constraint
        self.var1 = list(constraint.scope)[0] if len(constraint.scope) > 0 else None
        self.var2 = list(constraint.scope)[1] if len(constraint.scope) > 1 else None
        self.relation = constraint.relation
```

**Opción B: Refactorizar TopologyAnalyzer**

Modificar `TopologyAnalyzer` para aceptar `CSP` directamente:

```python
class TopologyAnalyzer:
    def __init__(self, csp: CSP):  # Cambiar de arc_engine a csp
        self.csp = csp
        self.consistency_graph = nx.Graph()
        # ...
    
    def build_consistency_graph(self):
        # Adaptar para trabajar con CSP directamente
        for var in self.csp.variables:
            for value in self.csp.domains[var]:
                node = (var, value)
                self.consistency_graph.add_node(node)
        
        for constraint in self.csp.constraints:
            if len(constraint.scope) == 2:
                var1, var2 = list(constraint.scope)
                for val1 in self.csp.domains[var1]:
                    for val2 in self.csp.domains[var2]:
                        if constraint.relation(val1, val2):
                            self.consistency_graph.add_edge((var1, val1), (var2, val2))
```

**Recomendación**: Opción B (refactorizar) es más limpia y sostenible.

### 7.4 Integración

```python
from lattice_weaver.strategies.base import AnalysisStrategy, AnalysisResult
from lattice_weaver.topology.analyzer import TopologyAnalyzer

class TopologicalAnalysisStrategy(AnalysisStrategy):
    """Estrategia de análisis topológico"""
    
    def analyze(self, context: SolverContext) -> AnalysisResult:
        analyzer = TopologyAnalyzer(context.csp)
        analyzer.build_consistency_graph()
        
        # Calcular métricas topológicas
        betti_numbers = analyzer.calculate_betti_numbers()
        components = analyzer.find_connected_components()
        
        return AnalysisResult(
            metadata={
                'betti_numbers': betti_numbers,
                'num_components': len(components),
                'graph_density': analyzer.get_graph_density()
            }
        )
```

### 7.5 Estimación

- **Esfuerzo**: 8-12 horas
- **Riesgo**: Medio
- **Impacto**: Medio (información para selección adaptativa)

---

## 8. Fase 5: Integración Básica de Mini-IAs

### 8.1 Objetivo

Integrar las Mini-IAs más simples para guiar selección de variables y valores.

### 8.2 Mini-IAs a Integrar

1. `VariableSelectorMiniIA`: Predicción de mejor variable
2. `ValueSelectorMiniIA`: Predicción de mejor valor
3. `BacktrackPredictorMiniIA`: Predicción de probabilidad de backtrack

### 8.3 Desafío: Entrenamiento

Las Mini-IAs requieren datos de entrenamiento. Estrategia:

1. **Fase 5a**: Generar datos de entrenamiento ejecutando solver en problemas variados
2. **Fase 5b**: Entrenar modelos básicos
3. **Fase 5c**: Integrar modelos entrenados

### 8.4 Diseño

**Estrategia ML-Guided**:

```python
from lattice_weaver.ml.mini_nets.csp_advanced import VariableSelectorMiniIA
from lattice_weaver.ml.adapters.feature_extractors import CSPFeatureExtractor

class MLGuidedVariableStrategy(VariableSelectionStrategy):
    """Estrategia de selección guiada por ML"""
    
    def __init__(self, model_path: str):
        self.model = VariableSelectorMiniIA()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.feature_extractor = CSPFeatureExtractor()
    
    def select_variable(self, context: SolverContext) -> Optional[str]:
        unassigned = [v for v in context.csp.variables if v not in context.assignment]
        if not unassigned:
            return None
        
        # Extraer features
        features = self.feature_extractor.extract(context)
        
        # Predecir scores para cada variable
        scores = {}
        for var in unassigned:
            var_features = self._get_variable_features(var, features)
            with torch.no_grad():
                score = self.model(var_features)
            scores[var] = score.item()
        
        # Seleccionar variable con mayor score
        return max(scores, key=scores.get)
```

### 8.5 Estimación

- **Esfuerzo**: 20-30 horas (incluyendo generación de datos y entrenamiento)
- **Riesgo**: Alto
- **Impacto**: Potencialmente muy alto (1.5-100x según docs)

---

## 9. Fase 6: Selección Adaptativa de Estrategias

### 9.1 Objetivo

Implementar meta-análisis para seleccionar automáticamente las mejores estrategias según características del problema.

### 9.2 Diseño

```python
from lattice_weaver.meta.analyzer import MetaAnalyzer

class AdaptiveCSPSolver(CSPSolver):
    """Solver que selecciona estrategias adaptativamente"""
    
    def __init__(self, csp: CSP, **kwargs):
        # Analizar problema
        meta_analyzer = MetaAnalyzer()
        archetype = meta_analyzer.classify_problem(csp)
        
        # Seleccionar estrategias según arquetipo
        variable_strategy = self._select_variable_strategy(archetype)
        value_strategy = self._select_value_strategy(archetype)
        propagation_strategy = self._select_propagation_strategy(archetype)
        analysis_strategy = self._select_analysis_strategy(archetype)
        
        super().__init__(
            csp=csp,
            variable_strategy=variable_strategy,
            value_strategy=value_strategy,
            propagation_strategy=propagation_strategy,
            analysis_strategy=analysis_strategy,
            **kwargs
        )
```

### 9.3 Estimación

- **Esfuerzo**: 12-16 horas
- **Riesgo**: Medio
- **Impacto**: Alto (optimización automática)

---

## 10. Plan de Validación

Cada fase debe pasar:

1. **Tests unitarios** (>90% cobertura)
2. **Tests de integración**
3. **Benchmarking** (comparación antes/después)
4. **Revisión de código**
5. **Validación del usuario**

---

## 11. Cronograma Estimado

| Fase | Descripción | Esfuerzo | Riesgo | Semanas |
|------|-------------|----------|--------|---------|
| 1 | Heurísticas MRV/Degree/LCV | 4-6h | Muy bajo | 0.5 |
| 2 | Sistema de Estrategias | 8-12h | Medio | 1 |
| 3 | Integración FCA | 12-16h | Medio-Alto | 1.5 |
| 4 | Adaptación TopologyAnalyzer | 8-12h | Medio | 1 |
| 5 | Mini-IAs básicas | 20-30h | Alto | 2-3 |
| 6 | Selección Adaptativa | 12-16h | Medio | 1.5 |

**Total**: 64-92 horas (8-11.5 semanas a tiempo parcial)

---

## 12. Documentación a Actualizar

Cada fase debe actualizar:

1. **README.md**: Nuevas capacidades
2. **API_REFERENCE.md**: Nuevas clases y métodos
3. **QUICKSTART.md**: Ejemplos de uso
4. **CHANGELOG.md**: Cambios por versión
5. **Este documento**: Progreso de integración

---

## 13. Próximos Pasos Inmediatos

1. ✅ Validar este diseño con el usuario
2. ⏳ Crear rama `feature/integration-phase1`
3. ⏳ Implementar Fase 1
4. ⏳ Tests y benchmarking Fase 1
5. ⏳ PR y merge a `main`
6. ⏳ Repetir para fases subsiguientes

---

**Autor**: Manus AI  
**Fecha**: 15 de Octubre, 2025  
**Versión**: 1.0

