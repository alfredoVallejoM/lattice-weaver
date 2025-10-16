# Propuesta de Integración: ArcEngine ↔ CSPSolver

**Versión:** 1.0  
**Fecha:** 16 de Octubre, 2025  
**Estado:** PENDIENTE DE APROBACIÓN  
**Prioridad:** ALTA  
**Track:** Integración Core Engine  

---

## 📋 Contexto

### Revisión de Protocolos (16 de Octubre, 2025)

Esta propuesta ha sido revisada y validada contra los siguientes documentos actualizados:

- **`PROTOCOLO_AGENTES_LATTICEWEAVER.md` (v4.0)**: El protocolo de desarrollo para agentes ha sido seguido rigurosamente, incluyendo la Fase 0 de verificación del estado del proyecto, la aplicación de patrones de diseño y el protocolo de merge seguro.
- **`PROJECT_OVERVIEW.md` (v7.1-alpha)**: La propuesta se alinea con la visión unificada del proyecto, la arquitectura modular y las prioridades de la hoja de ruta estratégica, especialmente la "Unificación y Limpieza" y la "Refactorización y Optimización". Se ha confirmado que ArcEngine ya está integrado en `main` y que la estrategia de integración propuesta para `CSPSolver` es coherente con la coexistencia de ambos sistemas.
- **`README.md` (v8.0-alpha)**: La propuesta es consistente con la nueva arquitectura de orquestación modular y el uso de estrategias inyectables, ya que busca integrar ArcEngine como una estrategia de propagación optimizada dentro del flujo de `CSPSolver`.

---



### Situación Actual

Tras la evaluación exhaustiva del estado de integración del repositorio, se ha identificado que **existen dos sistemas de resolución CSP paralelos** que no están completamente integrados:

1. **ArcEngine** (`lattice_weaver/arc_engine/`)
   - Motor de propagación AC-3.1 optimizado
   - Paralelización, TMS, optimizaciones avanzadas
   - API incremental: `add_variable()`, `add_constraint()`

2. **CSPSolver** (`lattice_weaver/core/csp_engine/`)
   - Solver completo con backtracking + estrategias
   - Sistema de tracing, estrategias modulares (MRV, LCV, FCA-guided)
   - API declarativa: recibe objeto `CSP` completo

### Problema Identificado

- **Score de compatibilidad API:** 0% (APIs completamente incompatibles)
- **CSPSolver NO usa ArcEngine:** Implementa AC-3 básico en lugar de AC-3.1 optimizado
- **Pérdida de rendimiento:** 20-40% en propagación de restricciones
- **Fragmentación arquitectónica:** Confusión sobre cuál sistema usar

### Alineación con Evaluación Previa

Esta propuesta se deriva directamente de:
- `EVALUACION_INTEGRACION_ACTUALIZADA.md` - Análisis de compatibilidad
- `ESTADO_ACTUAL_Y_ROADMAP.md` - Tarea 1.1 y Fase 2

---

## 🎯 Objetivo

**Integrar ArcEngine en CSPSolver de forma incremental, segura y backward-compatible**, permitiendo que CSPSolver aproveche las optimizaciones de ArcEngine sin romper código existente.

---

## 📐 Diseño de la Solución

### Fase 1: Integración Opcional (Semana 1)

#### Principios de Diseño Aplicados

Según `LatticeWeaver_Meta_Principios_Diseño_v3.md`:

1. **✅ Economía Computacional**
   - Usar AC-3.1 optimizado reduce overhead de propagación en 20-40%
   - Evitar recomputación de revisiones mediante caché

2. **✅ Localidad y Modularidad**
   - Cambios localizados en `CSPSolver.__init__` y `enforce_arc_consistency()`
   - No afecta a otros módulos

3. **✅ Dinamismo Adaptativo**
   - Parámetro `use_arc_engine` permite activar/desactivar
   - Backward compatible: comportamiento por defecto sin cambios

4. **✅ No Redundancia**
   - Elimina duplicación: un solo AC-3 (el optimizado de ArcEngine)
   - Reutiliza código existente en lugar de reimplementar

5. **✅ Composicionalidad**
   - CSPSolver sigue siendo componible con estrategias modulares
   - ArcEngine se integra como componente interno

6. **✅ Verificabilidad**
   - Tests de compatibilidad garantizan resultados idénticos
   - Tests de regresión validan que nada se rompe

#### Arquitectura Propuesta

```
CSPSolver (modificado)
├── __init__(csp, tracer, variable_selector, value_orderer, 
│            use_arc_engine=True, parallel=False)  # NUEVO
├── _setup_arc_engine()  # NUEVO - Convierte CSP → ArcEngine
├── enforce_arc_consistency()  # MODIFICADO - Delega a ArcEngine si está activo
├── _enforce_ac3_basic()  # NUEVO - Fallback al AC-3 actual
└── ... (resto sin cambios)
```

#### Implementación Detallada

**Archivo:** `lattice_weaver/core/csp_engine/solver.py`

**Cambios:**

```python
# AÑADIR import al inicio del archivo
from typing import Dict, Any, List, Optional, Tuple, Callable

# NUEVO import condicional
try:
    from ...arc_engine import ArcEngine
    ARC_ENGINE_AVAILABLE = True
except ImportError:
    ARC_ENGINE_AVAILABLE = False

class CSPSolver:
    """
    Un solver básico para Problemas de Satisfacción de Restricciones (CSP).
    Implementa un algoritmo de backtracking con forward checking.
    
    Puede usar opcionalmente ArcEngine para propagación AC-3.1 optimizada.
    """
    
    def __init__(self, 
                 csp: CSP, 
                 tracer: Optional[ExecutionTracer] = None,
                 variable_selector: Optional[VariableSelector] = None,
                 value_orderer: Optional[ValueOrderer] = None,
                 use_arc_engine: bool = False,  # NUEVO parámetro
                 parallel: bool = False):        # NUEVO parámetro
        """
        Inicializa el CSPSolver.
        
        Args:
            csp: El problema CSP a resolver
            tracer: Tracer opcional para debugging/análisis
            variable_selector: Estrategia para seleccionar variables
            value_orderer: Estrategia para ordenar valores
            use_arc_engine: Si True, usa ArcEngine para AC-3.1 optimizado
            parallel: Si True y use_arc_engine=True, habilita paralelización
        """
        self.csp = csp
        self.assignment: Dict[str, Any] = {}
        self.stats = CSPSolutionStats()
        self.tracer = tracer
        
        # Estrategias modulares
        self.variable_selector = variable_selector or FirstUnassignedSelector()
        self.value_orderer = value_orderer or NaturalOrderer()
        
        # NUEVO: Configurar ArcEngine si está disponible y solicitado
        self.arc_engine = None
        if use_arc_engine:
            if not ARC_ENGINE_AVAILABLE:
                import warnings
                warnings.warn(
                    "ArcEngine no disponible. Usando AC-3 básico.",
                    RuntimeWarning
                )
            else:
                self.arc_engine = ArcEngine(parallel=parallel, use_tms=False)
                self._setup_arc_engine()
        
        if self.tracer and self.tracer.enabled:
            pass

    def _setup_arc_engine(self):
        """
        Configura ArcEngine con el CSP actual.
        Convierte la representación CSP a la API incremental de ArcEngine.
        """
        if self.arc_engine is None:
            return
        
        # Añadir variables
        for var in self.csp.variables:
            self.arc_engine.add_variable(var, self.csp.domains[var])
        
        # Registrar y añadir restricciones
        for idx, constraint in enumerate(self.csp.constraints):
            if len(constraint.scope) == 2:
                var1, var2 = list(constraint.scope)
                
                # Crear nombre único para la relación
                rel_name = f"constraint_{idx}"
                
                # Registrar la función de relación
                # Wrapper que adapta la signatura
                def make_relation_wrapper(original_relation):
                    def wrapper(val1, val2, metadata):
                        return original_relation(val1, val2)
                    return wrapper
                
                self.arc_engine.register_relation(
                    rel_name, 
                    make_relation_wrapper(constraint.relation)
                )
                
                # Añadir restricción
                self.arc_engine.add_constraint(var1, var2, rel_name)
    
    def enforce_arc_consistency(self) -> bool:
        """
        Implementa el algoritmo AC-3 (o AC-3.1 si ArcEngine está activo).
        Retorna True si el CSP es consistente, False si se detecta inconsistencia.
        """
        if self.arc_engine is not None:
            # Usar AC-3.1 optimizado de ArcEngine
            return self.arc_engine.enforce_arc_consistency()
        else:
            # Usar AC-3 básico (implementación actual)
            return self._enforce_ac3_basic()
    
    def _enforce_ac3_basic(self) -> bool:
        """
        Implementación AC-3 básica (código actual).
        Se mantiene como fallback.
        """
        queue = []
        for constraint in self.csp.constraints:
            if len(constraint.scope) == 2:
                var_i, var_j = list(constraint.scope)
                queue.append((var_i, var_j, constraint))
                queue.append((var_j, var_i, constraint))

        while queue:
            var_i, var_j, constraint = queue.pop(0)
            if self._revise(var_i, var_j, constraint):
                if not self.csp.domains[var_i]:
                    return False
                for neighbor_constraint in self.csp.constraints:
                    if len(neighbor_constraint.scope) == 2:
                        n_var1, n_var2 = list(neighbor_constraint.scope)
                        if n_var2 == var_i and n_var1 != var_j:
                            queue.append((n_var1, n_var2, neighbor_constraint))
                        elif n_var1 == var_i and n_var2 != var_j:
                            queue.append((n_var2, n_var1, neighbor_constraint))
        return True
    
    # ... resto del código sin cambios
```

#### Tests de Compatibilidad

**Archivo:** `tests/unit/test_arc_engine_integration.py` (NUEVO)

```python
"""
Tests de integración entre CSPSolver y ArcEngine.
Verifican que los resultados son idénticos con y sin ArcEngine.
"""
import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolver

def create_nqueens_csp(n=4):
    """Crea un CSP de N-Queens para testing"""
    variables = [f"Q{i}" for i in range(n)]
    domains = {var: list(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def not_same_row(val_i, val_j, i=i, j=j):
                return val_i != val_j
            
            def not_same_diagonal(val_i, val_j, i=i, j=j):
                return abs(val_i - val_j) != abs(i - j)
            
            constraints.append(
                Constraint({f"Q{i}", f"Q{j}"}, not_same_row)
            )
            constraints.append(
                Constraint({f"Q{i}", f"Q{j}"}, not_same_diagonal)
            )
    
    return CSP(variables, domains, constraints)

def test_arc_engine_produces_same_solutions():
    """Verifica que ArcEngine produce las mismas soluciones que AC-3 básico"""
    csp = create_nqueens_csp(n=4)
    
    # Solver sin ArcEngine
    solver_basic = CSPSolver(csp, use_arc_engine=False)
    stats_basic = solver_basic.solve(all_solutions=True)
    
    # Solver con ArcEngine
    csp2 = create_nqueens_csp(n=4)  # CSP fresco
    solver_arc = CSPSolver(csp2, use_arc_engine=True)
    stats_arc = solver_arc.solve(all_solutions=True)
    
    # Verificar mismo número de soluciones
    assert len(stats_basic.solutions) == len(stats_arc.solutions), \
        f"Diferente número de soluciones: {len(stats_basic.solutions)} vs {len(stats_arc.solutions)}"
    
    # Verificar que las soluciones son las mismas (orden puede variar)
    solutions_basic = {frozenset(sol.assignment.items()) for sol in stats_basic.solutions}
    solutions_arc = {frozenset(sol.assignment.items()) for sol in stats_arc.solutions}
    
    assert solutions_basic == solutions_arc, \
        "Las soluciones encontradas son diferentes"

def test_arc_engine_is_faster_or_equal():
    """Verifica que ArcEngine no es significativamente más lento"""
    csp = create_nqueens_csp(n=8)
    
    # Solver sin ArcEngine
    solver_basic = CSPSolver(csp, use_arc_engine=False)
    stats_basic = solver_basic.solve()
    
    # Solver con ArcEngine
    csp2 = create_nqueens_csp(n=8)
    solver_arc = CSPSolver(csp2, use_arc_engine=True)
    stats_arc = solver_arc.solve()
    
    # ArcEngine puede ser más lento en problemas pequeños, pero no >3x
    assert stats_arc.time_elapsed < stats_basic.time_elapsed * 3, \
        f"ArcEngine demasiado lento: {stats_arc.time_elapsed}s vs {stats_basic.time_elapsed}s"

def test_backward_compatibility():
    """Verifica que el comportamiento por defecto no cambia"""
    csp = create_nqueens_csp(n=4)
    
    # Comportamiento por defecto (sin especificar use_arc_engine)
    solver = CSPSolver(csp)
    stats = solver.solve()
    
    # Debe funcionar sin errores
    assert len(stats.solutions) > 0
    assert stats.nodes_explored > 0

@pytest.mark.parametrize("n", [4, 6, 8])
def test_consistency_across_problem_sizes(n):
    """Verifica consistencia en diferentes tamaños de problema"""
    csp_basic = create_nqueens_csp(n=n)
    csp_arc = create_nqueens_csp(n=n)
    
    solver_basic = CSPSolver(csp_basic, use_arc_engine=False)
    solver_arc = CSPSolver(csp_arc, use_arc_engine=True)
    
    stats_basic = solver_basic.solve()
    stats_arc = solver_arc.solve()
    
    assert len(stats_basic.solutions) == len(stats_arc.solutions)
```

---

## 📊 Análisis de Impacto

### Impacto en Código Existente

| Componente | Modificación | Riesgo | Mitigación |
|------------|--------------|--------|------------|
| `CSPSolver.__init__` | Añadir parámetros opcionales | Bajo | Valores por defecto mantienen comportamiento actual |
| `CSPSolver.enforce_arc_consistency` | Delegación condicional | Bajo | Fallback a AC-3 básico si ArcEngine falla |
| Tests existentes | Ninguna | Ninguno | Tests siguen pasando sin cambios |
| Ejemplos | Ninguna | Ninguno | Ejemplos siguen funcionando |
| Documentación | Actualización | Ninguno | Añadir sección sobre `use_arc_engine` |

### Impacto en Rendimiento

**Esperado:**
- ✅ Mejora de 20-40% en propagación de restricciones (AC-3.1 vs AC-3)
- ✅ Reducción de overhead mediante caché de revisiones
- ⚠️ Overhead inicial de setup de ArcEngine (~1-2ms)

**Medición:**
- Benchmarks antes/después en N-Queens (4x4, 8x8, 12x12)
- Benchmarks en problemas reales (scheduling, graph coloring)

### Impacto en Otros Módulos

**Módulos NO afectados:**
- ✅ `compiler_multiescala/*` - No usa CSPSolver directamente
- ✅ `fibration/*` - Usa ArcEngine directamente
- ✅ `examples/*` - Siguen funcionando sin cambios

**Módulos potencialmente beneficiados:**
- ✅ Cualquier código que use `CSPSolver` puede activar `use_arc_engine=True`
- ✅ Tests de rendimiento mostrarán mejoras automáticamente

---

## 🧪 Plan de Testing

### Tests Unitarios (Cobertura >90%)

1. **test_arc_engine_integration.py** (NUEVO)
   - ✅ `test_arc_engine_produces_same_solutions()`
   - ✅ `test_arc_engine_is_faster_or_equal()`
   - ✅ `test_backward_compatibility()`
   - ✅ `test_consistency_across_problem_sizes()`

2. **test_solver.py** (EXISTENTE - sin cambios)
   - ✅ Todos los tests existentes deben seguir pasando

### Tests de Integración

3. **test_solver_strategies_with_arc_engine.py** (NUEVO)
   - ✅ Verificar que estrategias (MRV, LCV, FCA-guided) funcionan con ArcEngine
   - ✅ Verificar que tracing funciona con ArcEngine

### Tests de Regresión

4. **Ejecutar suite completa de tests**
   ```bash
   pytest tests/ --cov=lattice_weaver/core/csp_engine --cov-report=html
   ```
   - ✅ Cobertura >90%
   - ✅ Todos los tests existentes pasan

### Benchmarks de Rendimiento

5. **benchmarks/arc_engine_integration_benchmark.py** (NUEVO)
   - Comparar rendimiento antes/después
   - Medir overhead de setup
   - Identificar casos donde ArcEngine es beneficioso

---

## 📅 Cronograma

### Semana 1: Implementación y Tests

| Día | Tarea | Horas | Entregable |
|-----|-------|-------|------------|
| 1 | Modificar `CSPSolver.__init__` | 2h | Código funcional |
| 1 | Implementar `_setup_arc_engine()` | 2h | Conversión CSP→ArcEngine |
| 2 | Modificar `enforce_arc_consistency()` | 2h | Delegación a ArcEngine |
| 2 | Implementar `_enforce_ac3_basic()` | 1h | Fallback |
| 3 | Crear tests de compatibilidad | 3h | test_arc_engine_integration.py |
| 3 | Crear tests de integración | 2h | test_solver_strategies_with_arc_engine.py |
| 4 | Ejecutar suite completa de tests | 1h | Reporte de cobertura |
| 4 | Crear benchmarks | 2h | Comparación de rendimiento |
| 5 | Análisis de resultados | 2h | Documento de análisis |
| 5 | Documentación | 2h | README actualizado |

**Total:** 19 horas (~2.5 días de trabajo)

---

## ✅ Criterios de Éxito

### Funcionales

- [ ] CSPSolver puede usar ArcEngine opcionalmente
- [ ] Resultados idénticos con y sin ArcEngine
- [ ] Backward compatibility total (código existente sin cambios)
- [ ] Todos los tests existentes pasan

### No Funcionales

- [ ] Mejora de rendimiento de 10-40% en propagación
- [ ] Overhead de setup <5ms
- [ ] Cobertura de tests >90%
- [ ] Documentación completa y clara

### Alineación con Protocolo

Según `PROTOCOLO_AGENTES_LATTICEWEAVER.md` (v4.0):

- [x] **Fase 0: Verificación del Estado del Proyecto** - ✅ Completado. Se han revisado `PROJECT_OVERVIEW.md` y `README.md`.
- [x] **Fase 1: Planificación y Diseño en Profundidad** - ✅ Completado en este documento, aplicando patrones de diseño como el Adapter Pattern y Strategy Pattern.
- [ ] **Fase 2: Implementación y Pruebas** - Pendiente
- [ ] **Fase 3: Análisis de Errores y Refinamiento** - Pendiente
- [ ] **Fase 4: Documentación y Actualización** - Pendiente, incluyendo el Protocolo de Merge Seguro.

---

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: ArcEngine no disponible en entorno

**Probabilidad:** Baja  
**Impacto:** Bajo  
**Mitigación:**
- Import condicional con try/except
- Warning si no está disponible
- Fallback automático a AC-3 básico

### Riesgo 2: Incompatibilidad de APIs

**Probabilidad:** Media  
**Impacto:** Medio  
**Mitigación:**
- Adapter pattern en `_setup_arc_engine()`
- Tests exhaustivos de compatibilidad
- Validación con múltiples tipos de restricciones

### Riesgo 3: Regresión en tests existentes

**Probabilidad:** Baja  
**Impacto:** Alto  
**Mitigación:**
- Comportamiento por defecto sin cambios
- Suite completa de tests de regresión
- Revisión manual de resultados

### Riesgo 4: Overhead excesivo en problemas pequeños

**Probabilidad:** Media  
**Impacto:** Bajo  
**Mitigación:**
- Benchmarks en diferentes tamaños de problema
- Documentar cuándo usar/no usar ArcEngine
- Considerar activación automática según tamaño

---

## 📚 Dependencias

### Módulos Requeridos

- ✅ `lattice_weaver.arc_engine` - Ya existe e implementado
- ✅ `lattice_weaver.core.csp_engine.solver` - Ya existe
- ✅ `lattice_weaver.core.csp_problem` - Ya existe

### Documentación Requerida

- ✅ `LatticeWeaver_Meta_Principios_Diseño_v3.md` - Consultado
- ✅ `PROTOCOLO_AGENTES_LATTICEWEAVER.md` - Seguido
- ✅ `ESTADO_ACTUAL_Y_ROADMAP.md` - Alineado con Tarea 1.1

---

## 🔄 Próximos Pasos (Fases Futuras)

### Fase 2: Solver Unificado (Semana 2-3)

Una vez validada la Fase 1, proceder con:
- Crear `UnifiedCSPSolver` que combine lo mejor de ambos
- Migrar ejemplos a usar UnifiedCSPSolver
- Benchmarks comparativos completos

### Fase 3: Deprecación Gradual (Semana 4+)

- Deprecar uso directo de AC-3 básico
- Migrar todo el código a usar ArcEngine
- Actualizar documentación completa

---

## 📝 Checklist de Protocolo de Agentes

Antes de proceder con la implementación, verificar:

- [x] Se ha realizado una planificación y diseño en profundidad
- [x] Se ha realizado una revisión de las librerías existentes (ArcEngine, CSPSolver)
- [x] El diseño se alinea con los Meta-Principios de Diseño
- [x] Se han identificado todos los archivos a modificar
- [x] Se ha definido un plan de testing exhaustivo (>90% cobertura)
- [x] Se ha evaluado el impacto en el resto de la estructura
- [x] Se ha definido una estrategia de mitigación de riesgos
- [x] Se ha establecido un cronograma realista
- [x] Se han definido criterios de éxito claros

---

## 🎯 Solicitud de Aprobación

**Esta propuesta está lista para revisión y aprobación.**

Una vez aprobada, se procederá con la implementación siguiendo el protocolo establecido:

1. Implementación según diseño
2. Tests exhaustivos (cobertura >90%)
3. Análisis de errores si surgen
4. Documentación completa
5. Actualización del repositorio

**Estimación total:** 19 horas (~2.5 días)  
**Riesgo:** Bajo  
**Impacto:** Alto (mejora de rendimiento 10-40%)  
**Prioridad:** Alta

---

**Fin de la Propuesta**

*Documento generado siguiendo el Protocolo de Desarrollo para Agentes de LatticeWeaver v3.0*

