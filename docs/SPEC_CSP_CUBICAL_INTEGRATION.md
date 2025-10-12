# Especificación de Diseño: Integración CSP ↔ Tipos Cúbicos

**Proyecto:** LatticeWeaver v5.0  
**Track:** Fase 1 - Integración Profunda  
**Autor:** Manus AI  
**Fecha:** 12 de Octubre, 2025  
**Duración Estimada:** 8 semanas  
**Prioridad:** CRÍTICA

---

## 1. Resumen Ejecutivo

Esta especificación detalla el diseño e implementación de la **integración profunda entre el motor CSP (arc_engine) y el sistema de tipos cúbicos (formal/cubical_*)**, uno de los gaps críticos identificados en el análisis del proyecto.

### 1.1. Motivación

Actualmente, `lattice-weaver` tiene dos componentes poderosos pero **desconectados**:

1. **Motor CSP** (`arc_engine/`) - Resuelve problemas de satisfacción de restricciones
2. **Sistema de Tipos Cúbicos** (`formal/cubical_*`) - Verificación formal con teoría de tipos homotópicos

La integración actual (`formal/csp_integration.py`) usa **tipos Sigma simples**, pero **NO aprovecha** la estructura cúbica para:

- Verificar **equivalencias** de soluciones mediante caminos
- Representar **simetrías** del espacio de soluciones
- Optimizar búsqueda usando **propiedades topológicas**
- Extraer **invariantes** formales del problema

### 1.2. Objetivos

**Objetivo Principal:**  
Crear un puente bidireccional completo entre CSP y tipos cúbicos que permita:

1. **Traducir** problemas CSP a tipos cúbicos
2. **Verificar** soluciones usando type checking cúbico
3. **Representar** equivalencias de soluciones como caminos
4. **Extraer** propiedades formales del espacio de soluciones
5. **Optimizar** búsqueda CSP usando información topológica

**Objetivos Secundarios:**

- Performance comparable o mejor que integración Sigma
- API clara y bien documentada
- Tests exhaustivos (20+ tests de integración)
- Ejemplos educativos completos

### 1.3. Alcance

**Dentro del alcance:**

- ✅ Traducción CSP → Tipos Cúbicos
- ✅ Verificación de soluciones
- ✅ Representación de equivalencias como caminos
- ✅ Extracción de propiedades topológicas
- ✅ Optimización de traducción
- ✅ Tests y documentación completos

**Fuera del alcance (futuro):**

- ❌ Síntesis automática de CSPs desde tipos
- ❌ Optimización de búsqueda CSP usando HoTT (Track D)
- ❌ Visualización de tipos cúbicos (Track I)
- ❌ Integración con GPU

---

## 2. Análisis del Estado Actual

### 2.1. Componentes Existentes

#### 2.1.1. Motor CSP (`arc_engine/`)

**Archivos clave:**
- `core.py` - Motor principal de CSP
- `domains.py` - Gestión de dominios
- `constraints.py` - Restricciones
- `ac31.py` - Algoritmo AC-3
- `csp_solver.py` - Solver completo

**Capacidades:**
- Propagación de restricciones (AC-3, AC-3.1)
- Búsqueda con backtracking
- Heurísticas de selección de variable
- Gestión de dominios incremental

**Limitaciones:**
- No hay representación formal de equivalencias
- No se aprovechan simetrías
- No hay extracción de invariantes

#### 2.1.2. Sistema de Tipos Cúbicos (`formal/`)

**Archivos clave:**
- `cubical_syntax.py` - Sintaxis de tipos cúbicos
- `cubical_operations.py` - Operaciones (composición, transporte)
- `cubical_engine.py` - Type checker cúbico

**Capacidades:**
- Representación de cubos, caminos, identidades
- Type checking de términos cúbicos
- Evaluación y normalización básica
- Composición de caminos

**Limitaciones:**
- No está conectado con CSP
- No hay optimizaciones de performance
- Normalización no es completa

#### 2.1.3. Integración Actual (`formal/csp_integration.py`)

**Capacidades:**
- Traducción CSP → Tipos Sigma
- Conversión Solución → Prueba
- Verificación básica

**Limitaciones:**
- **NO usa tipos cúbicos** - Solo tipos Sigma simples
- No representa equivalencias
- No aprovecha estructura topológica

### 2.2. Gap Identificado

**El problema central:** Existe un sistema de tipos cúbicos completo pero **no se usa** para CSP.

**Consecuencias:**

1. **Pérdida de expresividad** - No podemos hablar de equivalencias de soluciones
2. **Pérdida de optimización** - No aprovechamos propiedades topológicas
3. **Duplicación** - Dos sistemas separados que deberían estar unidos
4. **Complejidad innecesaria** - Dos APIs diferentes para verificación

---

## 3. Diseño de la Solución

### 3.1. Arquitectura General

```
┌─────────────────────────────────────────────────────────────┐
│                     CSP Problem                              │
│  Variables: {X, Y, Z}                                        │
│  Domains: {X: {1,2,3}, Y: {1,2,3}, Z: {1,2,3}}             │
│  Constraints: {X < Y, Y < Z}                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ CSPToCubicalBridge.translate()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Cubical Type                                │
│  SolutionSpace : Type                                        │
│  SolutionSpace = Σ (x : Domain X)                           │
│                   Σ (y : Domain Y)                           │
│                   Σ (z : Domain Z)                           │
│                   (x < y) × (y < z)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ solve() + verify()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Solution + Proof                            │
│  solution : SolutionSpace                                    │
│  solution = (1, 2, 3, proof_x_lt_y, proof_y_lt_z)           │
│                                                              │
│  equivalence : Path SolutionSpace sol1 sol2                  │
│  (si hay simetrías)                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ extract_properties()
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Topological Properties                          │
│  - Número de componentes conexas                            │
│  - Simetrías (grupo fundamental)                            │
│  - Invariantes homotópicos                                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2. Componentes Principales

#### 3.2.1. `CSPToCubicalBridge` (Nuevo)

**Responsabilidad:** Traducir problemas CSP a tipos cúbicos.

**API:**

```python
class CSPToCubicalBridge:
    """
    Puente entre CSP y tipos cúbicos.
    
    Traduce problemas CSP a tipos cúbicos para verificación formal
    y análisis topológico.
    """
    
    def __init__(self, csp_problem: CSPProblem, cubical_engine: CubicalEngine):
        """
        Inicializa el bridge.
        
        Args:
            csp_problem: Problema CSP a traducir
            cubical_engine: Motor de tipos cúbicos
        """
        ...
    
    def translate_to_cubical_type(self) -> CubicalType:
        """
        Traduce el CSP a un tipo cúbico.
        
        Returns:
            Tipo cúbico representando el espacio de soluciones
            
        Example:
            CSP: X < Y, dominios {1,2,3}
            →
            Σ (x : {1,2,3}) Σ (y : {1,2,3}) (x < y)
        """
        ...
    
    def solution_to_term(self, solution: Dict[str, Any]) -> CubicalTerm:
        """
        Convierte una solución CSP a un término cúbico.
        
        Args:
            solution: Solución del CSP
            
        Returns:
            Término cúbico habitando el tipo de soluciones
        """
        ...
    
    def verify_solution(self, solution: Dict[str, Any]) -> bool:
        """
        Verifica una solución usando type checking cúbico.
        
        Args:
            solution: Solución a verificar
            
        Returns:
            True si la solución type-checks correctamente
        """
        ...
    
    def find_equivalences(
        self, 
        sol1: Dict[str, Any], 
        sol2: Dict[str, Any]
    ) -> Optional[CubicalPath]:
        """
        Encuentra un camino entre dos soluciones (si son equivalentes).
        
        Args:
            sol1, sol2: Soluciones a comparar
            
        Returns:
            Camino cúbico si son equivalentes, None en caso contrario
            
        Example:
            sol1 = {X: 1, Y: 2}
            sol2 = {X: 1, Y: 2}  # Misma solución
            → Path trivial (refl)
            
            sol1 = {X: 1, Y: 2}
            sol2 = {X: 2, Y: 1}  # Diferentes pero equivalentes por simetría
            → Path no trivial (si hay simetría)
        """
        ...
    
    def extract_symmetries(self) -> List[CubicalPath]:
        """
        Extrae simetrías del espacio de soluciones.
        
        Returns:
            Lista de caminos representando simetrías
            
        Example:
            CSP: X != Y, dominios {1,2}
            Soluciones: {X:1, Y:2} y {X:2, Y:1}
            → Simetría: swap(X, Y)
        """
        ...
    
    def compute_fundamental_group(self) -> Group:
        """
        Calcula el grupo fundamental del espacio de soluciones.
        
        Returns:
            Grupo fundamental (π₁)
            
        Notes:
            - Útil para detectar "agujeros" en el espacio de soluciones
            - Puede guiar la búsqueda CSP
        """
        ...
```

#### 3.2.2. `CubicalCSPType` (Nuevo)

**Responsabilidad:** Representar tipos cúbicos derivados de CSPs.

**API:**

```python
@dataclass
class CubicalCSPType:
    """
    Tipo cúbico representando un espacio de soluciones CSP.
    
    Attributes:
        variables: Variables del CSP
        domain_types: Tipos de los dominios
        constraint_props: Proposiciones de las restricciones
        solution_type: Tipo Sigma completo
    """
    
    variables: List[str]
    domain_types: Dict[str, CubicalType]
    constraint_props: List[CubicalProp]
    solution_type: CubicalType
    
    def __init__(self, csp_problem: CSPProblem):
        """
        Construye el tipo desde un CSP.
        
        Args:
            csp_problem: Problema CSP
        """
        ...
    
    def check_term(self, term: CubicalTerm) -> bool:
        """
        Verifica si un término habita este tipo.
        
        Args:
            term: Término a verificar
            
        Returns:
            True si term : solution_type
        """
        ...
    
    def synthesize_term(self, solution: Dict[str, Any]) -> CubicalTerm:
        """
        Sintetiza un término desde una solución.
        
        Args:
            solution: Solución CSP
            
        Returns:
            Término cúbico correspondiente
        """
        ...
```

#### 3.2.3. `PathFinder` (Nuevo)

**Responsabilidad:** Encontrar caminos entre soluciones.

**API:**

```python
class PathFinder:
    """
    Encuentra caminos cúbicos entre soluciones CSP.
    
    Usa algoritmos de búsqueda en el espacio de tipos cúbicos
    para encontrar equivalencias.
    """
    
    def __init__(self, cubical_type: CubicalCSPType):
        """
        Inicializa el buscador de caminos.
        
        Args:
            cubical_type: Tipo cúbico del espacio de soluciones
        """
        ...
    
    def find_path(
        self, 
        term1: CubicalTerm, 
        term2: CubicalTerm,
        max_depth: int = 10
    ) -> Optional[CubicalPath]:
        """
        Busca un camino entre dos términos.
        
        Args:
            term1, term2: Términos a conectar
            max_depth: Profundidad máxima de búsqueda
            
        Returns:
            Camino si existe, None en caso contrario
            
        Algorithm:
            1. Intentar camino trivial (refl) si term1 = term2
            2. Buscar caminos usando simetrías conocidas
            3. Búsqueda en profundidad limitada
            4. Usar heurísticas topológicas
        """
        ...
    
    def find_all_paths(
        self, 
        term1: CubicalTerm, 
        term2: CubicalTerm
    ) -> List[CubicalPath]:
        """
        Encuentra todos los caminos entre dos términos.
        
        Args:
            term1, term2: Términos a conectar
            
        Returns:
            Lista de todos los caminos
            
        Notes:
            - Puede ser costoso computacionalmente
            - Útil para análisis de simetrías
        """
        ...
    
    def compute_homotopy_class(self, path: CubicalPath) -> int:
        """
        Calcula la clase de homotopía de un camino.
        
        Args:
            path: Camino a clasificar
            
        Returns:
            Clase de homotopía (elemento de π₁)
        """
        ...
```

#### 3.2.4. `SymmetryExtractor` (Nuevo)

**Responsabilidad:** Extraer simetrías del espacio de soluciones.

**API:**

```python
class SymmetryExtractor:
    """
    Extrae simetrías de problemas CSP usando tipos cúbicos.
    
    Las simetrías son automorfismos del espacio de soluciones,
    representados como caminos cúbicos.
    """
    
    def __init__(self, bridge: CSPToCubicalBridge):
        """
        Inicializa el extractor.
        
        Args:
            bridge: Bridge CSP-Cubical
        """
        ...
    
    def extract_variable_symmetries(self) -> List[Symmetry]:
        """
        Extrae simetrías de variables.
        
        Returns:
            Lista de simetrías (permutaciones de variables)
            
        Example:
            CSP: X != Y, dominios idénticos
            → Simetría: swap(X, Y)
        """
        ...
    
    def extract_value_symmetries(self) -> List[Symmetry]:
        """
        Extrae simetrías de valores.
        
        Returns:
            Lista de simetrías (permutaciones de valores)
            
        Example:
            CSP: X < Y, dominio {1,2,3}
            → No hay simetrías de valores (< rompe simetría)
        """
        ...
    
    def build_symmetry_group(self) -> Group:
        """
        Construye el grupo de simetrías completo.
        
        Returns:
            Grupo de simetrías
            
        Notes:
            - Producto de simetrías de variables y valores
            - Útil para breaking symmetries en búsqueda
        """
        ...
    
    def apply_symmetry(
        self, 
        solution: Dict[str, Any], 
        symmetry: Symmetry
    ) -> Dict[str, Any]:
        """
        Aplica una simetría a una solución.
        
        Args:
            solution: Solución original
            symmetry: Simetría a aplicar
            
        Returns:
            Solución transformada
        """
        ...
```

### 3.3. Flujos de Datos

#### 3.3.1. Flujo de Traducción

```
CSPProblem
    │
    │ CSPToCubicalBridge.translate_to_cubical_type()
    ▼
CubicalCSPType
    │
    │ Para cada variable X:
    │   - Crear tipo Domain_X
    │   - Agregar a tipo Sigma
    │
    │ Para cada restricción C:
    │   - Traducir a proposición cúbica
    │   - Agregar como componente del Sigma
    │
    ▼
Σ (x1 : D1) Σ (x2 : D2) ... Σ (xn : Dn) (C1 × C2 × ... × Cm)
```

#### 3.3.2. Flujo de Verificación

```
Solution (Dict)
    │
    │ CSPToCubicalBridge.solution_to_term()
    ▼
CubicalTerm
    │
    │ CubicalEngine.type_check()
    ▼
TypeCheckResult
    │
    ├─ Success → Solución válida
    └─ Failure → Solución inválida (con razón)
```

#### 3.3.3. Flujo de Equivalencia

```
Solution1, Solution2
    │
    │ CSPToCubicalBridge.solution_to_term()
    ▼
Term1, Term2
    │
    │ PathFinder.find_path()
    ▼
Path (si existe)
    │
    ├─ Trivial (refl) → Soluciones idénticas
    ├─ No trivial → Soluciones equivalentes por simetría
    └─ None → Soluciones no equivalentes
```

---

## 4. Especificación de Implementación

### 4.1. Estructura de Archivos

```
lattice_weaver/
├── formal/
│   ├── csp_cubical_bridge.py          (NUEVO - 600 líneas)
│   ├── cubical_csp_type.py            (NUEVO - 400 líneas)
│   ├── path_finder.py                 (NUEVO - 500 líneas)
│   ├── symmetry_extractor.py          (NUEVO - 400 líneas)
│   ├── cubical_syntax.py              (MODIFICAR - añadir helpers)
│   ├── cubical_operations.py          (MODIFICAR - optimizar)
│   └── cubical_engine.py              (MODIFICAR - añadir caching)
│
├── tests/
│   ├── unit/
│   │   ├── test_csp_cubical_bridge.py     (NUEVO - 300 líneas)
│   │   ├── test_cubical_csp_type.py       (NUEVO - 200 líneas)
│   │   ├── test_path_finder.py            (NUEVO - 250 líneas)
│   │   └── test_symmetry_extractor.py     (NUEVO - 200 líneas)
│   │
│   └── integration/
│       ├── test_csp_cubical_integration.py (NUEVO - 400 líneas)
│       └── test_csp_cubical_performance.py (NUEVO - 200 líneas)
│
├── examples/
│   ├── csp_cubical_basic.py           (NUEVO - 150 líneas)
│   ├── csp_cubical_symmetries.py      (NUEVO - 200 líneas)
│   └── csp_cubical_verification.py    (NUEVO - 250 líneas)
│
└── docs/
    ├── SPEC_CSP_CUBICAL_INTEGRATION.md (ESTE DOCUMENTO)
    ├── API_CSP_CUBICAL.md              (NUEVO - 50 páginas)
    └── TUTORIAL_CSP_CUBICAL.md         (NUEVO - 30 páginas)
```

**Total estimado:** ~4,000 líneas de código nuevo + modificaciones

### 4.2. Hitos de Implementación

#### Hito 1: Fundamentos (Semanas 1-2)

**Tareas:**

1. Diseñar y documentar API completa
2. Implementar `CubicalCSPType` básico
3. Implementar traducción simple CSP → Tipo Sigma cúbico
4. Tests unitarios básicos

**Entregables:**

- `cubical_csp_type.py` funcional
- 10 tests unitarios pasando
- Documentación de API

**Criterios de aceptación:**

- ✅ Traducción correcta de CSPs simples (2-3 variables)
- ✅ Type checking básico funcional
- ✅ Documentación completa de API

#### Hito 2: Bridge Completo (Semanas 3-4)

**Tareas:**

1. Implementar `CSPToCubicalBridge` completo
2. Soportar todos los tipos de restricciones
3. Optimizar traducción
4. Tests de integración

**Entregables:**

- `csp_cubical_bridge.py` completo
- 15 tests de integración pasando
- Benchmarks de performance

**Criterios de aceptación:**

- ✅ Traducción correcta de CSPs complejos (10+ variables)
- ✅ Soporte para restricciones arbitrarias
- ✅ Performance aceptable (< 1s para CSPs medianos)

#### Hito 3: Caminos y Equivalencias (Semanas 5-6)

**Tareas:**

1. Implementar `PathFinder`
2. Algoritmos de búsqueda de caminos
3. Optimización de búsqueda
4. Tests exhaustivos

**Entregables:**

- `path_finder.py` completo
- 20 tests de caminos
- Análisis de complejidad

**Criterios de aceptación:**

- ✅ Encuentra caminos triviales correctamente
- ✅ Encuentra caminos no triviales (simetrías)
- ✅ Performance razonable (< 5s para búsquedas simples)

#### Hito 4: Simetrías (Semanas 7-8)

**Tareas:**

1. Implementar `SymmetryExtractor`
2. Algoritmos de detección de simetrías
3. Construcción de grupo de simetrías
4. Tests y ejemplos

**Entregables:**

- `symmetry_extractor.py` completo
- 15 tests de simetrías
- 3 ejemplos completos

**Criterios de aceptación:**

- ✅ Detecta simetrías de variables correctamente
- ✅ Detecta simetrías de valores correctamente
- ✅ Construye grupo de simetrías correctamente

### 4.3. Plan de Tests

#### Tests Unitarios (40 tests)

**`test_cubical_csp_type.py` (10 tests):**

1. `test_construct_from_simple_csp` - CSP simple → Tipo
2. `test_construct_from_complex_csp` - CSP complejo → Tipo
3. `test_check_valid_term` - Verificar término válido
4. `test_check_invalid_term` - Rechazar término inválido
5. `test_synthesize_term_from_solution` - Solución → Término
6. `test_domain_types_correct` - Tipos de dominios correctos
7. `test_constraint_props_correct` - Proposiciones correctas
8. `test_solution_type_structure` - Estructura de tipo Sigma
9. `test_equality_of_types` - Igualdad de tipos
10. `test_subtyping` - Relación de subtipado

**`test_csp_cubical_bridge.py` (15 tests):**

1. `test_translate_simple_csp` - Traducción simple
2. `test_translate_complex_csp` - Traducción compleja
3. `test_translate_all_constraint_types` - Todos los tipos de restricciones
4. `test_solution_to_term_valid` - Solución válida → Término
5. `test_solution_to_term_invalid` - Solución inválida → Error
6. `test_verify_solution_valid` - Verificar solución válida
7. `test_verify_solution_invalid` - Rechazar solución inválida
8. `test_find_trivial_equivalence` - Equivalencia trivial (refl)
9. `test_find_symmetry_equivalence` - Equivalencia por simetría
10. `test_no_equivalence` - Sin equivalencia
11. `test_extract_symmetries_simple` - Simetrías simples
12. `test_extract_symmetries_complex` - Simetrías complejas
13. `test_compute_fundamental_group` - Grupo fundamental
14. `test_caching_optimization` - Optimización con caché
15. `test_error_handling` - Manejo de errores

**`test_path_finder.py` (10 tests):**

1. `test_find_trivial_path` - Camino trivial
2. `test_find_simple_path` - Camino simple
3. `test_find_complex_path` - Camino complejo
4. `test_no_path_exists` - Sin camino
5. `test_find_all_paths` - Todos los caminos
6. `test_homotopy_class` - Clase de homotopía
7. `test_path_composition` - Composición de caminos
8. `test_path_inversion` - Inversión de caminos
9. `test_max_depth_limit` - Límite de profundidad
10. `test_performance_large_space` - Performance en espacios grandes

**`test_symmetry_extractor.py` (5 tests):**

1. `test_extract_variable_symmetries` - Simetrías de variables
2. `test_extract_value_symmetries` - Simetrías de valores
3. `test_build_symmetry_group` - Grupo de simetrías
4. `test_apply_symmetry` - Aplicar simetría
5. `test_symmetry_breaking` - Breaking de simetrías

#### Tests de Integración (20 tests)

**`test_csp_cubical_integration.py` (15 tests):**

1. `test_end_to_end_simple_csp` - Flujo completo simple
2. `test_end_to_end_nqueens` - N-Queens
3. `test_end_to_end_graph_coloring` - Coloreo de grafos
4. `test_end_to_end_sudoku` - Sudoku
5. `test_verification_pipeline` - Pipeline de verificación
6. `test_equivalence_detection` - Detección de equivalencias
7. `test_symmetry_exploitation` - Explotación de simetrías
8. `test_multiple_solutions` - Múltiples soluciones
9. `test_no_solution` - Sin solución
10. `test_partial_solution` - Solución parcial
11. `test_incremental_solving` - Resolución incremental
12. `test_backtracking_with_verification` - Backtracking + verificación
13. `test_heuristics_with_topology` - Heurísticas topológicas
14. `test_large_problem` - Problema grande (50+ variables)
15. `test_stress_test` - Test de estrés

**`test_csp_cubical_performance.py` (5 tests):**

1. `test_translation_performance` - Performance de traducción
2. `test_verification_performance` - Performance de verificación
3. `test_path_finding_performance` - Performance de búsqueda de caminos
4. `test_symmetry_extraction_performance` - Performance de extracción de simetrías
5. `test_scalability` - Escalabilidad

### 4.4. Optimizaciones

#### 4.4.1. Caching

**Niveles de caché:**

1. **Caché de traducción** - CSP → Tipo cúbico
2. **Caché de type checking** - Término → Resultado
3. **Caché de caminos** - (Term1, Term2) → Path
4. **Caché de simetrías** - CSP → Grupo de simetrías

**Implementación:**

```python
from functools import lru_cache
from typing import Dict, Any

class CachedCSPToCubicalBridge:
    """Bridge con caching multinivel."""
    
    def __init__(self, ...):
        self._translation_cache: Dict[int, CubicalType] = {}
        self._typecheck_cache: Dict[int, bool] = {}
        self._path_cache: Dict[Tuple[int, int], Optional[CubicalPath]] = {}
        self._symmetry_cache: Optional[Group] = None
    
    @lru_cache(maxsize=1000)
    def translate_to_cubical_type(self) -> CubicalType:
        # Usar hash del CSP como clave
        csp_hash = hash(self.csp_problem)
        if csp_hash in self._translation_cache:
            return self._translation_cache[csp_hash]
        
        # Traducir y cachear
        cubical_type = self._do_translation()
        self._translation_cache[csp_hash] = cubical_type
        return cubical_type
```

#### 4.4.2. Lazy Evaluation

**Estrategia:**

- No construir todo el tipo cúbico de una vez
- Construir componentes bajo demanda
- Usar generadores para listas grandes

**Implementación:**

```python
class LazyC ubicalCSPType:
    """Tipo cúbico con evaluación lazy."""
    
    def __init__(self, csp_problem: CSPProblem):
        self.csp_problem = csp_problem
        self._domain_types = None  # Lazy
        self._constraint_props = None  # Lazy
        self._solution_type = None  # Lazy
    
    @property
    def domain_types(self) -> Dict[str, CubicalType]:
        if self._domain_types is None:
            self._domain_types = self._build_domain_types()
        return self._domain_types
    
    # Similar para otras propiedades
```

#### 4.4.3. Optimización de Búsqueda de Caminos

**Heurísticas:**

1. **Distancia de Hamming** - Priorizar términos más cercanos
2. **Simetrías conocidas** - Usar simetrías como atajos
3. **Poda temprana** - Podar ramas imposibles
4. **Búsqueda bidireccional** - Buscar desde ambos extremos

**Implementación:**

```python
def find_path_optimized(
    self, 
    term1: CubicalTerm, 
    term2: CubicalTerm
) -> Optional[CubicalPath]:
    # 1. Intentar camino trivial
    if term1 == term2:
        return CubicalPath.refl(term1)
    
    # 2. Intentar simetrías conocidas
    for symmetry in self.known_symmetries:
        transformed = symmetry.apply(term1)
        if transformed == term2:
            return symmetry.to_path()
    
    # 3. Búsqueda con heurística
    return self._a_star_search(term1, term2, heuristic=hamming_distance)
```

---

## 5. Documentación

### 5.1. API Reference

**Documento:** `docs/API_CSP_CUBICAL.md` (50 páginas)

**Contenido:**

1. Introducción y conceptos
2. API de `CSPToCubicalBridge`
3. API de `CubicalCSPType`
4. API de `PathFinder`
5. API de `SymmetryExtractor`
6. Ejemplos de uso
7. Guía de troubleshooting
8. FAQ

### 5.2. Tutorial

**Documento:** `docs/TUTORIAL_CSP_CUBICAL.md` (30 páginas)

**Contenido:**

1. **Introducción** (5 páginas)
   - ¿Qué es la integración CSP-Cubical?
   - ¿Por qué es útil?
   - Conceptos previos necesarios

2. **Primeros Pasos** (10 páginas)
   - Instalación y setup
   - Primer ejemplo: CSP simple
   - Traducción a tipo cúbico
   - Verificación de solución

3. **Equivalencias y Caminos** (8 páginas)
   - Encontrar equivalencias
   - Interpretar caminos
   - Usar simetrías

4. **Casos de Uso Avanzados** (7 páginas)
   - N-Queens con simetrías
   - Coloreo de grafos
   - Sudoku
   - Problemas personalizados

### 5.3. Ejemplos

#### Ejemplo 1: CSP Básico

**Archivo:** `examples/csp_cubical_basic.py`

```python
"""
Ejemplo básico de integración CSP-Cubical.

Demuestra:
- Traducción de CSP a tipo cúbico
- Verificación de solución
- Detección de equivalencias
"""

from lattice_weaver.arc_engine.core import CSPProblem
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge
from lattice_weaver.formal.cubical_engine import CubicalEngine

# 1. Definir CSP simple: X < Y, dominios {1, 2, 3}
csp = CSPProblem(
    variables=['X', 'Y'],
    domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}},
    constraints=[('X', '<', 'Y')]
)

# 2. Crear bridge
engine = CubicalEngine()
bridge = CSPToCubicalBridge(csp, engine)

# 3. Traducir a tipo cúbico
cubical_type = bridge.translate_to_cubical_type()
print(f"Tipo cúbico: {cubical_type}")
# Output: Σ (x : {1,2,3}) Σ (y : {1,2,3}) (x < y)

# 4. Verificar solución válida
solution1 = {'X': 1, 'Y': 2}
is_valid = bridge.verify_solution(solution1)
print(f"Solución {solution1} válida: {is_valid}")
# Output: True

# 5. Verificar solución inválida
solution2 = {'X': 2, 'Y': 1}
is_valid = bridge.verify_solution(solution2)
print(f"Solución {solution2} válida: {is_valid}")
# Output: False

# 6. Encontrar equivalencias
solution3 = {'X': 1, 'Y': 2}  # Misma que solution1
path = bridge.find_equivalences(solution1, solution3)
print(f"Camino entre {solution1} y {solution3}: {path}")
# Output: Path.refl (camino trivial)
```

#### Ejemplo 2: Simetrías

**Archivo:** `examples/csp_cubical_symmetries.py`

```python
"""
Ejemplo de detección y explotación de simetrías.

Demuestra:
- Extracción de simetrías
- Construcción de grupo de simetrías
- Aplicación de simetrías
"""

from lattice_weaver.arc_engine.core import CSPProblem
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge
from lattice_weaver.formal.symmetry_extractor import SymmetryExtractor

# 1. CSP con simetría: X != Y, dominios {1, 2}
csp = CSPProblem(
    variables=['X', 'Y'],
    domains={'X': {1, 2}, 'Y': {1, 2}},
    constraints=[('X', '!=', 'Y')]
)

# 2. Crear extractor de simetrías
bridge = CSPToCubicalBridge(csp, ...)
extractor = SymmetryExtractor(bridge)

# 3. Extraer simetrías
symmetries = extractor.extract_variable_symmetries()
print(f"Simetrías encontradas: {symmetries}")
# Output: [swap(X, Y)]

# 4. Construir grupo
group = extractor.build_symmetry_group()
print(f"Grupo de simetrías: {group}")
# Output: Z_2 (grupo cíclico de orden 2)

# 5. Aplicar simetría
solution = {'X': 1, 'Y': 2}
transformed = extractor.apply_symmetry(solution, symmetries[0])
print(f"Solución transformada: {transformed}")
# Output: {'X': 2, 'Y': 1}

# 6. Verificar que son equivalentes
path = bridge.find_equivalences(solution, transformed)
print(f"Camino: {path}")
# Output: Path no trivial (simetría)
```

---

## 6. Métricas de Éxito

### 6.1. Métricas Funcionales

- ✅ **100% de tests pasando** (60 tests totales)
- ✅ **Traducción correcta** de todos los tipos de CSP
- ✅ **Verificación correcta** de soluciones
- ✅ **Detección correcta** de equivalencias
- ✅ **Extracción correcta** de simetrías

### 6.2. Métricas de Performance

- ✅ **Traducción:** < 1s para CSPs medianos (10-20 variables)
- ✅ **Verificación:** < 100ms por solución
- ✅ **Búsqueda de caminos:** < 5s para búsquedas simples
- ✅ **Extracción de simetrías:** < 2s para CSPs medianos

### 6.3. Métricas de Calidad

- ✅ **Cobertura de código:** > 90%
- ✅ **Documentación:** 100% de API documentada
- ✅ **Ejemplos:** 3+ ejemplos completos
- ✅ **Tutorial:** 30+ páginas

### 6.4. Métricas de Integración

- ✅ **Compatibilidad:** 100% compatible con código existente
- ✅ **Sin regresiones:** Todos los tests existentes siguen pasando
- ✅ **API coherente:** Sigue convenciones del proyecto

---

## 7. Riesgos y Mitigación

### 7.1. Riesgos Técnicos

**Riesgo 1: Complejidad de la traducción**

- **Probabilidad:** Alta
- **Impacto:** Alto
- **Mitigación:**
  - Prototipo rápido con CSPs simples
  - Revisión de diseño con expertos
  - Tests exhaustivos incrementales

**Riesgo 2: Performance insuficiente**

- **Probabilidad:** Media
- **Impacto:** Alto
- **Mitigación:**
  - Benchmarking temprano
  - Optimizaciones desde el inicio (caching, lazy eval)
  - Profiling continuo

**Riesgo 3: Búsqueda de caminos intratable**

- **Probabilidad:** Media
- **Impacto:** Medio
- **Mitigación:**
  - Límites de profundidad configurables
  - Heurísticas de poda
  - Fallback a búsqueda simple

### 7.2. Riesgos de Integración

**Riesgo 4: Incompatibilidad con código existente**

- **Probabilidad:** Baja
- **Impacto:** Alto
- **Mitigación:**
  - Tests de regresión exhaustivos
  - Revisión de API con equipo
  - Integración incremental

**Riesgo 5: Cambios en dependencias**

- **Probabilidad:** Baja
- **Impacto:** Medio
- **Mitigación:**
  - Versiones fijas de dependencias
  - Tests de compatibilidad
  - Documentación de requisitos

---

## 8. Cronograma Detallado

### Semana 1: Diseño y Fundamentos

**Lunes-Martes:**
- Revisar especificación completa
- Diseñar API detallada
- Crear estructura de archivos

**Miércoles-Jueves:**
- Implementar `CubicalCSPType` básico
- Tests unitarios básicos (5 tests)

**Viernes:**
- Revisión de código
- Documentación de API inicial

**Entregables:**
- `cubical_csp_type.py` (200 líneas)
- 5 tests pasando
- Documentación de API (10 páginas)

### Semana 2: Traducción Básica

**Lunes-Martes:**
- Implementar traducción simple CSP → Tipo
- Soporte para restricciones básicas (<, >, =, !=)

**Miércoles-Jueves:**
- Tests de traducción (10 tests)
- Optimización de traducción

**Viernes:**
- Benchmarking inicial
- Documentación

**Entregables:**
- Traducción funcional
- 10 tests adicionales
- Benchmarks baseline

### Semana 3: Bridge Completo

**Lunes-Martes:**
- Implementar `CSPToCubicalBridge`
- Método `solution_to_term()`

**Miércoles-Jueves:**
- Método `verify_solution()`
- Tests de verificación (10 tests)

**Viernes:**
- Integración con `CubicalEngine`
- Optimizaciones de performance

**Entregables:**
- `csp_cubical_bridge.py` (400 líneas)
- 10 tests de verificación
- Performance mejorada

### Semana 4: Restricciones Avanzadas

**Lunes-Martes:**
- Soporte para restricciones arbitrarias
- Restricciones globales (alldiff, etc.)

**Miércoles-Jueves:**
- Tests de restricciones complejas (10 tests)
- Optimización de traducción

**Viernes:**
- Tests de integración (5 tests)
- Documentación

**Entregables:**
- Soporte completo de restricciones
- 15 tests adicionales
- Documentación actualizada

### Semana 5: Búsqueda de Caminos

**Lunes-Martes:**
- Implementar `PathFinder`
- Algoritmo de búsqueda básico

**Miércoles-Jueves:**
- Optimización con heurísticas
- Caching de caminos

**Viernes:**
- Tests de caminos (10 tests)
- Benchmarking

**Entregables:**
- `path_finder.py` (400 líneas)
- 10 tests de caminos
- Análisis de performance

### Semana 6: Equivalencias

**Lunes-Martes:**
- Método `find_equivalences()`
- Detección de caminos triviales

**Miércoles-Jueves:**
- Detección de caminos no triviales
- Tests de equivalencias (10 tests)

**Viernes:**
- Optimización
- Documentación

**Entregables:**
- Detección de equivalencias funcional
- 10 tests adicionales
- Tutorial de equivalencias

### Semana 7: Simetrías

**Lunes-Martes:**
- Implementar `SymmetryExtractor`
- Extracción de simetrías de variables

**Miércoles-Jueves:**
- Extracción de simetrías de valores
- Construcción de grupo

**Viernes:**
- Tests de simetrías (10 tests)
- Ejemplos

**Entregables:**
- `symmetry_extractor.py` (350 líneas)
- 10 tests de simetrías
- Ejemplo de simetrías

### Semana 8: Finalización

**Lunes-Martes:**
- Tests de integración completos (15 tests)
- Tests de performance (5 tests)

**Miércoles-Jueves:**
- Documentación completa (API + Tutorial)
- Ejemplos completos (3 ejemplos)

**Viernes:**
- Revisión final
- Release

**Entregables:**
- Suite completa de tests (60 tests)
- Documentación completa (80 páginas)
- 3 ejemplos completos
- Release v5.1

---

## 9. Conclusión

Esta especificación detalla un plan completo y realista para implementar la **integración profunda CSP ↔ Tipos Cúbicos**, uno de los gaps críticos de `lattice-weaver`.

**Beneficios esperados:**

1. **Verificación formal avanzada** - Usar type checking cúbico para verificar soluciones
2. **Detección de equivalencias** - Representar equivalencias como caminos
3. **Explotación de simetrías** - Usar simetrías para optimizar búsqueda
4. **Análisis topológico** - Extraer propiedades topológicas del espacio de soluciones
5. **Base para futuras extensiones** - Prerequisito para Track D (Inference Engine)

**Próximos pasos:**

1. Revisión y aprobación de esta especificación
2. Inicio de implementación (Semana 1)
3. Revisiones semanales de progreso
4. Release en 8 semanas

**LatticeWeaver v5.1 con integración CSP-Cubical será un hito fundamental hacia el framework universal de modelado formal.**

---

**Fin de la Especificación**

