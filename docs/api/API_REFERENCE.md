# API Reference - LatticeWeaver v5.0

**Versión:** 5.0.0  
**Fecha:** 12 de Octubre, 2025

---

## Tabla de Contenidos

1. [CSP Engine API](#csp-engine-api)
2. [FCA Engine API](#fca-engine-api)
3. [TDA Engine API](#tda-engine-api)
4. [Visualization API](#visualization-api)
5. [Inference API](#inference-api)

---

## CSP Engine API

### AdaptiveConsistencyEngine

Motor principal para resolver problemas de satisfacción de restricciones.

#### Constructor

```python
from lattice_weaver.arc_engine import AdaptiveConsistencyEngine

engine = AdaptiveConsistencyEngine(
    algorithm='auto',        # 'ac3', 'ac31', 'parallel', 'auto'
    optimization_level=2,    # 0-3 (mayor = más optimizaciones)
    parallel=True,           # Habilitar paralelización
    num_workers=None,        # None = auto-detect cores
    cache_size=10000,        # Tamaño del cache LRU
    timeout=None             # Timeout en segundos (None = sin límite)
)
```

**Parámetros:**
- `algorithm` (str): Algoritmo de consistencia a usar
  - `'ac3'`: AC-3 clásico
  - `'ac31'`: AC-3.1 optimizado
  - `'parallel'`: AC-3 paralelo
  - `'auto'`: Selección automática según problema
- `optimization_level` (int): Nivel de optimizaciones (0-3)
  - `0`: Sin optimizaciones
  - `1`: Optimizaciones básicas (variable ordering)
  - `2`: Optimizaciones avanzadas (constraint propagation)
  - `3`: Todas las optimizaciones (puede ser más lento en problemas pequeños)
- `parallel` (bool): Habilitar ejecución paralela
- `num_workers` (int|None): Número de workers (None = número de cores)
- `cache_size` (int): Tamaño del cache para resultados intermedios
- `timeout` (float|None): Timeout máximo de ejecución

#### Métodos Principales

##### add_variable

Añade una variable al problema CSP.

```python
engine.add_variable(
    name: str,
    domain: List[Any],
    initial_value: Any = None
) -> None
```

**Parámetros:**
- `name` (str): Nombre único de la variable
- `domain` (List[Any]): Lista de valores posibles
- `initial_value` (Any, opcional): Valor inicial (debe estar en domain)

**Ejemplo:**
```python
engine.add_variable("x", [1, 2, 3, 4, 5])
engine.add_variable("y", ["red", "green", "blue"])
engine.add_variable("z", [0.1, 0.5, 1.0, 1.5], initial_value=0.5)
```

**Excepciones:**
- `ValueError`: Si el nombre ya existe o initial_value no está en domain
- `TypeError`: Si domain no es una lista

##### add_constraint

Añade una restricción entre variables.

```python
engine.add_constraint(
    var1: str,
    var2: str,
    constraint: Callable[[Any, Any], bool],
    name: str = None,
    bidirectional: bool = True
) -> None
```

**Parámetros:**
- `var1` (str): Nombre de la primera variable
- `var2` (str): Nombre de la segunda variable
- `constraint` (Callable): Función que retorna True si la asignación es válida
- `name` (str, opcional): Nombre de la restricción para debugging
- `bidirectional` (bool): Si la restricción es simétrica

**Ejemplo:**
```python
# Restricción: x != y
engine.add_constraint("x", "y", lambda a, b: a != b, name="x_neq_y")

# Restricción: x < y
engine.add_constraint("x", "y", lambda a, b: a < b, name="x_less_y", bidirectional=False)

# Restricción compleja
def complex_constraint(a, b):
    return (a + b) % 2 == 0 and a * b < 100

engine.add_constraint("x", "y", complex_constraint, name="complex")
```

**Excepciones:**
- `ValueError`: Si var1 o var2 no existen
- `TypeError`: Si constraint no es callable

##### solve

Resuelve el problema CSP.

```python
solution = engine.solve(
    return_all: bool = False,
    max_solutions: int = 1,
    randomize: bool = False
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]
```

**Parámetros:**
- `return_all` (bool): Retornar todas las soluciones
- `max_solutions` (int): Número máximo de soluciones a buscar
- `randomize` (bool): Aleatorizar orden de búsqueda

**Retorna:**
- `Dict[str, Any]`: Una solución (si return_all=False)
- `List[Dict[str, Any]]`: Lista de soluciones (si return_all=True)
- `None`: Si no hay solución

**Ejemplo:**
```python
# Una solución
solution = engine.solve()
if solution:
    print(f"x = {solution['x']}, y = {solution['y']}")
else:
    print("No hay solución")

# Todas las soluciones
solutions = engine.solve(return_all=True)
print(f"Encontradas {len(solutions)} soluciones")

# Hasta 10 soluciones
solutions = engine.solve(return_all=True, max_solutions=10)
```

**Excepciones:**
- `TimeoutError`: Si se excede el timeout configurado
- `MemoryError`: Si se agota la memoria

##### get_statistics

Obtiene estadísticas de la última ejecución.

```python
stats = engine.get_statistics() -> Dict[str, Any]
```

**Retorna:**
```python
{
    'execution_time': 0.234,        # Tiempo en segundos
    'iterations': 15,               # Iteraciones del algoritmo
    'domain_reductions': 42,        # Reducciones de dominio
    'constraint_checks': 156,       # Evaluaciones de restricciones
    'backtracks': 3,                # Número de backtracks
    'cache_hits': 89,               # Hits del cache
    'cache_misses': 23,             # Misses del cache
    'memory_used_mb': 12.5          # Memoria usada en MB
}
```

##### reset

Reinicia el motor eliminando todas las variables y restricciones.

```python
engine.reset() -> None
```

---

### Ejemplo Completo: N-Queens

```python
from lattice_weaver.arc_engine import AdaptiveConsistencyEngine

def solve_n_queens(n):
    """Resuelve el problema de las N reinas."""
    engine = AdaptiveConsistencyEngine(
        algorithm='parallel',
        optimization_level=2,
        parallel=True
    )
    
    # Añadir variables (una por columna)
    for i in range(n):
        engine.add_variable(f"Q{i}", list(range(n)))
    
    # Añadir restricciones
    for i in range(n):
        for j in range(i + 1, n):
            # No en la misma fila
            engine.add_constraint(
                f"Q{i}", f"Q{j}",
                lambda a, b: a != b,
                name=f"row_{i}_{j}"
            )
            
            # No en la misma diagonal
            col_diff = j - i
            engine.add_constraint(
                f"Q{i}", f"Q{j}",
                lambda a, b, cd=col_diff: abs(a - b) != cd,
                name=f"diag_{i}_{j}"
            )
    
    # Resolver
    solution = engine.solve()
    
    # Estadísticas
    stats = engine.get_statistics()
    print(f"Resuelto en {stats['execution_time']:.3f}s")
    print(f"Iteraciones: {stats['iterations']}")
    
    return solution

# Resolver 8-queens
solution = solve_n_queens(8)
if solution:
    print("Solución encontrada:")
    for i in range(8):
        print(f"Reina {i} en fila {solution[f'Q{i}']}")
```

---

## FCA Engine API

### FormalContext

Representa un contexto formal (objetos × atributos).

#### Constructor

```python
from lattice_weaver.locales import FormalContext

context = FormalContext(
    name: str = "Unnamed Context"
)
```

#### Métodos Principales

##### add_object

Añade un objeto con sus atributos.

```python
context.add_object(
    name: str,
    attributes: List[str]
) -> None
```

**Ejemplo:**
```python
context = FormalContext("Animales")
context.add_object("Perro", ["Animal", "Mamífero", "Doméstico"])
context.add_object("Gato", ["Animal", "Mamífero", "Doméstico"])
context.add_object("León", ["Animal", "Mamífero", "Salvaje"])
context.add_object("Águila", ["Animal", "Ave", "Salvaje"])
```

##### add_attribute

Añade un atributo a un objeto existente.

```python
context.add_attribute(
    object_name: str,
    attribute: str
) -> None
```

##### get_extent

Obtiene la extensión de un conjunto de atributos.

```python
extent = context.get_extent(
    attributes: Set[str]
) -> Set[str]
```

**Ejemplo:**
```python
# Objetos que son Mamíferos y Domésticos
extent = context.get_extent({"Mamífero", "Doméstico"})
print(extent)  # {'Perro', 'Gato'}
```

##### get_intent

Obtiene la intensión de un conjunto de objetos.

```python
intent = context.get_intent(
    objects: Set[str]
) -> Set[str]
```

**Ejemplo:**
```python
# Atributos comunes a Perro y Gato
intent = context.get_intent({"Perro", "Gato"})
print(intent)  # {'Animal', 'Mamífero', 'Doméstico'}
```

---

### ConceptLattice

Representa un lattice de conceptos formales.

#### Constructor

```python
from lattice_weaver.locales import build_concept_lattice

lattice = build_concept_lattice(
    context: FormalContext,
    algorithm: str = 'next_closure'  # 'next_closure', 'cbo', 'in_close'
) -> ConceptLattice
```

#### Métodos Principales

##### get_concepts

Obtiene todos los conceptos del lattice.

```python
concepts = lattice.get_concepts() -> List[FormalConcept]
```

##### get_top

Obtiene el concepto superior (⊤).

```python
top = lattice.get_top() -> FormalConcept
```

##### get_bottom

Obtiene el concepto inferior (⊥).

```python
bottom = lattice.get_bottom() -> FormalConcept
```

##### visualize

Visualiza el lattice.

```python
lattice.visualize(
    layout: str = 'hierarchical',  # 'hierarchical', 'force', 'circular'
    show_labels: bool = True,
    output_file: str = None
) -> None
```

---

### Ejemplo Completo: Análisis de Documentos

```python
from lattice_weaver.locales import FormalContext, build_concept_lattice

# Crear contexto: documentos × palabras clave
context = FormalContext("Documentos")

context.add_object("Doc1", ["Python", "Machine Learning", "Data Science"])
context.add_object("Doc2", ["Python", "Web Development", "Django"])
context.add_object("Doc3", ["Java", "Enterprise", "Spring"])
context.add_object("Doc4", ["Python", "Data Science", "Pandas"])

# Construir lattice
lattice = build_concept_lattice(context, algorithm='next_closure')

# Analizar conceptos
concepts = lattice.get_concepts()
print(f"Total de conceptos: {len(concepts)}")

for concept in concepts:
    if len(concept.extent) > 1:  # Conceptos con múltiples documentos
        print(f"Documentos: {concept.extent}")
        print(f"Palabras clave comunes: {concept.intent}")
        print("---")

# Visualizar
lattice.visualize(layout='hierarchical', output_file='docs_lattice.html')
```

---

## TDA Engine API

### SimplicialComplex

Representa un complejo simplicial.

#### Constructor

```python
from lattice_weaver.topology import SimplicialComplex

complex = SimplicialComplex(
    dimension: int = None  # Dimensión máxima (None = sin límite)
)
```

#### Métodos Principales

##### add_simplex

Añade un simplex al complejo.

```python
complex.add_simplex(
    vertices: List[int],
    time: float = 0.0  # Para filtraciones
) -> None
```

**Ejemplo:**
```python
complex = SimplicialComplex()

# Añadir vértices (0-simplices)
complex.add_simplex([0])
complex.add_simplex([1])
complex.add_simplex([2])

# Añadir aristas (1-simplices)
complex.add_simplex([0, 1])
complex.add_simplex([1, 2])
complex.add_simplex([0, 2])

# Añadir triángulo (2-simplex)
complex.add_simplex([0, 1, 2])
```

##### compute_persistent_homology

Calcula homología persistente.

```python
persistence = complex.compute_persistent_homology(
    dimension: int = 2  # Dimensión máxima a calcular
) -> PersistenceDiagram
```

---

### VietorisRipsComplex

Construye un complejo de Vietoris-Rips desde un dataset.

```python
from lattice_weaver.topology import VietorisRipsComplex

vr_complex = VietorisRipsComplex(
    points: np.ndarray,      # Array de puntos (n_points × n_dimensions)
    max_edge_length: float,  # Radio máximo
    max_dimension: int = 2   # Dimensión máxima de simplices
)
```

**Ejemplo:**
```python
import numpy as np
from lattice_weaver.topology import VietorisRipsComplex

# Generar puntos aleatorios
points = np.random.rand(100, 2)

# Construir complejo
vr = VietorisRipsComplex(points, max_edge_length=0.3, max_dimension=2)

# Calcular homología persistente
persistence = vr.compute_persistent_homology(dimension=1)

# Visualizar
persistence.plot_barcode(output_file='barcode.png')
persistence.plot_diagram(output_file='diagram.png')
```

---

## Visualization API

### VisualizationEngine

Motor principal de visualización.

#### Constructor

```python
from lattice_weaver.visualization import VisualizationEngine

viz_engine = VisualizationEngine(
    renderer: str = 'd3',  # 'd3', 'plotly', 'matplotlib'
    interactive: bool = True,
    theme: str = 'light'   # 'light', 'dark'
)
```

#### Métodos Principales

##### visualize_csp

Visualiza un problema CSP.

```python
viz_engine.visualize_csp(
    engine: AdaptiveConsistencyEngine,
    show_domains: bool = True,
    show_constraints: bool = True,
    highlight_solution: Dict[str, Any] = None,
    output_file: str = None
) -> None
```

##### visualize_lattice

Visualiza un lattice de conceptos.

```python
viz_engine.visualize_lattice(
    lattice: ConceptLattice,
    layout: str = 'hierarchical',
    show_labels: bool = True,
    output_file: str = None
) -> None
```

##### visualize_persistence

Visualiza homología persistente.

```python
viz_engine.visualize_persistence(
    persistence: PersistenceDiagram,
    plot_type: str = 'barcode',  # 'barcode', 'diagram', 'both'
    output_file: str = None
) -> None
```

---

## Inference API

### InferenceEngine

Motor de inferencia que integra CSP, FCA y TDA.

#### Constructor

```python
from lattice_weaver.inference import InferenceEngine

inference = InferenceEngine(
    csp_engine: AdaptiveConsistencyEngine = None,
    fca_context: FormalContext = None,
    tda_complex: SimplicialComplex = None
)
```

#### Métodos Principales

##### infer_from_csp_to_fca

Infiere un contexto formal desde un CSP.

```python
context = inference.infer_from_csp_to_fca(
    solution: Dict[str, Any]
) -> FormalContext
```

##### infer_from_fca_to_csp

Infiere restricciones CSP desde un lattice FCA.

```python
constraints = inference.infer_from_fca_to_csp(
    lattice: ConceptLattice
) -> List[Constraint]
```

---

**Para más ejemplos y tutoriales, ver [`docs/tutorials/`](../tutorials/).**

