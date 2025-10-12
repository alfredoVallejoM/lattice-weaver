# Track C: Problem Families - Plan de Desarrollo

**Responsable:** Dev C (Problem Families Specialist)  
**Duración:** 6 semanas  
**Dependencias:** Débil con Track A (usa API de ArcEngine)

---

## 📋 Objetivo

Implementar un catálogo extenso de **familias de problemas CSP** con generadores paramétricos que permitan:

1. **Generar instancias** de problemas clásicos de CSP de forma programática
2. **Parametrizar** la complejidad y características de cada problema
3. **Facilitar benchmarking** y experimentación masiva
4. **Proporcionar base** para el Track H (Problemas Matemáticos Formales)

---

## 🎯 Familias de Problemas a Implementar

### Familia 1: N-Queens (N-Reinas)

**Descripción:** Colocar N reinas en un tablero de ajedrez NxN sin que se ataquen.

**Parámetros:**
- `n`: Tamaño del tablero (4 ≤ n ≤ 1000)

**Variables:** `Q0, Q1, ..., Q(n-1)` (posición de cada reina)

**Dominios:** `[0, n-1]` (columna de cada reina)

**Restricciones:**
- No dos reinas en la misma columna
- No dos reinas en la misma diagonal

**Complejidad:** O(n²) restricciones

---

### Familia 2: Graph Coloring (Coloración de Grafos)

**Descripción:** Asignar colores a nodos de un grafo tal que nodos adyacentes tengan colores diferentes.

**Parámetros:**
- `graph_type`: Tipo de grafo ('random', 'complete', 'bipartite', 'planar', 'grid')
- `n_nodes`: Número de nodos (10 ≤ n ≤ 1000)
- `n_colors`: Número de colores disponibles (2 ≤ k ≤ 10)
- `edge_probability`: Probabilidad de arista (para grafos aleatorios, 0.0 ≤ p ≤ 1.0)

**Variables:** `V0, V1, ..., V(n-1)` (color de cada nodo)

**Dominios:** `[0, k-1]` (colores disponibles)

**Restricciones:** Para cada arista (i, j): `color[i] != color[j]`

**Complejidad:** O(|E|) restricciones

---

### Familia 3: Sudoku

**Descripción:** Completar una cuadrícula 9x9 con dígitos 1-9 siguiendo las reglas del Sudoku.

**Parámetros:**
- `size`: Tamaño de la cuadrícula (4, 9, 16, 25)
- `n_clues`: Número de pistas iniciales (17 ≤ clues ≤ 60 para 9x9)
- `difficulty`: Nivel de dificultad ('easy', 'medium', 'hard', 'expert')

**Variables:** `C_{i,j}` para cada celda (i, j)

**Dominios:** `[1, size]` (dígitos posibles)

**Restricciones:**
- Fila: todos diferentes
- Columna: todos diferentes
- Bloque: todos diferentes

**Complejidad:** O(n³) restricciones (n = size)

---

### Familia 4: Map Coloring (Coloración de Mapas)

**Descripción:** Colorear regiones de un mapa tal que regiones adyacentes tengan colores diferentes.

**Parámetros:**
- `map_type`: Tipo de mapa ('usa_states', 'europe', 'random_planar')
- `n_colors`: Número de colores (típicamente 4)

**Variables:** `R0, R1, ..., R(n-1)` (color de cada región)

**Dominios:** `[0, k-1]` (colores disponibles)

**Restricciones:** Para cada frontera: `color[region1] != color[region2]`

---

### Familia 5: Job Shop Scheduling

**Descripción:** Asignar trabajos a máquinas minimizando el tiempo total (makespan).

**Parámetros:**
- `n_jobs`: Número de trabajos (5 ≤ n ≤ 100)
- `n_machines`: Número de máquinas (2 ≤ m ≤ 20)
- `max_duration`: Duración máxima de cada tarea (1 ≤ d ≤ 100)

**Variables:** `Start_{job,task}` (tiempo de inicio de cada tarea)

**Dominios:** `[0, max_makespan]`

**Restricciones:**
- Precedencia: tareas del mismo trabajo en orden
- No solapamiento: tareas en la misma máquina no se solapan

---

### Familia 6: Latin Square

**Descripción:** Llenar una cuadrícula nxn con símbolos tal que cada símbolo aparece exactamente una vez en cada fila y columna.

**Parámetros:**
- `n`: Tamaño de la cuadrícula (3 ≤ n ≤ 100)

**Variables:** `L_{i,j}` para cada celda (i, j)

**Dominios:** `[0, n-1]` (símbolos)

**Restricciones:**
- Fila: todos diferentes (AllDifferent)
- Columna: todos diferentes (AllDifferent)

---

### Familia 7: Magic Square

**Descripción:** Llenar una cuadrícula nxn con números 1..n² tal que todas las filas, columnas y diagonales sumen lo mismo.

**Parámetros:**
- `n`: Tamaño de la cuadrícula (3 ≤ n ≤ 10)

**Variables:** `M_{i,j}` para cada celda (i, j)

**Dominios:** `[1, n²]` (números)

**Restricciones:**
- Todos diferentes (AllDifferent global)
- Suma de cada fila = magic_constant
- Suma de cada columna = magic_constant
- Suma de cada diagonal = magic_constant

**Magic constant:** `n * (n² + 1) / 2`

---

### Familia 8: Knapsack (Mochila)

**Descripción:** Seleccionar items para maximizar valor sin exceder capacidad.

**Parámetros:**
- `n_items`: Número de items (10 ≤ n ≤ 1000)
- `capacity`: Capacidad de la mochila (100 ≤ C ≤ 10000)
- `value_range`: Rango de valores (1, 100)
- `weight_range`: Rango de pesos (1, 100)

**Variables:** `X_i` para cada item (incluido o no)

**Dominios:** `{0, 1}` (binario)

**Restricciones:**
- Suma de pesos ≤ capacidad

---

### Familia 9: Zebra Puzzle (Logic Puzzles)

**Descripción:** Resolver puzzles lógicos tipo "Einstein's Riddle".

**Parámetros:**
- `n_houses`: Número de casas (típicamente 5)
- `n_attributes`: Número de atributos por casa (color, nacionalidad, bebida, etc.)

**Variables:** Múltiples categorías de variables

**Dominios:** Valores específicos para cada categoría

**Restricciones:** Lógicas complejas (igualdad, adyacencia, etc.)

---

## 🏗️ Arquitectura del Módulo

### Estructura de Directorios

```
lattice_weaver/
└── problems/
    ├── __init__.py
    ├── base.py                    # Clase base ProblemFamily
    ├── catalog.py                 # ProblemCatalog (registro de familias)
    ├── generators/
    │   ├── __init__.py
    │   ├── nqueens.py            # Generador N-Queens
    │   ├── graph_coloring.py     # Generador Graph Coloring
    │   ├── sudoku.py             # Generador Sudoku
    │   ├── map_coloring.py       # Generador Map Coloring
    │   ├── scheduling.py         # Generador Job Shop
    │   ├── latin_square.py       # Generador Latin Square
    │   ├── magic_square.py       # Generador Magic Square
    │   ├── knapsack.py           # Generador Knapsack
    │   └── logic_puzzles.py      # Generador Logic Puzzles
    └── utils/
        ├── __init__.py
        ├── graph_generators.py   # Generadores de grafos
        └── validators.py         # Validadores de soluciones
```

### Clase Base: `ProblemFamily`

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from lattice_weaver.arc_engine import ArcEngine

class ProblemFamily(ABC):
    """
    Clase base abstracta para familias de problemas CSP.
    
    Cada familia de problemas debe implementar:
    - generate(): Generar una instancia del problema
    - validate_solution(): Validar una solución
    - get_metadata(): Obtener metadatos del problema
    """
    
    def __init__(self, name: str, description: str):
        """
        Inicializa la familia de problemas.
        
        Args:
            name: Nombre de la familia
            description: Descripción de la familia
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def generate(self, **params) -> ArcEngine:
        """
        Genera una instancia del problema con los parámetros dados.
        
        Args:
            **params: Parámetros específicos de la familia
            
        Returns:
            ArcEngine: Motor CSP configurado con el problema
        """
        pass
    
    @abstractmethod
    def validate_solution(self, solution: Dict[str, Any]) -> bool:
        """
        Valida si una solución es correcta.
        
        Args:
            solution: Diccionario variable -> valor
            
        Returns:
            bool: True si la solución es válida
        """
        pass
    
    @abstractmethod
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Obtiene metadatos del problema generado.
        
        Args:
            **params: Parámetros del problema
            
        Returns:
            Dict con metadatos (complejidad, # variables, # restricciones, etc.)
        """
        pass
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        Retorna parámetros por defecto para esta familia.
        
        Returns:
            Dict con parámetros por defecto
        """
        return {}
```

### Clase `ProblemCatalog`

```python
from typing import Dict, List, Optional
from .base import ProblemFamily

class ProblemCatalog:
    """
    Catálogo centralizado de familias de problemas CSP.
    
    Permite registrar, buscar y listar familias de problemas.
    """
    
    def __init__(self):
        """Inicializa el catálogo vacío."""
        self._families: Dict[str, ProblemFamily] = {}
    
    def register(self, family: ProblemFamily):
        """
        Registra una familia de problemas en el catálogo.
        
        Args:
            family: Instancia de ProblemFamily
        """
        if family.name in self._families:
            raise ValueError(f"Family '{family.name}' already registered")
        self._families[family.name] = family
    
    def get(self, name: str) -> Optional[ProblemFamily]:
        """
        Obtiene una familia por nombre.
        
        Args:
            name: Nombre de la familia
            
        Returns:
            ProblemFamily o None si no existe
        """
        return self._families.get(name)
    
    def list_families(self) -> List[str]:
        """
        Lista todas las familias registradas.
        
        Returns:
            Lista de nombres de familias
        """
        return list(self._families.keys())
    
    def generate_problem(self, family_name: str, **params) -> ArcEngine:
        """
        Genera un problema de una familia específica.
        
        Args:
            family_name: Nombre de la familia
            **params: Parámetros del problema
            
        Returns:
            ArcEngine configurado
        """
        family = self.get(family_name)
        if family is None:
            raise ValueError(f"Unknown family: {family_name}")
        return family.generate(**params)

# Instancia global del catálogo
_catalog = ProblemCatalog()

def get_catalog() -> ProblemCatalog:
    """Retorna la instancia global del catálogo."""
    return _catalog
```

---

## 📅 Plan de Implementación (6 Semanas)

### Semana 1: Infraestructura Base

**Tareas:**
1. Crear estructura de directorios
2. Implementar `ProblemFamily` (clase base)
3. Implementar `ProblemCatalog`
4. Implementar utilidades (`graph_generators.py`, `validators.py`)
5. Tests unitarios de infraestructura

**Entregables:**
- `lattice_weaver/problems/base.py`
- `lattice_weaver/problems/catalog.py`
- `lattice_weaver/problems/utils/`
- `tests/unit/test_problem_base.py`
- `tests/unit/test_catalog.py`

---

### Semana 2: Familias Básicas (1-3)

**Tareas:**
1. Implementar `NQueensProblem`
2. Implementar `GraphColoringProblem`
3. Implementar `SudokuProblem`
4. Tests unitarios para cada familia
5. Tests de integración

**Entregables:**
- `lattice_weaver/problems/generators/nqueens.py`
- `lattice_weaver/problems/generators/graph_coloring.py`
- `lattice_weaver/problems/generators/sudoku.py`
- Tests correspondientes

---

### Semana 3: Familias Intermedias (4-6)

**Tareas:**
1. Implementar `MapColoringProblem`
2. Implementar `JobShopSchedulingProblem`
3. Implementar `LatinSquareProblem`
4. Tests unitarios
5. Benchmarks de rendimiento

**Entregables:**
- Generadores correspondientes
- Tests y benchmarks

---

### Semana 4: Familias Avanzadas (7-9)

**Tareas:**
1. Implementar `MagicSquareProblem`
2. Implementar `KnapsackProblem`
3. Implementar `LogicPuzzlesProblem` (Zebra, Einstein)
4. Tests unitarios
5. Documentación de API

**Entregables:**
- Generadores correspondientes
- Documentación completa

---

### Semana 5: Experimentación y Benchmarking

**Tareas:**
1. Suite de experimentación masiva
2. Generación de datasets de benchmarking
3. Análisis de complejidad empírica
4. Visualizaciones de resultados
5. Documentación de experimentos

**Entregables:**
- `lattice_weaver/problems/experiments/`
- Datasets de benchmarking
- Reportes de análisis

---

### Semana 6: Integración y Documentación Final

**Tareas:**
1. Integración con Track A (ArcEngine)
2. Preparación para Track H (Formal Math)
3. Documentación exhaustiva
4. Tutoriales y ejemplos
5. Empaquetado y entrega

**Entregables:**
- Documentación completa
- Tutoriales
- `track-c-families-v1.0.tar.gz`

---

## 🧪 Estrategia de Testing

### Tests Unitarios

Para cada familia de problemas:

```python
def test_nqueens_generation():
    """Test que N-Queens genera correctamente."""
    family = NQueensProblem()
    engine = family.generate(n=8)
    assert len(engine.variables) == 8
    assert len(engine.constraints) == 28  # 8*7/2 = 28

def test_nqueens_solution_validation():
    """Test que valida soluciones correctas."""
    family = NQueensProblem()
    solution = {'Q0': 0, 'Q1': 4, 'Q2': 7, 'Q3': 5, 
                'Q4': 2, 'Q5': 6, 'Q6': 1, 'Q7': 3}
    assert family.validate_solution(solution) == True

def test_nqueens_metadata():
    """Test que metadatos son correctos."""
    family = NQueensProblem()
    metadata = family.get_metadata(n=8)
    assert metadata['n_variables'] == 8
    assert metadata['n_constraints'] == 28
    assert metadata['complexity'] == 'O(n^2)'
```

### Tests de Integración

```python
def test_catalog_registration():
    """Test que el catálogo registra familias."""
    catalog = get_catalog()
    assert 'nqueens' in catalog.list_families()
    assert 'graph_coloring' in catalog.list_families()

def test_end_to_end_nqueens():
    """Test end-to-end: generar y resolver N-Queens."""
    catalog = get_catalog()
    engine = catalog.generate_problem('nqueens', n=8)
    from lattice_weaver.arc_engine import CSPSolver
    solver = CSPSolver(engine)
    solution = solver.solve()
    assert solution is not None
    family = catalog.get('nqueens')
    assert family.validate_solution(solution) == True
```

### Benchmarks

```python
@pytest.mark.benchmark
def test_nqueens_performance(benchmark):
    """Benchmark de generación de N-Queens."""
    family = NQueensProblem()
    result = benchmark(family.generate, n=100)
    assert result is not None
```

---

## 📐 Principios de Diseño

### 1. Modularidad

Cada familia de problemas es un módulo independiente que puede desarrollarse y testearse por separado.

### 2. Extensibilidad

Nuevas familias se añaden simplemente creando una nueva clase que herede de `ProblemFamily` y registrándola en el catálogo.

### 3. Parametrización

Todos los aspectos relevantes de cada problema son parametrizables para facilitar experimentación.

### 4. Validación

Cada familia incluye un validador de soluciones para verificar corrección.

### 5. Metadatos

Cada problema generado incluye metadatos (complejidad, tamaño, etc.) para análisis posterior.

---

## 🔗 Dependencias con Otros Tracks

### Track A (Core Engine)

**Dependencia:** Débil (interfaz)

**Uso:** `ProblemFamily.generate()` retorna un `ArcEngine` configurado

**Interfaz:**
```python
from lattice_weaver.arc_engine import ArcEngine

def generate(self, **params) -> ArcEngine:
    engine = ArcEngine()
    engine.add_variable(...)
    engine.add_constraint(...)
    return engine
```

### Track H (Formal Math)

**Dependencia:** Track H depende de Track C

**Provisión:** Catálogo de familias de problemas como base para problemas matemáticos formales

**Interfaz:**
```python
from lattice_weaver.problems import get_catalog

catalog = get_catalog()
# Track H puede extender familias existentes o crear nuevas
```

---

## 📊 Métricas de Éxito

- ✅ 9 familias de problemas implementadas
- ✅ 100% cobertura de tests
- ✅ Documentación completa de API
- ✅ 3+ tutoriales de uso
- ✅ Suite de benchmarking funcional
- ✅ Integración exitosa con Track A

---

## 📚 Referencias

1. Dechter, R. (2003). *Constraint Processing*. Morgan Kaufmann.
2. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
3. CSPLib: A problem library for constraints - http://www.csplib.org/
4. Benchmarking Constraint Programming - https://www.cs.ox.ac.uk/files/2366/RR-09-07.pdf

---

**Última actualización:** 12 de Octubre, 2025  
**Estado:** Planificación completa - Listo para implementación

