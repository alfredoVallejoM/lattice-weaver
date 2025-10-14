# Track D - Inference Engine: Documento de Diseño

**Versión:** 1.0  
**Fecha:** 12 de Octubre, 2025  
**Track:** D - Inference Engine  
**Duración estimada:** 8 semanas  
**Agente responsable:** agent-track-d  
**Estado:** DISEÑO

---

## 📋 Tabla de Contenidos

1. [Visión General](#visión-general)
2. [Objetivos y Alcance](#objetivos-y-alcance)
3. [Análisis de Dependencias](#análisis-de-dependencias)
4. [Arquitectura del Sistema](#arquitectura-del-sistema)
5. [Componentes Principales](#componentes-principales)
6. [Diseño Detallado](#diseño-detallado)
7. [Interfaces y APIs](#interfaces-y-apis)
8. [Estrategias de Implementación](#estrategias-de-implementación)
9. [Plan de Fases](#plan-de-fases)
10. [Testing y Validación](#testing-y-validación)
11. [Principios de Diseño Aplicados](#principios-de-diseño-aplicados)
12. [Ejemplos de Uso](#ejemplos-de-uso)
13. [Riesgos y Mitigaciones](#riesgos-y-mitigaciones)

---

## 1. Visión General

### 1.1 Propósito

El **Inference Engine** (Motor de Inferencia) es el componente que permite a LatticeWeaver traducir especificaciones textuales de problemas complejos a representaciones formales (CSP, FCA, estructuras topológicas) que pueden ser procesadas por el motor de resolución.

### 1.2 Motivación

Actualmente, para usar LatticeWeaver es necesario:
1. Conocer la API del `ConstraintGraph`
2. Programar manualmente variables, dominios y restricciones
3. Tener conocimientos de programación en Python

El Inference Engine democratiza el acceso permitiendo:
- **Especificaciones textuales** en lenguaje natural o semi-formal
- **Inferencia automática** de restricciones implícitas
- **Traducción inteligente** a estructuras formales
- **Validación semántica** de especificaciones

### 1.3 Casos de Uso Principales

1. **Especificación textual de CSP:** "Tengo 4 reinas que no pueden atacarse en un tablero 4x4"
2. **Problemas de scheduling:** "3 tareas con duraciones [2h, 3h, 1h] y dependencias A→B, A→C"
3. **Problemas de asignación:** "Asignar 5 profesores a 8 cursos según preferencias y restricciones"
4. **Problemas de coloración:** "Colorear un mapa con 4 colores sin que regiones adyacentes compartan color"
5. **Integración con Track E (Web):** Frontend envía especificación textual, backend la traduce y resuelve
6. **Integración con Track H (Formal Math):** Parsear especificaciones formales matemáticas

---

## 2. Objetivos y Alcance

### 2.1 Objetivos Principales

1. **Parser flexible:** Interpretar especificaciones en múltiples formatos (natural, semi-formal, JSON, YAML)
2. **Inferencia inteligente:** Deducir restricciones implícitas y estructuras del problema
3. **Traducción robusta:** Convertir especificaciones a `ConstraintGraph` válidos
4. **Validación semántica:** Detectar inconsistencias y ambigüedades antes de resolver
5. **Extensibilidad:** Arquitectura modular para añadir nuevos tipos de problemas

### 2.2 Alcance de la Versión 1.0

**Incluido:**
- Parser de especificaciones textuales estructuradas (JSON/YAML)
- Parser de lenguaje natural simple (patrones predefinidos)
- Traducción a CSP (ConstraintGraph)
- Inferencia de restricciones básicas (alldifferent, precedencia, capacidad)
- Validación de consistencia sintáctica
- Integración con AdaptiveConsistencyEngine
- API Python y CLI

**Excluido (versiones futuras):**
- Parser de lenguaje natural avanzado (NLP con ML)
- Traducción a FCA y estructuras topológicas
- Inferencia de restricciones complejas (simetría, dominancia)
- Optimización automática de modelos
- Generación de explicaciones en lenguaje natural

### 2.3 Métricas de Éxito

- **Cobertura:** Soportar 10+ tipos de problemas CSP clásicos
- **Precisión:** 95%+ de especificaciones válidas traducidas correctamente
- **Robustez:** Detectar 90%+ de errores semánticos antes de resolver
- **Performance:** Traducción <100ms para problemas con <100 variables
- **Usabilidad:** Reducir en 80% el código necesario para definir un problema

---

## 3. Análisis de Dependencias

### 3.1 Dependencias Fuertes (Bloqueantes)

#### Track A - Core Engine ✅ COMPLETADO

**Componentes requeridos:**
- `ConstraintGraph` - Estructura de datos base
- `AdaptiveConsistencyEngine` - Motor de resolución
- `SearchSpaceTracer` - Para debugging y análisis

**Estado:** Completado al 100%. API estable y documentada.

**Interfaces utilizadas:**
```python
from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
```

### 3.2 Dependencias Débiles (Interfaces)

#### Track C - Problem Families (40% completado)

**Uso:** Catálogo de problemas para validación y testing.

**Estrategia:** Implementar generadores propios inicialmente, integrar con Track C cuando esté disponible.

#### Track B - Locales y Frames (60% completado)

**Uso futuro:** Traducción a FCA (v2.0 del Inference Engine).

**Estrategia:** No bloqueante para v1.0.

### 3.3 Tracks Dependientes

#### Track E - Web Application (IDLE)

**Dependencia:** Requiere Inference Engine completo.

**Interface esperada:**
```python
POST /api/inference/parse
POST /api/inference/solve
GET /api/inference/status/{job_id}
```

#### Track H - Problemas Matemáticos (IDLE)

**Dependencia parcial:** Puede usar parser de especificaciones formales.

**Interface esperada:**
```python
def parse_formal_specification(text: str) -> CSPProblem
```

---

## 4. Arquitectura del Sistema

### 4.1 Visión de Alto Nivel

```
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE ENGINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Parsers    │──────│  Inference   │──────│ Builders │ │
│  │              │      │    Layer     │      │          │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│        │                      │                     │      │
│        ▼                      ▼                     ▼      │
│  ┌──────────────────────────────────────────────────────┐ │
│  │            Intermediate Representation (IR)          │ │
│  └──────────────────────────────────────────────────────┘ │
│                              │                             │
└──────────────────────────────┼─────────────────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   ConstraintGraph    │
                    │  (Track A - ACE)     │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ AdaptiveConsistency  │
                    │      Engine          │
                    └──────────────────────┘
```

### 4.2 Flujo de Datos

```
Especificación Textual
        │
        ▼
   [Parser Layer]
        │
        ▼
Intermediate Representation (IR)
        │
        ▼
   [Inference Layer]
        │
        ▼
Enhanced IR (con restricciones inferidas)
        │
        ▼
   [Builder Layer]
        │
        ▼
ConstraintGraph
        │
        ▼
   [Validation]
        │
        ▼
Problema Listo para Resolver
```

---

## 5. Componentes Principales

### 5.1 Parser Layer

**Responsabilidad:** Convertir texto a representación intermedia (IR).

**Componentes:**
- `TextParser` - Parser de lenguaje natural simple
- `JSONParser` - Parser de especificaciones JSON
- `YAMLParser` - Parser de especificaciones YAML
- `FormalParser` - Parser de notación matemática formal (v2.0)

**Entrada:** String (texto, JSON, YAML)  
**Salida:** `ProblemIR` (Intermediate Representation)

### 5.2 Inference Layer

**Responsabilidad:** Enriquecer IR con restricciones y estructuras inferidas.

**Componentes:**
- `ConstraintInferencer` - Infiere restricciones implícitas
- `StructureDetector` - Detecta patrones estructurales (cliques, bipartitos, etc.)
- `DomainInferencer` - Infiere dominios de variables
- `SymmetryDetector` - Detecta simetrías (v2.0)

**Entrada:** `ProblemIR`  
**Salida:** `EnhancedProblemIR`

### 5.3 Builder Layer

**Responsabilidad:** Construir `ConstraintGraph` desde IR enriquecido.

**Componentes:**
- `ConstraintGraphBuilder` - Constructor principal
- `ConstraintFactory` - Fábrica de funciones de restricción
- `DomainBuilder` - Constructor de dominios

**Entrada:** `EnhancedProblemIR`  
**Salida:** `ConstraintGraph`

### 5.4 Validation Layer

**Responsabilidad:** Validar consistencia semántica del problema.

**Componentes:**
- `SemanticValidator` - Valida consistencia semántica
- `SyntaxValidator` - Valida sintaxis de especificaciones
- `ConsistencyChecker` - Verifica satisfacibilidad básica

**Entrada:** `ProblemIR` o `ConstraintGraph`  
**Salida:** `ValidationResult`

---

## 6. Diseño Detallado

### 6.1 Intermediate Representation (IR)

La IR es la estructura de datos central que desacopla parsers de builders.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any
from enum import Enum


class VariableType(Enum):
    """Tipos de variables soportados."""
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    REAL = "real"  # v2.0


class ConstraintType(Enum):
    """Tipos de restricciones soportados."""
    BINARY = "binary"           # Restricción entre 2 variables
    ALLDIFFERENT = "alldifferent"  # Todas diferentes
    PRECEDENCE = "precedence"   # Orden temporal
    CAPACITY = "capacity"       # Límite de capacidad
    SUM = "sum"                 # Suma de variables
    CUSTOM = "custom"           # Función personalizada


@dataclass
class Variable:
    """Representación de una variable en IR."""
    name: str
    var_type: VariableType
    domain: Optional[Set[Any]] = None
    domain_range: Optional[tuple] = None  # (min, max) para enteros
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validar que se especifique dominio o rango."""
        if self.domain is None and self.domain_range is None:
            raise ValueError(f"Variable '{self.name}' must have domain or domain_range")


@dataclass
class Constraint:
    """Representación de una restricción en IR."""
    constraint_type: ConstraintType
    variables: List[str]  # Nombres de variables involucradas
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    # Para restricciones binarias custom
    function: Optional[Callable[[Any, Any], bool]] = None
    function_code: Optional[str] = None  # Código fuente para serialización
    
    def __post_init__(self):
        """Validar consistencia."""
        if self.constraint_type == ConstraintType.BINARY and len(self.variables) != 2:
            raise ValueError("BINARY constraint must have exactly 2 variables")
        if self.constraint_type == ConstraintType.CUSTOM and self.function is None:
            raise ValueError("CUSTOM constraint must have a function")


@dataclass
class ProblemIR:
    """
    Representación Intermedia de un problema CSP.
    
    Esta estructura desacopla el parsing de la construcción del ConstraintGraph.
    """
    name: str
    description: Optional[str] = None
    
    variables: Dict[str, Variable] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)
    
    # Metadata adicional
    problem_type: Optional[str] = None  # "n-queens", "scheduling", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_variable(self, var: Variable):
        """Añade una variable al problema."""
        if var.name in self.variables:
            raise ValueError(f"Variable '{var.name}' already exists")
        self.variables[var.name] = var
    
    def add_constraint(self, constraint: Constraint):
        """Añade una restricción al problema."""
        # Validar que las variables existen
        for var_name in constraint.variables:
            if var_name not in self.variables:
                raise ValueError(f"Variable '{var_name}' not found in problem")
        self.constraints.append(constraint)
    
    def get_variable(self, name: str) -> Variable:
        """Obtiene una variable por nombre."""
        if name not in self.variables:
            raise KeyError(f"Variable '{name}' not found")
        return self.variables[name]
    
    def validate(self) -> List[str]:
        """
        Valida la consistencia del IR.
        
        Returns:
            Lista de errores (vacía si es válido)
        """
        errors = []
        
        # Validar que hay al menos una variable
        if not self.variables:
            errors.append("Problem must have at least one variable")
        
        # Validar que las restricciones referencian variables existentes
        for i, constraint in enumerate(self.constraints):
            for var_name in constraint.variables:
                if var_name not in self.variables:
                    errors.append(
                        f"Constraint {i} references undefined variable '{var_name}'"
                    )
        
        return errors


@dataclass
class EnhancedProblemIR(ProblemIR):
    """
    IR enriquecido con información inferida.
    
    Extiende ProblemIR con restricciones y estructuras detectadas
    por el Inference Layer.
    """
    inferred_constraints: List[Constraint] = field(default_factory=list)
    detected_patterns: Dict[str, Any] = field(default_factory=dict)
    symmetries: List[Any] = field(default_factory=list)  # v2.0
```

### 6.2 Parser Layer

#### 6.2.1 Base Parser Interface

```python
from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Interfaz base para todos los parsers."""
    
    @abstractmethod
    def parse(self, input_text: str) -> ProblemIR:
        """
        Parsea una especificación textual a IR.
        
        Args:
            input_text: Especificación del problema
        
        Returns:
            ProblemIR representando el problema
        
        Raises:
            ParseError: Si la especificación es inválida
        """
        pass
    
    @abstractmethod
    def supports(self, input_text: str) -> bool:
        """
        Verifica si este parser puede manejar la entrada.
        
        Args:
            input_text: Texto a verificar
        
        Returns:
            True si puede parsear, False en caso contrario
        """
        pass


class ParseError(Exception):
    """Excepción lanzada cuando el parsing falla."""
    
    def __init__(self, message: str, line: Optional[int] = None, column: Optional[int] = None):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.line is not None and self.column is not None:
            return f"Parse error at line {self.line}, column {self.column}: {self.message}"
        elif self.line is not None:
            return f"Parse error at line {self.line}: {self.message}"
        else:
            return f"Parse error: {self.message}"
```

#### 6.2.2 JSON Parser

```python
import json
from typing import Dict, Any


class JSONParser(BaseParser):
    """
    Parser de especificaciones en formato JSON.
    
    Formato esperado:
    {
        "name": "Problem Name",
        "description": "Optional description",
        "variables": [
            {
                "name": "X",
                "type": "integer",
                "domain": [1, 2, 3]
            },
            ...
        ],
        "constraints": [
            {
                "type": "binary",
                "variables": ["X", "Y"],
                "relation": "!="
            },
            ...
        ]
    }
    """
    
    def supports(self, input_text: str) -> bool:
        """Verifica si es JSON válido."""
        try:
            json.loads(input_text)
            return True
        except json.JSONDecodeError:
            return False
    
    def parse(self, input_text: str) -> ProblemIR:
        """Parsea JSON a ProblemIR."""
        try:
            data = json.loads(input_text)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e.msg}", line=e.lineno, column=e.colno)
        
        # Validar estructura básica
        if "variables" not in data:
            raise ParseError("Missing 'variables' field")
        
        # Crear IR
        problem = ProblemIR(
            name=data.get("name", "Unnamed Problem"),
            description=data.get("description"),
            problem_type=data.get("type"),
            metadata=data.get("metadata", {})
        )
        
        # Parsear variables
        for var_data in data["variables"]:
            var = self._parse_variable(var_data)
            problem.add_variable(var)
        
        # Parsear restricciones
        for constraint_data in data.get("constraints", []):
            constraint = self._parse_constraint(constraint_data)
            problem.add_constraint(constraint)
        
        return problem
    
    def _parse_variable(self, var_data: Dict[str, Any]) -> Variable:
        """Parsea una variable desde JSON."""
        if "name" not in var_data:
            raise ParseError("Variable missing 'name' field")
        
        name = var_data["name"]
        var_type_str = var_data.get("type", "integer")
        
        try:
            var_type = VariableType(var_type_str)
        except ValueError:
            raise ParseError(f"Invalid variable type: {var_type_str}")
        
        # Parsear dominio
        domain = None
        domain_range = None
        
        if "domain" in var_data:
            domain = set(var_data["domain"])
        elif "range" in var_data:
            range_data = var_data["range"]
            if isinstance(range_data, list) and len(range_data) == 2:
                domain_range = tuple(range_data)
            else:
                raise ParseError(f"Invalid range format for variable '{name}'")
        else:
            raise ParseError(f"Variable '{name}' must have 'domain' or 'range'")
        
        return Variable(
            name=name,
            var_type=var_type,
            domain=domain,
            domain_range=domain_range,
            description=var_data.get("description"),
            metadata=var_data.get("metadata", {})
        )
    
    def _parse_constraint(self, constraint_data: Dict[str, Any]) -> Constraint:
        """Parsea una restricción desde JSON."""
        if "type" not in constraint_data:
            raise ParseError("Constraint missing 'type' field")
        if "variables" not in constraint_data:
            raise ParseError("Constraint missing 'variables' field")
        
        constraint_type_str = constraint_data["type"]
        try:
            constraint_type = ConstraintType(constraint_type_str)
        except ValueError:
            raise ParseError(f"Invalid constraint type: {constraint_type_str}")
        
        variables = constraint_data["variables"]
        if not isinstance(variables, list):
            raise ParseError("'variables' must be a list")
        
        # Parsear parámetros según el tipo
        parameters = {}
        function = None
        function_code = None
        
        if constraint_type == ConstraintType.BINARY:
            # Relación binaria: "!=", "<", ">", "==", etc.
            relation = constraint_data.get("relation", "!=")
            parameters["relation"] = relation
            function = self._create_binary_function(relation)
        
        elif constraint_type == ConstraintType.SUM:
            # Suma: sum(variables) op value
            operator = constraint_data.get("operator", "==")
            value = constraint_data.get("value", 0)
            parameters["operator"] = operator
            parameters["value"] = value
        
        else:
            # Otros tipos: copiar parámetros directamente
            parameters = constraint_data.get("parameters", {})
        
        return Constraint(
            constraint_type=constraint_type,
            variables=variables,
            parameters=parameters,
            description=constraint_data.get("description"),
            function=function,
            function_code=function_code
        )
    
    def _create_binary_function(self, relation: str) -> Callable[[Any, Any], bool]:
        """Crea función de restricción binaria desde operador."""
        operators = {
            "!=": lambda a, b: a != b,
            "==": lambda a, b: a == b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
        }
        
        if relation not in operators:
            raise ParseError(f"Unknown binary relation: {relation}")
        
        return operators[relation]
```

#### 6.2.3 YAML Parser

```python
import yaml


class YAMLParser(BaseParser):
    """
    Parser de especificaciones en formato YAML.
    
    Usa el mismo esquema que JSONParser pero con sintaxis YAML.
    """
    
    def __init__(self):
        self.json_parser = JSONParser()
    
    def supports(self, input_text: str) -> bool:
        """Verifica si es YAML válido."""
        try:
            yaml.safe_load(input_text)
            return True
        except yaml.YAMLError:
            return False
    
    def parse(self, input_text: str) -> ProblemIR:
        """Parsea YAML a ProblemIR."""
        try:
            data = yaml.safe_load(input_text)
        except yaml.YAMLError as e:
            raise ParseError(f"Invalid YAML: {str(e)}")
        
        # Convertir a JSON string y usar JSONParser
        json_str = json.dumps(data)
        return self.json_parser.parse(json_str)
```

#### 6.2.4 Text Parser (Lenguaje Natural Simple)

```python
import re
from typing import List, Tuple


class TextParser(BaseParser):
    """
    Parser de especificaciones en lenguaje natural simple.
    
    Soporta patrones predefinidos para problemas comunes:
    - N-Queens: "N reinas en tablero MxM sin atacarse"
    - Graph Coloring: "Colorear grafo con K colores"
    - Scheduling: "Asignar N tareas a M recursos"
    
    Versión 1.0: Patrones fijos con regex.
    Versión 2.0: NLP con ML para mayor flexibilidad.
    """
    
    def __init__(self):
        self.patterns = [
            (r"(\d+)\s+reinas?\s+.*tablero\s+(\d+)x\2", self._parse_nqueens),
            (r"colorear\s+.*grafo\s+.*(\d+)\s+colores?", self._parse_graph_coloring),
            # Más patrones...
        ]
    
    def supports(self, input_text: str) -> bool:
        """Verifica si coincide con algún patrón."""
        text_lower = input_text.lower()
        for pattern, _ in self.patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def parse(self, input_text: str) -> ProblemIR:
        """Parsea texto a ProblemIR usando patrones."""
        text_lower = input_text.lower()
        
        for pattern, parser_func in self.patterns:
            match = re.search(pattern, text_lower)
            if match:
                return parser_func(match)
        
        raise ParseError("No matching pattern found for input text")
    
    def _parse_nqueens(self, match: re.Match) -> ProblemIR:
        """Parsea problema de N-Reinas."""
        n = int(match.group(1))
        board_size = int(match.group(2))
        
        if n != board_size:
            raise ParseError(f"Number of queens ({n}) must match board size ({board_size})")
        
        problem = ProblemIR(
            name=f"{n}-Queens",
            description=f"Place {n} queens on a {n}x{n} board without attacking each other",
            problem_type="n-queens"
        )
        
        # Variables: una por fila, dominio = columnas
        for i in range(n):
            problem.add_variable(Variable(
                name=f"Q{i}",
                var_type=VariableType.INTEGER,
                domain=set(range(n)),
                description=f"Column position of queen in row {i}"
            ))
        
        # Restricciones: no misma columna, no misma diagonal
        for i in range(n):
            for j in range(i + 1, n):
                # No misma columna
                problem.add_constraint(Constraint(
                    constraint_type=ConstraintType.BINARY,
                    variables=[f"Q{i}", f"Q{j}"],
                    parameters={"relation": "!="},
                    function=lambda vi, vj: vi != vj,
                    description=f"Queens {i} and {j} not in same column"
                ))
                
                # No misma diagonal
                row_diff = j - i
                problem.add_constraint(Constraint(
                    constraint_type=ConstraintType.CUSTOM,
                    variables=[f"Q{i}", f"Q{j}"],
                    parameters={"row_diff": row_diff},
                    function=lambda vi, vj, rd=row_diff: abs(vi - vj) != rd,
                    description=f"Queens {i} and {j} not in same diagonal"
                ))
        
        return problem
    
    def _parse_graph_coloring(self, match: re.Match) -> ProblemIR:
        """Parsea problema de coloración de grafos."""
        k = int(match.group(1))
        
        # Nota: Este parser necesita información adicional sobre el grafo
        # En v1.0, requerir especificación JSON/YAML del grafo
        raise ParseError(
            "Graph coloring requires graph structure. "
            "Please use JSON/YAML format to specify nodes and edges."
        )
```

### 6.3 Inference Layer

```python
class ConstraintInferencer:
    """
    Infiere restricciones implícitas desde el IR.
    
    Estrategias:
    - Detectar patrones AllDifferent
    - Inferir precedencias temporales
    - Detectar límites de capacidad
    """
    
    def infer(self, problem: ProblemIR) -> List[Constraint]:
        """
        Infiere restricciones adicionales.
        
        Args:
            problem: IR del problema
        
        Returns:
            Lista de restricciones inferidas
        """
        inferred = []
        
        # Estrategia 1: Detectar AllDifferent
        alldiff_constraints = self._detect_alldifferent(problem)
        inferred.extend(alldiff_constraints)
        
        # Estrategia 2: Inferir precedencias (para scheduling)
        if problem.problem_type == "scheduling":
            precedence_constraints = self._infer_precedences(problem)
            inferred.extend(precedence_constraints)
        
        return inferred
    
    def _detect_alldifferent(self, problem: ProblemIR) -> List[Constraint]:
        """
        Detecta si múltiples restricciones binarias != forman un AllDifferent.
        
        Si todas las variables están conectadas con !=, es más eficiente
        usar una restricción global AllDifferent.
        """
        # Construir grafo de restricciones !=
        neq_graph = {}
        for constraint in problem.constraints:
            if (constraint.constraint_type == ConstraintType.BINARY and
                constraint.parameters.get("relation") == "!="):
                v1, v2 = constraint.variables
                neq_graph.setdefault(v1, set()).add(v2)
                neq_graph.setdefault(v2, set()).add(v1)
        
        # Detectar cliques (componentes completamente conectados)
        cliques = self._find_cliques(neq_graph)
        
        # Crear restricciones AllDifferent para cliques grandes
        alldiff_constraints = []
        for clique in cliques:
            if len(clique) >= 3:  # Solo si tiene sentido
                alldiff_constraints.append(Constraint(
                    constraint_type=ConstraintType.ALLDIFFERENT,
                    variables=list(clique),
                    description=f"All variables {clique} must be different (inferred)"
                ))
        
        return alldiff_constraints
    
    def _find_cliques(self, graph: Dict[str, Set[str]]) -> List[Set[str]]:
        """Encuentra cliques maximales en el grafo."""
        # Implementación simple: Bron-Kerbosch
        cliques = []
        
        def bron_kerbosch(r, p, x):
            if not p and not x:
                if len(r) >= 3:
                    cliques.append(set(r))
                return
            
            for v in list(p):
                neighbors = graph.get(v, set())
                bron_kerbosch(
                    r | {v},
                    p & neighbors,
                    x & neighbors
                )
                p.remove(v)
                x.add(v)
        
        all_nodes = set(graph.keys())
        bron_kerbosch(set(), all_nodes, set())
        
        return cliques
    
    def _infer_precedences(self, problem: ProblemIR) -> List[Constraint]:
        """Infiere precedencias temporales para problemas de scheduling."""
        # v2.0: Analizar metadata para detectar dependencias implícitas
        return []


class StructureDetector:
    """Detecta patrones estructurales en el problema."""
    
    def detect(self, problem: ProblemIR) -> Dict[str, Any]:
        """
        Detecta patrones estructurales.
        
        Returns:
            Diccionario con patrones detectados
        """
        patterns = {}
        
        # Detectar si es bipartito
        patterns["is_bipartite"] = self._is_bipartite(problem)
        
        # Detectar si es árbol
        patterns["is_tree"] = self._is_tree(problem)
        
        # Detectar cliques
        patterns["cliques"] = self._detect_cliques(problem)
        
        return patterns
    
    def _is_bipartite(self, problem: ProblemIR) -> bool:
        """Verifica si el grafo de restricciones es bipartito."""
        # Implementación: 2-coloring con BFS
        # v1.0: Simplificado
        return False
    
    def _is_tree(self, problem: ProblemIR) -> bool:
        """Verifica si el grafo de restricciones es un árbol."""
        # Implementación: verificar conectividad y |E| = |V| - 1
        # v1.0: Simplificado
        return False
    
    def _detect_cliques(self, problem: ProblemIR) -> List[Set[str]]:
        """Detecta cliques en el grafo de restricciones."""
        # Similar a ConstraintInferencer._find_cliques
        return []
```

### 6.4 Builder Layer

```python
class ConstraintGraphBuilder:
    """
    Construye ConstraintGraph desde EnhancedProblemIR.
    """
    
    def __init__(self):
        self.constraint_factory = ConstraintFactory()
    
    def build(self, problem_ir: ProblemIR) -> ConstraintGraph:
        """
        Construye ConstraintGraph desde IR.
        
        Args:
            problem_ir: Representación intermedia del problema
        
        Returns:
            ConstraintGraph listo para resolver
        
        Raises:
            BuildError: Si la construcción falla
        """
        cg = ConstraintGraph()
        
        # Paso 1: Añadir variables
        for var_name, var in problem_ir.variables.items():
            domain = self._build_domain(var)
            cg.add_variable(var_name, domain)
        
        # Paso 2: Añadir restricciones explícitas
        for constraint in problem_ir.constraints:
            self._add_constraint_to_graph(cg, constraint)
        
        # Paso 3: Añadir restricciones inferidas (si es EnhancedProblemIR)
        if isinstance(problem_ir, EnhancedProblemIR):
            for constraint in problem_ir.inferred_constraints:
                self._add_constraint_to_graph(cg, constraint)
        
        return cg
    
    def _build_domain(self, var: Variable) -> set:
        """Construye dominio de una variable."""
        if var.domain is not None:
            return var.domain
        elif var.domain_range is not None:
            min_val, max_val = var.domain_range
            return set(range(min_val, max_val + 1))
        else:
            raise BuildError(f"Variable '{var.name}' has no domain or range")
    
    def _add_constraint_to_graph(self, cg: ConstraintGraph, constraint: Constraint):
        """Añade una restricción al ConstraintGraph."""
        if constraint.constraint_type == ConstraintType.BINARY:
            # Restricción binaria simple
            v1, v2 = constraint.variables
            func = constraint.function
            if func is None:
                # Crear función desde parámetros
                relation = constraint.parameters.get("relation", "!=")
                func = self.constraint_factory.create_binary(relation)
            
            cg.add_constraint(v1, v2, func)
        
        elif constraint.constraint_type == ConstraintType.ALLDIFFERENT:
            # AllDifferent: descomponer en restricciones binarias
            variables = constraint.variables
            for i in range(len(variables)):
                for j in range(i + 1, len(variables)):
                    cg.add_constraint(
                        variables[i],
                        variables[j],
                        lambda a, b: a != b
                    )
        
        elif constraint.constraint_type == ConstraintType.CUSTOM:
            # Restricción custom: debe tener función
            if constraint.function is None:
                raise BuildError(f"CUSTOM constraint must have a function")
            
            if len(constraint.variables) == 2:
                v1, v2 = constraint.variables
                cg.add_constraint(v1, v2, constraint.function)
            else:
                # Restricciones n-arias: descomponer o error
                raise BuildError(
                    f"N-ary constraints not supported in v1.0. "
                    f"Constraint has {len(constraint.variables)} variables."
                )
        
        else:
            raise BuildError(f"Unsupported constraint type: {constraint.constraint_type}")


class ConstraintFactory:
    """Fábrica de funciones de restricción."""
    
    def create_binary(self, relation: str) -> Callable[[Any, Any], bool]:
        """Crea función de restricción binaria."""
        operators = {
            "!=": lambda a, b: a != b,
            "==": lambda a, b: a == b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
        }
        
        if relation not in operators:
            raise BuildError(f"Unknown binary relation: {relation}")
        
        return operators[relation]


class BuildError(Exception):
    """Excepción lanzada cuando la construcción del grafo falla."""
    pass
```

### 6.5 Validation Layer

```python
@dataclass
class ValidationResult:
    """Resultado de validación."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str):
        """Añade un error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Añade una advertencia."""
        self.warnings.append(message)


class SemanticValidator:
    """Valida consistencia semántica del problema."""
    
    def validate(self, problem_ir: ProblemIR) -> ValidationResult:
        """
        Valida el IR del problema.
        
        Args:
            problem_ir: IR a validar
        
        Returns:
            ValidationResult con errores y advertencias
        """
        result = ValidationResult(is_valid=True)
        
        # Validación 1: Variables sin restricciones
        self._check_isolated_variables(problem_ir, result)
        
        # Validación 2: Dominios vacíos o muy pequeños
        self._check_domain_sizes(problem_ir, result)
        
        # Validación 3: Restricciones contradictorias
        self._check_contradictory_constraints(problem_ir, result)
        
        # Validación 4: Problema trivialmente insatisfacible
        self._check_trivial_unsatisfiability(problem_ir, result)
        
        return result
    
    def _check_isolated_variables(self, problem_ir: ProblemIR, result: ValidationResult):
        """Detecta variables sin restricciones."""
        constrained_vars = set()
        for constraint in problem_ir.constraints:
            constrained_vars.update(constraint.variables)
        
        for var_name in problem_ir.variables:
            if var_name not in constrained_vars:
                result.add_warning(
                    f"Variable '{var_name}' has no constraints (unconstrained)"
                )
    
    def _check_domain_sizes(self, problem_ir: ProblemIR, result: ValidationResult):
        """Verifica tamaños de dominios."""
        for var_name, var in problem_ir.variables.items():
            domain_size = len(var.domain) if var.domain else (
                var.domain_range[1] - var.domain_range[0] + 1
            )
            
            if domain_size == 0:
                result.add_error(f"Variable '{var_name}' has empty domain")
            elif domain_size == 1:
                result.add_warning(
                    f"Variable '{var_name}' has domain of size 1 (already assigned)"
                )
    
    def _check_contradictory_constraints(self, problem_ir: ProblemIR, result: ValidationResult):
        """Detecta restricciones contradictorias obvias."""
        # v1.0: Detección simple
        # v2.0: Análisis más sofisticado
        pass
    
    def _check_trivial_unsatisfiability(self, problem_ir: ProblemIR, result: ValidationResult):
        """Detecta problemas trivialmente insatisfacibles."""
        # Ejemplo: 2 variables con dominios {1} y restricción !=
        # v1.0: Casos simples
        # v2.0: Análisis más profundo
        pass
```

---

## 7. Interfaces y APIs

### 7.1 API Python

```python
# lattice_weaver/inference/__init__.py

from .parser import InferenceEngine, parse_problem
from .ir import ProblemIR, Variable, Constraint, VariableType, ConstraintType

__all__ = [
    'InferenceEngine',
    'parse_problem',
    'ProblemIR',
    'Variable',
    'Constraint',
    'VariableType',
    'ConstraintType',
]


# lattice_weaver/inference/engine.py

class InferenceEngine:
    """
    Motor de inferencia principal.
    
    Orquesta el flujo completo: parsing → inference → building → validation.
    
    Examples:
        >>> engine = InferenceEngine()
        >>> cg = engine.parse_and_build(json_spec)
        >>> solver = AdaptiveConsistencyEngine()
        >>> stats = solver.solve(cg)
    """
    
    def __init__(
        self,
        enable_inference: bool = True,
        enable_validation: bool = True
    ):
        """
        Inicializa el motor de inferencia.
        
        Args:
            enable_inference: Habilitar inferencia de restricciones
            enable_validation: Habilitar validación semántica
        """
        self.parsers = [
            JSONParser(),
            YAMLParser(),
            TextParser(),
        ]
        self.inferencer = ConstraintInferencer()
        self.structure_detector = StructureDetector()
        self.builder = ConstraintGraphBuilder()
        self.validator = SemanticValidator()
        
        self.enable_inference = enable_inference
        self.enable_validation = enable_validation
    
    def parse(self, input_text: str) -> ProblemIR:
        """
        Parsea especificación textual a IR.
        
        Args:
            input_text: Especificación del problema
        
        Returns:
            ProblemIR
        
        Raises:
            ParseError: Si no se puede parsear
        """
        # Intentar con cada parser
        for parser in self.parsers:
            if parser.supports(input_text):
                return parser.parse(input_text)
        
        raise ParseError("No parser found for input format")
    
    def infer(self, problem_ir: ProblemIR) -> EnhancedProblemIR:
        """
        Enriquece IR con restricciones inferidas.
        
        Args:
            problem_ir: IR del problema
        
        Returns:
            EnhancedProblemIR con restricciones inferidas
        """
        enhanced = EnhancedProblemIR(**problem_ir.__dict__)
        
        if self.enable_inference:
            # Inferir restricciones
            inferred_constraints = self.inferencer.infer(problem_ir)
            enhanced.inferred_constraints = inferred_constraints
            
            # Detectar patrones estructurales
            patterns = self.structure_detector.detect(problem_ir)
            enhanced.detected_patterns = patterns
        
        return enhanced
    
    def build(self, problem_ir: ProblemIR) -> ConstraintGraph:
        """
        Construye ConstraintGraph desde IR.
        
        Args:
            problem_ir: IR del problema
        
        Returns:
            ConstraintGraph
        
        Raises:
            BuildError: Si la construcción falla
        """
        return self.builder.build(problem_ir)
    
    def validate(self, problem_ir: ProblemIR) -> ValidationResult:
        """
        Valida el IR del problema.
        
        Args:
            problem_ir: IR a validar
        
        Returns:
            ValidationResult
        """
        if not self.enable_validation:
            return ValidationResult(is_valid=True)
        
        return self.validator.validate(problem_ir)
    
    def parse_and_build(
        self,
        input_text: str,
        validate: bool = True
    ) -> ConstraintGraph:
        """
        Flujo completo: parse → infer → validate → build.
        
        Args:
            input_text: Especificación del problema
            validate: Si validar antes de construir
        
        Returns:
            ConstraintGraph listo para resolver
        
        Raises:
            ParseError: Si el parsing falla
            ValidationError: Si la validación falla
            BuildError: Si la construcción falla
        """
        # Paso 1: Parse
        problem_ir = self.parse(input_text)
        
        # Paso 2: Infer
        enhanced_ir = self.infer(problem_ir)
        
        # Paso 3: Validate
        if validate:
            validation_result = self.validate(enhanced_ir)
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Validation failed with {len(validation_result.errors)} errors",
                    errors=validation_result.errors
                )
        
        # Paso 4: Build
        cg = self.build(enhanced_ir)
        
        return cg


class ValidationError(Exception):
    """Excepción lanzada cuando la validación falla."""
    
    def __init__(self, message: str, errors: List[str]):
        self.message = message
        self.errors = errors
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        error_list = "\n".join(f"  - {err}" for err in self.errors)
        return f"{self.message}:\n{error_list}"


# Función de conveniencia
def parse_problem(input_text: str, **kwargs) -> ConstraintGraph:
    """
    Función de conveniencia para parsear y construir problema.
    
    Args:
        input_text: Especificación del problema
        **kwargs: Argumentos para InferenceEngine
    
    Returns:
        ConstraintGraph listo para resolver
    
    Examples:
        >>> cg = parse_problem(json_spec)
        >>> solver = AdaptiveConsistencyEngine()
        >>> stats = solver.solve(cg)
    """
    engine = InferenceEngine(**kwargs)
    return engine.parse_and_build(input_text)
```

### 7.2 CLI (Command Line Interface)

```python
# lattice_weaver/inference/cli.py

import argparse
import sys
import json
from pathlib import Path

from lattice_weaver.inference import InferenceEngine
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine


def main():
    """CLI principal del Inference Engine."""
    parser = argparse.ArgumentParser(
        description="LatticeWeaver Inference Engine - Parse and solve CSP problems"
    )
    
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input file with problem specification (JSON, YAML, or text)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for solutions (JSON)"
    )
    
    parser.add_argument(
        "--max-solutions", "-n",
        type=int,
        default=1,
        help="Maximum number of solutions to find (default: 1)"
    )
    
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        help="Timeout in seconds (default: no limit)"
    )
    
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Disable constraint inference"
    )
    
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable semantic validation"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Leer archivo de entrada
    if not args.input_file.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)
    
    input_text = args.input_file.read_text()
    
    # Crear motor de inferencia
    engine = InferenceEngine(
        enable_inference=not args.no_inference,
        enable_validation=not args.no_validation
    )
    
    try:
        # Parse y build
        if args.verbose:
            print("Parsing problem specification...")
        
        cg = engine.parse_and_build(input_text)
        
        if args.verbose:
            print(f"Problem parsed successfully:")
            print(f"  Variables: {len(cg.get_all_variables())}")
            print(f"  Constraints: {len(cg.get_all_constraints())}")
            print()
        
        # Resolver
        if args.verbose:
            print("Solving problem...")
        
        solver = AdaptiveConsistencyEngine()
        stats = solver.solve(cg, max_solutions=args.max_solutions, timeout=args.timeout)
        
        # Mostrar resultados
        print(f"Solutions found: {len(stats.solutions)}")
        print(f"Nodes explored: {stats.nodes_explored}")
        print(f"Backtracks: {stats.backtracks}")
        print(f"Time elapsed: {stats.time_elapsed:.4f}s")
        print()
        
        for i, solution in enumerate(stats.solutions, 1):
            print(f"Solution {i}:")
            for var, val in sorted(solution.items()):
                print(f"  {var} = {val}")
            print()
        
        # Guardar a archivo si se especifica
        if args.output:
            output_data = {
                "solutions": stats.solutions,
                "statistics": {
                    "nodes_explored": stats.nodes_explored,
                    "backtracks": stats.backtracks,
                    "time_elapsed": stats.time_elapsed,
                }
            }
            args.output.write_text(json.dumps(output_data, indent=2))
            if args.verbose:
                print(f"Results saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 7.3 API REST (para Track E)

```python
# lattice_weaver/inference/api.py

from flask import Flask, request, jsonify
from typing import Dict, Any
import uuid
import threading
import time

from lattice_weaver.inference import InferenceEngine
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine


app = Flask(__name__)

# Almacenamiento en memoria de jobs (v1.0)
# v2.0: Usar Redis o base de datos
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


@app.route('/api/inference/parse', methods=['POST'])
def parse_problem():
    """
    Parsea una especificación de problema a IR.
    
    Request body:
        {
            "specification": "...",  # JSON, YAML, o texto
            "enable_inference": true,
            "enable_validation": true
        }
    
    Response:
        {
            "problem_ir": {...},
            "validation_result": {...}
        }
    """
    try:
        data = request.get_json()
        
        if 'specification' not in data:
            return jsonify({"error": "Missing 'specification' field"}), 400
        
        specification = data['specification']
        enable_inference = data.get('enable_inference', True)
        enable_validation = data.get('enable_validation', True)
        
        # Crear engine
        engine = InferenceEngine(
            enable_inference=enable_inference,
            enable_validation=enable_validation
        )
        
        # Parse
        problem_ir = engine.parse(specification)
        
        # Infer
        enhanced_ir = engine.infer(problem_ir)
        
        # Validate
        validation_result = engine.validate(enhanced_ir)
        
        # Serializar IR (simplificado)
        ir_dict = {
            "name": enhanced_ir.name,
            "description": enhanced_ir.description,
            "variables": {
                name: {
                    "type": var.var_type.value,
                    "domain": list(var.domain) if var.domain else None,
                    "domain_range": var.domain_range,
                }
                for name, var in enhanced_ir.variables.items()
            },
            "constraints": len(enhanced_ir.constraints),
            "inferred_constraints": len(enhanced_ir.inferred_constraints),
        }
        
        return jsonify({
            "problem_ir": ir_dict,
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/inference/solve', methods=['POST'])
def solve_problem():
    """
    Parsea y resuelve un problema de forma asíncrona.
    
    Request body:
        {
            "specification": "...",
            "max_solutions": 1,
            "timeout": null,
            "enable_inference": true,
            "enable_validation": true
        }
    
    Response:
        {
            "job_id": "uuid",
            "status": "pending"
        }
    """
    try:
        data = request.get_json()
        
        if 'specification' not in data:
            return jsonify({"error": "Missing 'specification' field"}), 400
        
        # Crear job
        job_id = str(uuid.uuid4())
        
        with jobs_lock:
            jobs[job_id] = {
                "status": "pending",
                "created_at": time.time(),
                "specification": data['specification'],
                "max_solutions": data.get('max_solutions', 1),
                "timeout": data.get('timeout'),
                "enable_inference": data.get('enable_inference', True),
                "enable_validation": data.get('enable_validation', True),
                "result": None,
                "error": None,
            }
        
        # Ejecutar en thread separado
        thread = threading.Thread(target=_solve_job, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "pending"
        }), 202
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/inference/status/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """
    Obtiene el estado de un job.
    
    Response:
        {
            "job_id": "uuid",
            "status": "pending" | "running" | "completed" | "failed",
            "result": {...} | null,
            "error": "..." | null
        }
    """
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
        
        job = jobs[job_id]
    
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    })


def _solve_job(job_id: str):
    """Ejecuta un job de resolución."""
    with jobs_lock:
        job = jobs[job_id]
        job["status"] = "running"
    
    try:
        # Crear engine
        engine = InferenceEngine(
            enable_inference=job["enable_inference"],
            enable_validation=job["enable_validation"]
        )
        
        # Parse y build
        cg = engine.parse_and_build(job["specification"])
        
        # Resolver
        solver = AdaptiveConsistencyEngine()
        stats = solver.solve(
            cg,
            max_solutions=job["max_solutions"],
            timeout=job["timeout"]
        )
        
        # Guardar resultado
        with jobs_lock:
            job["status"] = "completed"
            job["result"] = {
                "solutions": stats.solutions,
                "statistics": {
                    "nodes_explored": stats.nodes_explored,
                    "backtracks": stats.backtracks,
                    "time_elapsed": stats.time_elapsed,
                }
            }
    
    except Exception as e:
        with jobs_lock:
            job["status"] = "failed"
            job["error"] = str(e)


def run_api(host='0.0.0.0', port=5001):
    """Inicia el servidor API."""
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    run_api()
```

---

## 8. Estrategias de Implementación

### 8.1 Principios de Diseño Aplicados

#### Economía Computacional
- **Parser:** Usar parsers incrementales, no rehacer trabajo
- **Inference:** Solo inferir restricciones que mejoren la resolución
- **Validation:** Validaciones rápidas primero (fail-fast)

#### Modularidad
- Cada componente (Parser, Inferencer, Builder) es independiente
- Interfaces claras entre capas
- Fácil añadir nuevos parsers o estrategias de inferencia

#### No Redundancia
- IR única compartida por todos los parsers
- Constraint Factory centralizada
- Reutilizar lógica del Track A (ConstraintGraph, ACE)

#### Testeabilidad
- Cada componente testeable de forma aislada
- Inyección de dependencias (parsers, validators)
- Mocks para testing de API REST

### 8.2 Gestión de Errores

**Principio:** "Falla rápido, falla ruidosamente, falla con contexto"

```python
# Jerarquía de excepciones
class InferenceError(Exception):
    """Excepción base del Inference Engine."""
    pass

class ParseError(InferenceError):
    """Error durante el parsing."""
    pass

class ValidationError(InferenceError):
    """Error durante la validación."""
    pass

class BuildError(InferenceError):
    """Error durante la construcción del grafo."""
    pass

class InferenceEngineError(InferenceError):
    """Error general del motor de inferencia."""
    pass
```

### 8.3 Logging y Debugging

```python
import logging

logger = logging.getLogger('lattice_weaver.inference')

# En cada componente:
logger.debug(f"Parsing specification with {parser.__class__.__name__}")
logger.info(f"Inferred {len(inferred_constraints)} additional constraints")
logger.warning(f"Variable '{var_name}' has no constraints")
logger.error(f"Failed to build ConstraintGraph: {error}")
```

---

## 9. Plan de Fases

### Fase 1: Fundamentos (Semanas 1-2)

**Objetivos:**
- Implementar IR (ProblemIR, Variable, Constraint)
- Implementar JSONParser
- Implementar ConstraintGraphBuilder básico
- Tests unitarios de componentes base

**Entregables:**
- `lattice_weaver/inference/ir.py`
- `lattice_weaver/inference/parsers/json_parser.py`
- `lattice_weaver/inference/builders/graph_builder.py`
- `tests/unit/test_ir.py`
- `tests/unit/test_json_parser.py`
- `tests/unit/test_graph_builder.py`

**Criterio de éxito:**
- Parsear JSON simple a IR
- Construir ConstraintGraph desde IR
- 100% tests pasando

### Fase 2: Parsers Adicionales (Semanas 3-4)

**Objetivos:**
- Implementar YAMLParser
- Implementar TextParser (patrones básicos)
- Extender JSONParser con más tipos de restricciones
- Tests de integración parser → builder

**Entregables:**
- `lattice_weaver/inference/parsers/yaml_parser.py`
- `lattice_weaver/inference/parsers/text_parser.py`
- `tests/unit/test_yaml_parser.py`
- `tests/unit/test_text_parser.py`
- `tests/integration/test_parsing_pipeline.py`

**Criterio de éxito:**
- Parsear YAML y texto simple
- Soportar 5+ tipos de problemas (N-Queens, Graph Coloring, etc.)
- 90%+ cobertura de tests

### Fase 3: Inference Layer (Semanas 5-6)

**Objetivos:**
- Implementar ConstraintInferencer
- Implementar StructureDetector
- Integrar inferencia en InferenceEngine
- Tests de inferencia

**Entregables:**
- `lattice_weaver/inference/inference/constraint_inferencer.py`
- `lattice_weaver/inference/inference/structure_detector.py`
- `tests/unit/test_constraint_inferencer.py`
- `tests/unit/test_structure_detector.py`

**Criterio de éxito:**
- Detectar AllDifferent automáticamente
- Detectar patrones estructurales (bipartito, árbol)
- Mejorar eficiencia de resolución en 20%+ en problemas con AllDifferent

### Fase 4: Validation y CLI (Semana 7)

**Objetivos:**
- Implementar SemanticValidator
- Implementar CLI
- Documentación de usuario
- Tests end-to-end

**Entregables:**
- `lattice_weaver/inference/validation/semantic_validator.py`
- `lattice_weaver/inference/cli.py`
- `docs/INFERENCE_ENGINE_USER_GUIDE.md`
- `tests/e2e/test_cli.py`

**Criterio de éxito:**
- Detectar 90%+ de errores semánticos comunes
- CLI funcional con ejemplos
- Documentación completa

### Fase 5: API REST e Integración (Semana 8)

**Objetivos:**
- Implementar API REST
- Integración con Track E (preparación)
- Documentación de API
- Tests de API

**Entregables:**
- `lattice_weaver/inference/api.py`
- `docs/INFERENCE_ENGINE_API.md`
- `tests/integration/test_api.py`
- Ejemplos de uso con curl/Postman

**Criterio de éxito:**
- API REST funcional con 3 endpoints
- Resolución asíncrona de problemas
- Documentación de API completa
- Track E puede comenzar integración

---

## 10. Testing y Validación

### 10.1 Estrategia de Testing

**Pirámide de Testing:**
- **70% Tests Unitarios:** Cada componente aislado
- **20% Tests de Integración:** Flujo completo parser → builder → solver
- **10% Tests E2E:** CLI y API REST

### 10.2 Tests Unitarios

```python
# tests/unit/test_json_parser.py

def test_parse_simple_problem():
    """Test parsing de problema simple."""
    spec = """
    {
        "name": "Simple Problem",
        "variables": [
            {"name": "X", "type": "integer", "domain": [1, 2, 3]},
            {"name": "Y", "type": "integer", "domain": [1, 2, 3]}
        ],
        "constraints": [
            {"type": "binary", "variables": ["X", "Y"], "relation": "!="}
        ]
    }
    """
    
    parser = JSONParser()
    problem_ir = parser.parse(spec)
    
    assert problem_ir.name == "Simple Problem"
    assert len(problem_ir.variables) == 2
    assert len(problem_ir.constraints) == 1
    assert problem_ir.variables["X"].domain == {1, 2, 3}


def test_parse_invalid_json():
    """Test manejo de JSON inválido."""
    spec = "{ invalid json"
    
    parser = JSONParser()
    with pytest.raises(ParseError):
        parser.parse(spec)


def test_parse_missing_variables():
    """Test error cuando faltan variables."""
    spec = '{"name": "Test"}'
    
    parser = JSONParser()
    with pytest.raises(ParseError, match="Missing 'variables'"):
        parser.parse(spec)
```

### 10.3 Tests de Integración

```python
# tests/integration/test_parsing_pipeline.py

def test_full_pipeline_nqueens():
    """Test pipeline completo con N-Queens."""
    spec = """
    {
        "name": "4-Queens",
        "type": "n-queens",
        "variables": [
            {"name": "Q0", "type": "integer", "domain": [0, 1, 2, 3]},
            {"name": "Q1", "type": "integer", "domain": [0, 1, 2, 3]},
            {"name": "Q2", "type": "integer", "domain": [0, 1, 2, 3]},
            {"name": "Q3", "type": "integer", "domain": [0, 1, 2, 3]}
        ],
        "constraints": [
            {"type": "binary", "variables": ["Q0", "Q1"], "relation": "!="},
            {"type": "binary", "variables": ["Q0", "Q2"], "relation": "!="},
            {"type": "binary", "variables": ["Q0", "Q3"], "relation": "!="},
            {"type": "binary", "variables": ["Q1", "Q2"], "relation": "!="},
            {"type": "binary", "variables": ["Q1", "Q3"], "relation": "!="},
            {"type": "binary", "variables": ["Q2", "Q3"], "relation": "!="}
        ]
    }
    """
    
    # Parse y build
    engine = InferenceEngine()
    cg = engine.parse_and_build(spec)
    
    # Verificar ConstraintGraph
    assert len(cg.get_all_variables()) == 4
    assert len(cg.get_all_constraints()) >= 6
    
    # Resolver
    solver = AdaptiveConsistencyEngine()
    stats = solver.solve(cg, max_solutions=2)
    
    # Verificar soluciones
    assert len(stats.solutions) == 2
    
    # Verificar que las soluciones son válidas
    for solution in stats.solutions:
        # Todas diferentes
        values = list(solution.values())
        assert len(values) == len(set(values))
```

### 10.4 Tests E2E

```python
# tests/e2e/test_cli.py

def test_cli_solve_from_file(tmp_path):
    """Test CLI resolviendo desde archivo."""
    # Crear archivo de especificación
    spec_file = tmp_path / "problem.json"
    spec_file.write_text("""
    {
        "name": "Simple Problem",
        "variables": [
            {"name": "X", "type": "integer", "domain": [1, 2]},
            {"name": "Y", "type": "integer", "domain": [1, 2]}
        ],
        "constraints": [
            {"type": "binary", "variables": ["X", "Y"], "relation": "!="}
        ]
    }
    """)
    
    # Ejecutar CLI
    result = subprocess.run(
        ["python", "-m", "lattice_weaver.inference.cli", str(spec_file)],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Solutions found: 2" in result.stdout
```

### 10.5 Benchmarks

```python
# tests/benchmarks/test_inference_performance.py

def test_parsing_performance():
    """Benchmark de parsing."""
    spec = generate_large_problem_spec(n_vars=100, n_constraints=500)
    
    engine = InferenceEngine()
    
    start = time.time()
    problem_ir = engine.parse(spec)
    elapsed = time.time() - start
    
    assert elapsed < 0.1  # < 100ms
    assert len(problem_ir.variables) == 100


def test_inference_performance():
    """Benchmark de inferencia."""
    problem_ir = create_problem_with_alldifferent_pattern(n_vars=50)
    
    engine = InferenceEngine()
    
    start = time.time()
    enhanced_ir = engine.infer(problem_ir)
    elapsed = time.time() - start
    
    assert elapsed < 0.5  # < 500ms
    assert len(enhanced_ir.inferred_constraints) > 0
```

---

## 11. Principios de Diseño Aplicados

### 11.1 Economía Computacional

- **Parser:** Parsing incremental, no reconstruir estructuras
- **Inference:** Solo inferir restricciones útiles (costo-beneficio)
- **Validation:** Validaciones baratas primero (fail-fast)

### 11.2 Modularidad y Composición

- Cada parser es independiente (JSONParser, YAMLParser, TextParser)
- Inference Layer separado de Parsing y Building
- Fácil añadir nuevos tipos de parsers o estrategias de inferencia

### 11.3 No Redundancia

- IR única compartida por todos los parsers
- ConstraintFactory centralizada
- Reutilización del código del Track A (ConstraintGraph, ACE)

### 11.4 Testeabilidad

- Inyección de dependencias (parsers, validators)
- Cada componente testeable aisladamente
- Mocks para testing de API REST

### 11.5 Inmutabilidad

- `ProblemIR` y `EnhancedProblemIR` son dataclasses
- Dominios de variables son sets (inmutables en contexto)
- Funciones puras en ConstraintFactory

### 11.6 Fail Fast

- Validación agresiva en parsing
- Excepciones específicas con contexto
- No silenciar errores

---

## 12. Ejemplos de Uso

### 12.1 Ejemplo 1: N-Queens desde JSON

```python
from lattice_weaver.inference import parse_problem
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine

# Especificación JSON
spec = """
{
    "name": "4-Queens",
    "variables": [
        {"name": "Q0", "type": "integer", "domain": [0, 1, 2, 3]},
        {"name": "Q1", "type": "integer", "domain": [0, 1, 2, 3]},
        {"name": "Q2", "type": "integer", "domain": [0, 1, 2, 3]},
        {"name": "Q3", "type": "integer", "domain": [0, 1, 2, 3]}
    ],
    "constraints": [
        {"type": "alldifferent", "variables": ["Q0", "Q1", "Q2", "Q3"]}
    ]
}
"""

# Parse y build
cg = parse_problem(spec)

# Resolver
solver = AdaptiveConsistencyEngine()
stats = solver.solve(cg, max_solutions=2)

# Mostrar soluciones
for i, solution in enumerate(stats.solutions, 1):
    print(f"Solution {i}: {solution}")
```

### 12.2 Ejemplo 2: Texto Natural

```python
from lattice_weaver.inference import parse_problem

# Especificación en texto
spec = "4 reinas en tablero 4x4"

# Parse y build (automático)
cg = parse_problem(spec)

# Resolver
solver = AdaptiveConsistencyEngine()
stats = solver.solve(cg)

print(f"Found {len(stats.solutions)} solutions")
```

### 12.3 Ejemplo 3: YAML

```python
from lattice_weaver.inference import parse_problem

# Especificación YAML
spec = """
name: Graph Coloring
variables:
  - name: Node1
    type: categorical
    domain: [red, green, blue]
  - name: Node2
    type: categorical
    domain: [red, green, blue]
  - name: Node3
    type: categorical
    domain: [red, green, blue]
constraints:
  - type: binary
    variables: [Node1, Node2]
    relation: "!="
  - type: binary
    variables: [Node2, Node3]
    relation: "!="
"""

cg = parse_problem(spec)
solver = AdaptiveConsistencyEngine()
stats = solver.solve(cg)
```

### 12.4 Ejemplo 4: CLI

```bash
# Crear archivo de especificación
cat > problem.json << EOF
{
    "name": "Simple CSP",
    "variables": [
        {"name": "X", "type": "integer", "domain": [1, 2, 3]},
        {"name": "Y", "type": "integer", "domain": [1, 2, 3]}
    ],
    "constraints": [
        {"type": "binary", "variables": ["X", "Y"], "relation": "!="}
    ]
}
EOF

# Resolver con CLI
python -m lattice_weaver.inference.cli problem.json --max-solutions 5 --verbose

# Salida:
# Parsing problem specification...
# Problem parsed successfully:
#   Variables: 2
#   Constraints: 1
#
# Solving problem...
# Solutions found: 5
# Nodes explored: 8
# Backtracks: 2
# Time elapsed: 0.0023s
#
# Solution 1:
#   X = 1
#   Y = 2
# ...
```

### 12.5 Ejemplo 5: API REST

```bash
# Iniciar servidor
python -m lattice_weaver.inference.api

# En otra terminal, enviar request
curl -X POST http://localhost:5001/api/inference/solve \
  -H "Content-Type: application/json" \
  -d '{
    "specification": "{\"name\": \"Test\", \"variables\": [...], \"constraints\": [...]}",
    "max_solutions": 2
  }'

# Respuesta:
# {
#   "job_id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "pending"
# }

# Consultar estado
curl http://localhost:5001/api/inference/status/550e8400-e29b-41d4-a716-446655440000

# Respuesta:
# {
#   "job_id": "550e8400-e29b-41d4-a716-446655440000",
#   "status": "completed",
#   "result": {
#     "solutions": [...],
#     "statistics": {...}
#   },
#   "error": null
# }
```

---

## 13. Riesgos y Mitigaciones

### 13.1 Riesgo: Parsing de Lenguaje Natural Complejo

**Descripción:** Lenguaje natural es ambiguo y difícil de parsear sin ML.

**Impacto:** Alto (funcionalidad clave)

**Probabilidad:** Alta

**Mitigación:**
- v1.0: Patrones fijos con regex (limitado pero funcional)
- v2.0: Integrar NLP con modelos pre-entrenados (spaCy, Transformers)
- Documentar claramente limitaciones de v1.0
- Proveer ejemplos de especificaciones soportadas

### 13.2 Riesgo: Inferencia Incorrecta de Restricciones

**Descripción:** Inferir restricciones incorrectas puede hacer el problema insatisfacible.

**Impacto:** Crítico (resultados incorrectos)

**Probabilidad:** Media

**Mitigación:**
- Tests exhaustivos de inferencia
- Validación semántica post-inferencia
- Logging detallado de restricciones inferidas
- Opción para deshabilitar inferencia
- Revisión manual de restricciones inferidas en modo verbose

### 13.3 Riesgo: Performance de Parsing

**Descripción:** Parsing lento puede bloquear resolución.

**Impacto:** Medio (experiencia de usuario)

**Probabilidad:** Baja

**Mitigación:**
- Benchmarks de parsing en CI/CD
- Parsing incremental y lazy
- Caché de especificaciones parseadas
- Timeout en parsing

### 13.4 Riesgo: Incompatibilidad con Track E

**Descripción:** API REST no cumple expectativas del Track E.

**Impacto:** Alto (bloquea Track E)

**Probabilidad:** Baja

**Mitigación:**
- Definir contrato de API temprano (Semana 1)
- Revisión con equipo del Track E
- Tests de integración con mock del frontend
- Documentación exhaustiva de API

---

## 14. Conclusión

El **Track D - Inference Engine** es un componente crítico que democratiza el acceso a LatticeWeaver, permitiendo especificar problemas de forma textual en lugar de programática. El diseño propuesto es modular, extensible y sigue los principios de diseño de LatticeWeaver.

### Próximos Pasos

1. **Validación de este diseño** por el usuario
2. **Inicio de Fase 1:** Implementación de IR y parsers básicos
3. **Iteración incremental** siguiendo el plan de fases
4. **Sincronización con Track E** para preparar integración

---

**Documento preparado por:** agent-track-d  
**Fecha:** 12 de Octubre, 2025  
**Estado:** Pendiente de validación

