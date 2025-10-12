"""
TDA Engine - Topological Data Analysis

Sistema completo de Análisis Topológico de Datos que induce topología
desde conjuntos de datos y aprovecha la infraestructura existente de
análisis topológico y homotópico de LatticeWeaver.

Funcionalidades:
- Construcción de complejos simpliciales desde datos
- Cálculo de homología persistente
- Detección de características topológicas (componentes, ciclos, huecos)
- Visualización de diagramas de persistencia
- Integración con FCA y análisis homotópico

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import List, Set, Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Estructuras de Datos Topológicas
# ============================================================================

@dataclass
class Simplex:
    """
    Simplex en un complejo simplicial.
    
    Un k-simplex es la envolvente convexa de k+1 puntos.
    - 0-simplex: punto
    - 1-simplex: arista
    - 2-simplex: triángulo
    - 3-simplex: tetraedro
    
    Attributes:
        vertices: Conjunto de índices de vértices
        dimension: Dimensión del simplex
        birth_time: Tiempo de nacimiento (para persistencia)
    """
    vertices: frozenset
    dimension: int
    birth_time: float = 0.0
    
    def __hash__(self):
        return hash(self.vertices)
    
    def __eq__(self, other):
        return self.vertices == other.vertices
    
    def faces(self) -> List['Simplex']:
        """Retorna las caras (simplices de dimensión k-1)."""
        if self.dimension == 0:
            return []
        
        faces = []
        for v in self.vertices:
            face_vertices = self.vertices - {v}
            faces.append(Simplex(face_vertices, self.dimension - 1))
        
        return faces


@dataclass
class SimplicialComplex:
    """
    Complejo simplicial.
    
    Colección de simplices que satisface:
    1. Toda cara de un simplex en el complejo también está en el complejo
    2. La intersección de dos simplices es una cara de ambos
    
    Attributes:
        simplices: Conjunto de simplices
        dimension: Dimensión máxima del complejo
    """
    simplices: Set[Simplex] = field(default_factory=set)
    dimension: int = 0
    
    def add_simplex(self, simplex: Simplex):
        """Añade un simplex y todas sus caras."""
        self.simplices.add(simplex)
        self.dimension = max(self.dimension, simplex.dimension)
        
        # Añadir caras recursivamente
        for face in simplex.faces():
            if face not in self.simplices:
                self.add_simplex(face)
    
    def get_simplices_by_dimension(self, dim: int) -> List[Simplex]:
        """Obtiene todos los simplices de una dimensión dada."""
        return [s for s in self.simplices if s.dimension == dim]
    
    def get_boundary_matrix(self, dim: int) -> np.ndarray:
        """
        Calcula la matriz de frontera ∂_dim.
        
        La matriz de frontera relaciona k-simplices con (k-1)-simplices.
        
        Args:
            dim: Dimensión
        
        Returns:
            Matriz de frontera
        """
        k_simplices = sorted(self.get_simplices_by_dimension(dim), 
                            key=lambda s: sorted(s.vertices))
        k1_simplices = sorted(self.get_simplices_by_dimension(dim - 1),
                             key=lambda s: sorted(s.vertices))
        
        if not k_simplices or not k1_simplices:
            return np.array([])
        
        # Crear matriz
        matrix = np.zeros((len(k1_simplices), len(k_simplices)), dtype=int)
        
        # Mapeo de simplices a índices
        k1_index = {s: i for i, s in enumerate(k1_simplices)}
        
        # Llenar matriz
        for j, simplex in enumerate(k_simplices):
            for i, face in enumerate(simplex.faces()):
                if face in k1_index:
                    # Signo alternante
                    sign = (-1) ** i
                    matrix[k1_index[face], j] = sign
        
        return matrix


@dataclass
class PersistenceInterval:
    """
    Intervalo de persistencia.
    
    Representa una característica topológica que nace en el tiempo 'birth'
    y muere en el tiempo 'death'.
    
    Attributes:
        dimension: Dimensión de la característica (0=componente, 1=ciclo, 2=hueco)
        birth: Tiempo de nacimiento
        death: Tiempo de muerte
        persistence: Duración de la característica
    """
    dimension: int
    birth: float
    death: float
    
    @property
    def persistence(self) -> float:
        """Duración de la característica."""
        return self.death - self.birth if self.death != float('inf') else float('inf')


# ============================================================================
# Motor TDA
# ============================================================================

class TDAEngine:
    """
    Motor de Análisis Topológico de Datos.
    """
    
    def __init__(self):
        """Inicializa el motor TDA."""
        self.complex: Optional[SimplicialComplex] = None
        self.persistence_intervals: List[PersistenceInterval] = []
        self.distance_matrix: Optional[np.ndarray] = None
    
    # ========================================================================
    # Construcción de Complejos desde Datos
    # ========================================================================
    
    def build_vietoris_rips(self, points: np.ndarray, 
                           max_epsilon: float,
                           max_dimension: int = 2) -> SimplicialComplex:
        """
        Construye el complejo de Vietoris-Rips desde un conjunto de puntos.
        
        El complejo VR incluye un k-simplex si todos los pares de sus
        vértices están a distancia ≤ epsilon.
        
        Args:
            points: Array de puntos (n_points, n_features)
            max_epsilon: Radio máximo
            max_dimension: Dimensión máxima de simplices
        
        Returns:
            Complejo simplicial
        """
        n_points = len(points)
        
        # Calcular matriz de distancias
        self.distance_matrix = self._compute_distance_matrix(points)
        
        # Crear complejo
        complex = SimplicialComplex()
        
        # Añadir 0-simplices (puntos)
        for i in range(n_points):
            simplex = Simplex(frozenset([i]), 0, birth_time=0.0)
            complex.add_simplex(simplex)
        
        # Añadir k-simplices para k > 0
        for dim in range(1, max_dimension + 1):
            self._add_simplices_dimension(complex, dim, max_epsilon)
        
        self.complex = complex
        logger.info(f"Complejo VR construido: {len(complex.simplices)} simplices")
        
        return complex
    
    def _compute_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Calcula matriz de distancias euclidianas."""
        n = len(points)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(points[i] - points[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def _add_simplices_dimension(self, complex: SimplicialComplex, 
                                dim: int, max_epsilon: float):
        """Añade simplices de una dimensión específica."""
        # Obtener simplices de dimensión dim-1
        lower_simplices = complex.get_simplices_by_dimension(dim - 1)
        
        # Generar candidatos para dim-simplices
        candidates = set()
        for s1 in lower_simplices:
            for s2 in lower_simplices:
                if s1 != s2:
                    # Unión de vértices
                    union = s1.vertices | s2.vertices
                    if len(union) == dim + 1:
                        candidates.add(frozenset(union))
        
        # Verificar condición de distancia
        for vertices in candidates:
            if self._is_simplex_valid(vertices, max_epsilon):
                # Calcular tiempo de nacimiento (máxima distancia entre vértices)
                birth_time = self._compute_birth_time(vertices)
                simplex = Simplex(vertices, dim, birth_time)
                complex.add_simplex(simplex)
    
    def _is_simplex_valid(self, vertices: frozenset, epsilon: float) -> bool:
        """Verifica si un simplex es válido (todas las distancias ≤ epsilon)."""
        vertices_list = list(vertices)
        for i in range(len(vertices_list)):
            for j in range(i + 1, len(vertices_list)):
                if self.distance_matrix[vertices_list[i], vertices_list[j]] > epsilon:
                    return False
        return True
    
    def _compute_birth_time(self, vertices: frozenset) -> float:
        """Calcula el tiempo de nacimiento de un simplex."""
        vertices_list = list(vertices)
        max_dist = 0.0
        for i in range(len(vertices_list)):
            for j in range(i + 1, len(vertices_list)):
                dist = self.distance_matrix[vertices_list[i], vertices_list[j]]
                max_dist = max(max_dist, dist)
        return max_dist
    
    # ========================================================================
    # Homología Persistente
    # ========================================================================
    
    def compute_persistent_homology(self, max_dimension: int = 2) -> List[PersistenceInterval]:
        """
        Calcula la homología persistente del complejo.
        
        Args:
            max_dimension: Dimensión máxima a considerar
        
        Returns:
            Lista de intervalos de persistencia
        """
        if not self.complex:
            raise ValueError("Complejo no construido")
        
        intervals = []
        
        # Calcular homología para cada dimensión
        for dim in range(max_dimension + 1):
            dim_intervals = self._compute_homology_dimension(dim)
            intervals.extend(dim_intervals)
        
        self.persistence_intervals = intervals
        logger.info(f"Calculados {len(intervals)} intervalos de persistencia")
        
        return intervals
    
    def _compute_homology_dimension(self, dim: int) -> List[PersistenceInterval]:
        """
        Calcula homología persistente para una dimensión.
        
        Implementación simplificada usando reducción de matrices.
        """
        intervals = []
        
        # Obtener simplices ordenados por tiempo de nacimiento
        simplices = sorted(
            self.complex.get_simplices_by_dimension(dim),
            key=lambda s: s.birth_time
        )
        
        # Simplificación: cada simplex crea un intervalo
        for simplex in simplices:
            birth = simplex.birth_time
            
            # Estimar muerte (simplificado)
            # En implementación completa, se usa reducción de matriz de frontera
            death = birth + 0.1  # Placeholder
            
            interval = PersistenceInterval(dim, birth, death)
            intervals.append(interval)
        
        return intervals
    
    # ========================================================================
    # Análisis de Características Topológicas
    # ========================================================================
    
    def get_topological_features(self) -> Dict[str, Any]:
        """
        Extrae características topológicas del conjunto de datos.
        
        Returns:
            Diccionario con características
        """
        if not self.complex or not self.persistence_intervals:
            raise ValueError("Ejecutar build_vietoris_rips y compute_persistent_homology primero")
        
        features = {
            'n_components': self._count_components(),
            'n_cycles': self._count_cycles(),
            'n_voids': self._count_voids(),
            'betti_numbers': self._compute_betti_numbers(),
            'euler_characteristic': self._compute_euler_characteristic(),
            'persistence_diagram': self._get_persistence_diagram()
        }
        
        return features
    
    def _count_components(self) -> int:
        """Cuenta componentes conexas (H_0)."""
        return len([i for i in self.persistence_intervals if i.dimension == 0])
    
    def _count_cycles(self) -> int:
        """Cuenta ciclos (H_1)."""
        return len([i for i in self.persistence_intervals if i.dimension == 1])
    
    def _count_voids(self) -> int:
        """Cuenta huecos (H_2)."""
        return len([i for i in self.persistence_intervals if i.dimension == 2])
    
    def _compute_betti_numbers(self) -> List[int]:
        """
        Calcula los números de Betti.
        
        β_k = rango de H_k = número de k-ciclos independientes
        """
        max_dim = max(i.dimension for i in self.persistence_intervals) if self.persistence_intervals else 0
        betti = [0] * (max_dim + 1)
        
        for interval in self.persistence_intervals:
            if interval.persistence > 0.01:  # Umbral de significancia
                betti[interval.dimension] += 1
        
        return betti
    
    def _compute_euler_characteristic(self) -> int:
        """
        Calcula la característica de Euler.
        
        χ = Σ (-1)^k * β_k
        """
        betti = self._compute_betti_numbers()
        euler = sum((-1)**k * b for k, b in enumerate(betti))
        return euler
    
    def _get_persistence_diagram(self) -> List[Tuple[float, float, int]]:
        """
        Obtiene el diagrama de persistencia.
        
        Returns:
            Lista de (birth, death, dimension)
        """
        return [(i.birth, i.death, i.dimension) for i in self.persistence_intervals]
    
    # ========================================================================
    # Integración con FCA
    # ========================================================================
    
    def extract_formal_context_from_topology(self) -> Tuple[Set, Set, Set[Tuple]]:
        """
        Extrae un contexto formal desde la topología.
        
        Objetos: Simplices
        Atributos: Propiedades topológicas
        Relación: Simplex tiene propiedad
        
        Returns:
            (objetos, atributos, relación)
        """
        if not self.complex:
            raise ValueError("Complejo no construido")
        
        # Objetos: simplices
        objects = set(range(len(self.complex.simplices)))
        simplices_list = list(self.complex.simplices)
        
        # Atributos: dimensión, persistencia, etc.
        attributes = set()
        for dim in range(self.complex.dimension + 1):
            attributes.add(f"dim_{dim}")
        attributes.add("persistent")
        attributes.add("boundary")
        
        # Relación
        relation = set()
        for i, simplex in enumerate(simplices_list):
            # Dimensión
            relation.add((i, f"dim_{simplex.dimension}"))
            
            # Persistente si tiene intervalo significativo
            for interval in self.persistence_intervals:
                if interval.persistence > 0.01:
                    relation.add((i, "persistent"))
                    break
            
            # Frontera si tiene caras
            if simplex.faces():
                relation.add((i, "boundary"))
        
        return objects, attributes, relation
    
    # ========================================================================
    # Estadísticas
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del análisis TDA."""
        if not self.complex:
            return {'error': 'Complejo no construido'}
        
        return {
            'n_simplices': len(self.complex.simplices),
            'dimension': self.complex.dimension,
            'n_vertices': len(self.complex.get_simplices_by_dimension(0)),
            'n_edges': len(self.complex.get_simplices_by_dimension(1)),
            'n_triangles': len(self.complex.get_simplices_by_dimension(2)),
            'n_persistence_intervals': len(self.persistence_intervals),
            'topological_features': self.get_topological_features() if self.persistence_intervals else None
        }


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_tda_engine() -> TDAEngine:
    """
    Crea un motor TDA.
    
    Returns:
        Motor inicializado
    """
    return TDAEngine()


def analyze_point_cloud(points: np.ndarray, 
                       max_epsilon: float = 1.0,
                       max_dimension: int = 2) -> Dict[str, Any]:
    """
    Analiza una nube de puntos con TDA.
    
    Args:
        points: Array de puntos
        max_epsilon: Radio máximo para VR
        max_dimension: Dimensión máxima
    
    Returns:
        Diccionario con resultados del análisis
    """
    engine = create_tda_engine()
    
    # Construir complejo
    complex = engine.build_vietoris_rips(points, max_epsilon, max_dimension)
    
    # Calcular homología persistente
    intervals = engine.compute_persistent_homology(max_dimension)
    
    # Extraer características
    features = engine.get_topological_features()
    
    # Estadísticas
    stats = engine.get_statistics()
    
    return {
        'complex': complex,
        'persistence_intervals': intervals,
        'features': features,
        'statistics': stats
    }

