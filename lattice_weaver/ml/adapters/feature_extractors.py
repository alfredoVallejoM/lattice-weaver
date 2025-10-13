"""
Capa 1: Feature Extractors

Convierte estructuras de LatticeWeaver en tensores ML.

Extractores implementados:
1. CSPFeatureExtractor - CSP → 18 dims
2. TDAFeatureExtractor - Point clouds/complexes → 32 dims
3. CubicalFeatureExtractor - Proof contexts → 24 dims
4. FCAFeatureExtractor - Formal contexts → 20 dims
5. HomotopyFeatureExtractor - Homotopy structures → 22 dims
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ============================================================================
# Base classes
# ============================================================================

class FeatureExtractor(ABC):
    """Base class for feature extractors."""
    
    @abstractmethod
    def extract(self, *args, **kwargs) -> torch.Tensor:
        """Extract features from input."""
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimensionality of extracted features."""
        pass


# ============================================================================
# 1. CSP Feature Extractor
# ============================================================================

@dataclass
class CSPState:
    """Simplified CSP state representation."""
    num_variables: int
    num_constraints: int
    domains: Dict[int, Set[int]]  # variable_id → domain
    constraint_graph: nx.Graph
    depth: int = 0
    num_backtracks: int = 0
    num_propagations: int = 0
    constraint_checks: int = 0
    time_elapsed_ms: float = 0.0


class CSPFeatureExtractor(FeatureExtractor):
    """
    Extrae features de estados CSP.
    
    Output: 18 dimensional feature vector
    """
    
    def __init__(self):
        self._feature_dim = 18
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract(self, csp_state: CSPState) -> torch.Tensor:
        """
        Extrae features de CSP state.
        
        Args:
            csp_state: Estado del CSP
        
        Returns:
            features: [18] tensor
        """
        features = []
        
        # 0-5: Características del problema
        features.append(float(csp_state.num_variables))
        features.append(float(csp_state.num_constraints))
        
        domain_sizes = [len(domain) for domain in csp_state.domains.values()]
        features.append(np.mean(domain_sizes) if domain_sizes else 0.0)
        features.append(float(max(domain_sizes)) if domain_sizes else 0.0)
        features.append(float(min(domain_sizes)) if domain_sizes else 0.0)
        features.append(float(np.std(domain_sizes)) if len(domain_sizes) > 1 else 0.0)
        
        # 6-9: Características del grafo
        graph = csp_state.constraint_graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0.0
        features.append(density)
        
        degrees = [d for _, d in graph.degree()]
        features.append(np.mean(degrees) if degrees else 0.0)
        features.append(float(max(degrees)) if degrees else 0.0)
        
        try:
            clustering = nx.average_clustering(graph)
        except:
            clustering = 0.0
        features.append(clustering)
        
        # 10-11: Características de restricciones
        constraint_tightness = num_edges / (num_nodes ** 2) if num_nodes > 0 else 0.0
        features.append(constraint_tightness)
        
        avg_arity = 2.0  # Asumimos restricciones binarias
        features.append(avg_arity)
        
        # 12-14: Estado de búsqueda
        features.append(float(csp_state.depth))
        features.append(float(csp_state.num_backtracks))
        features.append(float(csp_state.num_propagations))
        
        # 15-17: Métricas dinámicas
        domain_reduction_rate = 1.0 - (np.mean(domain_sizes) / 10.0) if domain_sizes else 0.0
        features.append(max(0.0, min(1.0, domain_reduction_rate)))
        
        features.append(float(csp_state.constraint_checks))
        features.append(float(csp_state.time_elapsed_ms))
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_batch(self, csp_states: List[CSPState]) -> torch.Tensor:
        """
        Extrae features de múltiples estados.
        
        Args:
            csp_states: Lista de estados CSP
        
        Returns:
            features: [batch_size, 18] tensor
        """
        return torch.stack([self.extract(state) for state in csp_states])


# ============================================================================
# 2. TDA Feature Extractor
# ============================================================================

@dataclass
class PointCloud:
    """Point cloud representation."""
    points: np.ndarray  # [N, d]
    
    @property
    def num_points(self) -> int:
        return len(self.points)
    
    @property
    def dimension(self) -> int:
        return self.points.shape[1] if len(self.points) > 0 else 0


@dataclass
class SimplicialComplex:
    """Simplicial complex representation."""
    simplices: Dict[int, List[Tuple]]  # dimension → list of simplices
    
    @property
    def num_simplices(self) -> Dict[int, int]:
        return {dim: len(simps) for dim, simps in self.simplices.items()}
    
    @property
    def max_dimension(self) -> int:
        return max(self.simplices.keys()) if self.simplices else 0


class TDAFeatureExtractor(FeatureExtractor):
    """
    Extrae features de point clouds y simplicial complexes.
    
    Output: 32 dimensional feature vector
    """
    
    def __init__(self):
        self._feature_dim = 32
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract_from_points(self, point_cloud: PointCloud) -> torch.Tensor:
        """
        Extrae features desde point cloud.
        
        Args:
            point_cloud: Point cloud
        
        Returns:
            features: [32] tensor
        """
        features = []
        
        points = point_cloud.points
        
        # 0-4: Características básicas
        features.append(float(point_cloud.num_points))
        features.append(float(point_cloud.dimension))
        
        # Diameter (distancia máxima)
        if len(points) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(points)
            diameter = float(np.max(distances))
            avg_distance = float(np.mean(distances))
            std_distance = float(np.std(distances))
        else:
            diameter = 0.0
            avg_distance = 0.0
            std_distance = 0.0
        
        features.append(diameter)
        features.append(avg_distance)
        features.append(std_distance)
        
        # 5-7: Betti numbers estimados (heurísticas simples)
        # Estos son placeholders - en producción usarías estimadores más sofisticados
        estimated_betti_0 = 1.0  # Asumimos conexo
        estimated_betti_1 = max(0.0, (len(points) - 10) / 100.0)  # Heurística
        estimated_betti_2 = 0.0
        
        features.append(estimated_betti_0)
        features.append(estimated_betti_1)
        features.append(estimated_betti_2)
        
        # 8-11: Características del complex (estimadas)
        features.append(float(len(points)))  # num_simplices_0
        features.append(float(len(points) * 3))  # num_simplices_1 (estimado)
        features.append(float(len(points) * 2))  # num_simplices_2 (estimado)
        
        euler = len(points) - len(points) * 3 + len(points) * 2
        features.append(float(euler))
        
        # 12-13: Densidad local
        if len(points) > 5:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, len(points))).fit(points)
            distances, _ = nbrs.kneighbors(points)
            local_density_mean = float(np.mean(distances))
            local_density_std = float(np.std(distances))
        else:
            local_density_mean = 0.0
            local_density_std = 0.0
        
        features.append(local_density_mean)
        features.append(local_density_std)
        
        # 14-16: Características de escala
        if len(points) > 1:
            min_edge = float(np.min(distances[:, 1]))  # Distancia al vecino más cercano
            max_edge = diameter
            avg_edge = avg_distance
        else:
            min_edge = 0.0
            max_edge = 0.0
            avg_edge = 0.0
        
        features.append(min_edge)
        features.append(max_edge)
        features.append(avg_edge)
        
        # 17-31: Padding con ceros (para llegar a 32 dims)
        features.extend([0.0] * 15)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_from_complex(self, complex: SimplicialComplex) -> torch.Tensor:
        """
        Extrae features desde simplicial complex.
        
        Args:
            complex: Simplicial complex
        
        Returns:
            features: [32] tensor
        """
        features = []
        
        # 0-1: Características básicas
        num_simplices = complex.num_simplices
        total_simplices = sum(num_simplices.values())
        features.append(float(total_simplices))
        features.append(float(complex.max_dimension))
        
        # 2-7: Número de simplices por dimensión
        for dim in range(6):
            features.append(float(num_simplices.get(dim, 0)))
        
        # 8-11: Euler characteristic y Betti numbers (estimados)
        euler = sum((-1) ** dim * count for dim, count in num_simplices.items())
        features.append(float(euler))
        
        # Betti numbers estimados (heurísticas)
        betti_0 = 1.0
        betti_1 = max(0.0, num_simplices.get(1, 0) - num_simplices.get(0, 0) + 1)
        betti_2 = 0.0
        
        features.append(betti_0)
        features.append(betti_1)
        features.append(betti_2)
        
        # 12-31: Padding
        features.extend([0.0] * 20)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract(self, obj: Any) -> torch.Tensor:
        """
        Extrae features (detecta automáticamente el tipo).
        
        Args:
            obj: PointCloud o SimplicialComplex
        
        Returns:
            features: [32] tensor
        """
        if isinstance(obj, PointCloud):
            return self.extract_from_points(obj)
        elif isinstance(obj, SimplicialComplex):
            return self.extract_from_complex(obj)
        else:
            raise TypeError(f"Unsupported type: {type(obj)}")


# ============================================================================
# 3. Cubical (Theorem Proving) Feature Extractor
# ============================================================================

@dataclass
class ProofContext:
    """Proof context representation."""
    goal_complexity: int
    goal_depth: int
    num_free_variables: int
    num_bound_variables: int
    num_hypotheses: int
    avg_hypothesis_complexity: float
    context_size: int
    num_definitions: int
    num_lemmas_available: int
    proof_depth: int
    num_subgoals: int
    num_tactics_tried: int


class CubicalFeatureExtractor(FeatureExtractor):
    """
    Extrae features de contextos de prueba de teoremas.
    
    Output: 24 dimensional feature vector
    """
    
    def __init__(self):
        self._feature_dim = 24
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract(self, proof_context: ProofContext) -> torch.Tensor:
        """
        Extrae features de proof context.
        
        Args:
            proof_context: Contexto de prueba
        
        Returns:
            features: [24] tensor
        """
        features = [
            float(proof_context.goal_complexity),
            float(proof_context.goal_depth),
            float(proof_context.num_free_variables),
            float(proof_context.num_bound_variables),
            float(proof_context.num_hypotheses),
            float(proof_context.avg_hypothesis_complexity),
            float(proof_context.context_size),
            float(proof_context.num_definitions),
            float(proof_context.num_lemmas_available),
            float(proof_context.proof_depth),
            float(proof_context.num_subgoals),
            float(proof_context.num_tactics_tried),
        ]
        
        # Padding to 24 dims
        features.extend([0.0] * 12)
        
        return torch.tensor(features, dtype=torch.float32)


# ============================================================================
# 4. FCA Feature Extractor
# ============================================================================

@dataclass
class FormalContext:
    """Formal context representation."""
    objects: Set[str]
    attributes: Set[str]
    incidence: Set[Tuple[str, str]]  # (object, attribute) pairs


class FCAFeatureExtractor(FeatureExtractor):
    """
    Extrae features de formal contexts.
    
    Output: 20 dimensional feature vector
    """
    
    def __init__(self):
        self._feature_dim = 20
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract(self, context: FormalContext) -> torch.Tensor:
        """
        Extrae features de formal context.
        
        Args:
            context: Formal context
        
        Returns:
            features: [20] tensor
        """
        features = []
        
        num_objects = len(context.objects)
        num_attributes = len(context.attributes)
        num_incidences = len(context.incidence)
        
        # 0-3: Características básicas
        features.append(float(num_objects))
        features.append(float(num_attributes))
        features.append(float(num_incidences))
        
        density = num_incidences / (num_objects * num_attributes) if num_objects > 0 and num_attributes > 0 else 0.0
        features.append(density)
        
        # 4-7: Distribución
        object_attr_counts = {}
        attr_object_counts = {}
        
        for obj, attr in context.incidence:
            object_attr_counts[obj] = object_attr_counts.get(obj, 0) + 1
            attr_object_counts[attr] = attr_object_counts.get(attr, 0) + 1
        
        obj_counts = list(object_attr_counts.values())
        attr_counts = list(attr_object_counts.values())
        
        features.append(np.mean(obj_counts) if obj_counts else 0.0)
        features.append(np.std(obj_counts) if len(obj_counts) > 1 else 0.0)
        features.append(np.mean(attr_counts) if attr_counts else 0.0)
        features.append(np.std(attr_counts) if len(attr_counts) > 1 else 0.0)
        
        # 8-9: Estimaciones estructurales
        # Número estimado de conceptos (log scale)
        estimated_concepts = min(num_objects, num_attributes) * 2
        features.append(np.log10(estimated_concepts + 1))
        
        # Altura estimada del lattice
        estimated_height = np.log2(estimated_concepts + 1)
        features.append(estimated_height)
        
        # 10-19: Padding
        features.extend([0.0] * 10)
        
        return torch.tensor(features, dtype=torch.float32)


# ============================================================================
# 5. Homotopy Feature Extractor
# ============================================================================

@dataclass
class HomotopyStructure:
    """Homotopy structure representation."""
    num_cells: Dict[int, int]  # dimension → count
    dimension: int
    euler_characteristic: int
    fundamental_group_rank: int


class HomotopyFeatureExtractor(FeatureExtractor):
    """
    Extrae features de estructuras homotópicas.
    
    Output: 22 dimensional feature vector
    """
    
    def __init__(self):
        self._feature_dim = 22
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def extract(self, structure: HomotopyStructure) -> torch.Tensor:
        """
        Extrae features de homotopy structure.
        
        Args:
            structure: Homotopy structure
        
        Returns:
            features: [22] tensor
        """
        features = []
        
        # 0-1: Características básicas
        total_cells = sum(structure.num_cells.values())
        features.append(float(total_cells))
        features.append(float(structure.dimension))
        
        # 2-7: Células por dimensión
        for dim in range(6):
            features.append(float(structure.num_cells.get(dim, 0)))
        
        # 8-9: Invariantes topológicos
        features.append(float(structure.euler_characteristic))
        features.append(float(structure.fundamental_group_rank))
        
        # 10-21: Padding
        features.extend([0.0] * 12)
        
        return torch.tensor(features, dtype=torch.float32)


# ============================================================================
# Demo y tests
# ============================================================================

if __name__ == "__main__":
    print("=== Feature Extractors Demo ===\n")
    
    # 1. CSP Feature Extractor
    print("1. CSP Feature Extractor")
    
    # Create dummy CSP state
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    csp_state = CSPState(
        num_variables=4,
        num_constraints=3,
        domains={0: {1, 2, 3}, 1: {1, 2}, 2: {1, 2, 3, 4}, 3: {1}},
        constraint_graph=graph,
        depth=2,
        num_backtracks=1,
        num_propagations=5
    )
    
    csp_extractor = CSPFeatureExtractor()
    csp_features = csp_extractor.extract(csp_state)
    print(f"   Feature dim: {csp_extractor.feature_dim}")
    print(f"   Features shape: {csp_features.shape}")
    print(f"   First 5 features: {csp_features[:5].tolist()}")
    
    # 2. TDA Feature Extractor
    print("\n2. TDA Feature Extractor")
    
    points = np.random.randn(50, 3)
    point_cloud = PointCloud(points=points)
    
    tda_extractor = TDAFeatureExtractor()
    tda_features = tda_extractor.extract_from_points(point_cloud)
    print(f"   Feature dim: {tda_extractor.feature_dim}")
    print(f"   Features shape: {tda_features.shape}")
    print(f"   First 5 features: {tda_features[:5].tolist()}")
    
    # 3. Cubical Feature Extractor
    print("\n3. Cubical Feature Extractor")
    
    proof_context = ProofContext(
        goal_complexity=10,
        goal_depth=3,
        num_free_variables=2,
        num_bound_variables=1,
        num_hypotheses=5,
        avg_hypothesis_complexity=7.5,
        context_size=20,
        num_definitions=3,
        num_lemmas_available=15,
        proof_depth=2,
        num_subgoals=3,
        num_tactics_tried=8
    )
    
    cubical_extractor = CubicalFeatureExtractor()
    cubical_features = cubical_extractor.extract(proof_context)
    print(f"   Feature dim: {cubical_extractor.feature_dim}")
    print(f"   Features shape: {cubical_features.shape}")
    print(f"   First 5 features: {cubical_features[:5].tolist()}")
    
    # 4. FCA Feature Extractor
    print("\n4. FCA Feature Extractor")
    
    formal_context = FormalContext(
        objects={'o1', 'o2', 'o3', 'o4'},
        attributes={'a1', 'a2', 'a3'},
        incidence={('o1', 'a1'), ('o1', 'a2'), ('o2', 'a2'), ('o3', 'a3')}
    )
    
    fca_extractor = FCAFeatureExtractor()
    fca_features = fca_extractor.extract(formal_context)
    print(f"   Feature dim: {fca_extractor.feature_dim}")
    print(f"   Features shape: {fca_features.shape}")
    print(f"   First 5 features: {fca_features[:5].tolist()}")
    
    # 5. Homotopy Feature Extractor
    print("\n5. Homotopy Feature Extractor")
    
    homotopy_structure = HomotopyStructure(
        num_cells={0: 10, 1: 15, 2: 8, 3: 2},
        dimension=3,
        euler_characteristic=5,
        fundamental_group_rank=2
    )
    
    homotopy_extractor = HomotopyFeatureExtractor()
    homotopy_features = homotopy_extractor.extract(homotopy_structure)
    print(f"   Feature dim: {homotopy_extractor.feature_dim}")
    print(f"   Features shape: {homotopy_features.shape}")
    print(f"   First 5 features: {homotopy_features[:5].tolist()}")
    
    print("\n=== All extractors functional ===")

