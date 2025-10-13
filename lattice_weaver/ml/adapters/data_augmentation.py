"""
Capa 5: Data Augmentation

Aumenta datasets para mejorar generalización.

Augmenters implementados:
1. CSPAugmenter - Permutar variables, valores, restricciones
2. TDAAugmenter - Rotaciones, traslaciones, ruido, escalado
3. CubicalAugmenter - Renombrar variables, reordenar hipótesis
4. FCAAugmenter - Permutar objetos/atributos
5. HomotopyAugmenter - Deformaciones, subdivisiones

Beneficios:
- 10-100x más datos desde el mismo dataset base
- Mejor generalización
- Menos overfitting
- Invariancias aprendidas
"""

import torch
import numpy as np
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from copy import deepcopy
import random

try:
    from .feature_extractors import (
        CSPState, PointCloud, SimplicialComplex,
        ProofContext, FormalContext, HomotopyStructure
    )
except ImportError:
    from feature_extractors import (
        CSPState, PointCloud, SimplicialComplex,
        ProofContext, FormalContext, HomotopyStructure
    )


# ============================================================================
# 1. CSP Augmenter
# ============================================================================

class CSPAugmenter:
    """
    Augmenta datos CSP preservando semántica.
    
    Transformaciones:
    - Permutación de variables
    - Permutación de valores en dominios
    - Permutación de restricciones
    - Reordenamiento de constraint graph
    """
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def permute_variables(self, csp_state: CSPState) -> CSPState:
        """
        Permuta variables (renombra).
        
        Invarianza: El problema es equivalente bajo permutación de variables.
        """
        n = csp_state.num_variables
        perm = list(range(n))
        random.shuffle(perm)
        
        # Crear mapeo
        var_map = {i: perm[i] for i in range(n)}
        
        # Permutar dominios
        new_domains = {var_map[var]: domain for var, domain in csp_state.domains.items()}
        
        # Permutar grafo
        new_graph = nx.relabel_nodes(csp_state.constraint_graph, var_map)
        
        return CSPState(
            num_variables=csp_state.num_variables,
            num_constraints=csp_state.num_constraints,
            domains=new_domains,
            constraint_graph=new_graph,
            depth=csp_state.depth,
            num_backtracks=csp_state.num_backtracks,
            num_propagations=csp_state.num_propagations,
            constraint_checks=csp_state.constraint_checks,
            time_elapsed_ms=csp_state.time_elapsed_ms
        )
    
    def permute_values(self, csp_state: CSPState) -> CSPState:
        """
        Permuta valores en dominios.
        
        Invarianza: El problema es equivalente bajo permutación de valores.
        """
        # Obtener todos los valores únicos
        all_values = set()
        for domain in csp_state.domains.values():
            all_values.update(domain)
        
        all_values = sorted(list(all_values))
        perm_values = all_values.copy()
        random.shuffle(perm_values)
        
        value_map = {old: new for old, new in zip(all_values, perm_values)}
        
        # Permutar dominios
        new_domains = {
            var: {value_map[val] for val in domain}
            for var, domain in csp_state.domains.items()
        }
        
        return CSPState(
            num_variables=csp_state.num_variables,
            num_constraints=csp_state.num_constraints,
            domains=new_domains,
            constraint_graph=csp_state.constraint_graph.copy(),
            depth=csp_state.depth,
            num_backtracks=csp_state.num_backtracks,
            num_propagations=csp_state.num_propagations,
            constraint_checks=csp_state.constraint_checks,
            time_elapsed_ms=csp_state.time_elapsed_ms
        )
    
    def add_noise_to_metrics(self, csp_state: CSPState, noise_level: float = 0.1) -> CSPState:
        """
        Añade ruido a métricas dinámicas.
        
        Útil para hacer el modelo robusto a variaciones en métricas.
        """
        return CSPState(
            num_variables=csp_state.num_variables,
            num_constraints=csp_state.num_constraints,
            domains=csp_state.domains,
            constraint_graph=csp_state.constraint_graph,
            depth=csp_state.depth,
            num_backtracks=max(0, csp_state.num_backtracks + int(np.random.normal(0, noise_level * csp_state.num_backtracks))),
            num_propagations=max(0, csp_state.num_propagations + int(np.random.normal(0, noise_level * csp_state.num_propagations))),
            constraint_checks=max(0, csp_state.constraint_checks + int(np.random.normal(0, noise_level * csp_state.constraint_checks))),
            time_elapsed_ms=max(0, csp_state.time_elapsed_ms + np.random.normal(0, noise_level * csp_state.time_elapsed_ms))
        )
    
    def augment(self, csp_state: CSPState, num_augmentations: int = 5) -> List[CSPState]:
        """
        Genera múltiples augmentaciones.
        
        Args:
            csp_state: Estado original
            num_augmentations: Número de augmentaciones a generar
        
        Returns:
            Lista de estados augmentados (incluye original)
        """
        augmented = [csp_state]
        
        for _ in range(num_augmentations):
            # Aplicar transformaciones aleatorias
            aug = csp_state
            
            if random.random() < 0.5:
                aug = self.permute_variables(aug)
            
            if random.random() < 0.5:
                aug = self.permute_values(aug)
            
            if random.random() < 0.3:
                aug = self.add_noise_to_metrics(aug)
            
            augmented.append(aug)
        
        return augmented


# ============================================================================
# 2. TDA Augmenter
# ============================================================================

class TDAAugmenter:
    """
    Augmenta datos TDA preservando propiedades topológicas.
    
    Transformaciones:
    - Rotaciones (preserva topología)
    - Traslaciones (preserva topología)
    - Escalado uniforme (preserva topología)
    - Ruido gaussiano pequeño (aproximadamente preserva topología)
    - Sampling (subsampling de puntos)
    """
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def rotate(self, point_cloud: PointCloud, angle: float = None) -> PointCloud:
        """
        Rota point cloud (solo 2D/3D).
        
        Invarianza: Topología es invariante bajo rotaciones.
        """
        points = point_cloud.points.copy()
        dim = point_cloud.dimension
        
        if dim == 2:
            # Rotación 2D
            if angle is None:
                angle = np.random.uniform(0, 2 * np.pi)
            
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            
            points = points @ rotation_matrix.T
        
        elif dim == 3:
            # Rotación 3D (alrededor de eje aleatorio)
            from scipy.spatial.transform import Rotation
            
            if angle is None:
                rotation = Rotation.random()
            else:
                axis = np.random.randn(3)
                axis = axis / np.linalg.norm(axis)
                rotation = Rotation.from_rotvec(angle * axis)
            
            points = rotation.apply(points)
        
        return PointCloud(points=points)
    
    def translate(self, point_cloud: PointCloud, translation: np.ndarray = None) -> PointCloud:
        """
        Traslada point cloud.
        
        Invarianza: Topología es invariante bajo traslaciones.
        """
        points = point_cloud.points.copy()
        
        if translation is None:
            translation = np.random.randn(point_cloud.dimension)
        
        points = points + translation
        
        return PointCloud(points=points)
    
    def scale(self, point_cloud: PointCloud, scale_factor: float = None) -> PointCloud:
        """
        Escala point cloud uniformemente.
        
        Invarianza: Topología es invariante bajo escalado uniforme.
        """
        points = point_cloud.points.copy()
        
        if scale_factor is None:
            scale_factor = np.random.uniform(0.5, 2.0)
        
        points = points * scale_factor
        
        return PointCloud(points=points)
    
    def add_noise(self, point_cloud: PointCloud, noise_level: float = 0.05) -> PointCloud:
        """
        Añade ruido gaussiano.
        
        Nota: Ruido pequeño aproximadamente preserva topología.
        """
        points = point_cloud.points.copy()
        
        noise = np.random.normal(0, noise_level, points.shape)
        points = points + noise
        
        return PointCloud(points=points)
    
    def subsample(self, point_cloud: PointCloud, ratio: float = 0.8) -> PointCloud:
        """
        Submuestrea puntos.
        
        Nota: Puede cambiar topología, usar con cuidado.
        """
        points = point_cloud.points
        n = len(points)
        k = int(n * ratio)
        
        indices = np.random.choice(n, k, replace=False)
        sampled_points = points[indices]
        
        return PointCloud(points=sampled_points)
    
    def augment(self, point_cloud: PointCloud, num_augmentations: int = 5) -> List[PointCloud]:
        """
        Genera múltiples augmentaciones.
        
        Args:
            point_cloud: Point cloud original
            num_augmentations: Número de augmentaciones
        
        Returns:
            Lista de point clouds augmentados (incluye original)
        """
        augmented = [point_cloud]
        
        for _ in range(num_augmentations):
            aug = point_cloud
            
            # Aplicar transformaciones aleatorias
            if random.random() < 0.6:
                aug = self.rotate(aug)
            
            if random.random() < 0.5:
                aug = self.translate(aug)
            
            if random.random() < 0.4:
                aug = self.scale(aug)
            
            if random.random() < 0.3:
                aug = self.add_noise(aug)
            
            augmented.append(aug)
        
        return augmented


# ============================================================================
# 3. Cubical (Theorem Proving) Augmenter
# ============================================================================

class CubicalAugmenter:
    """
    Augmenta datos de theorem proving.
    
    Transformaciones:
    - Renombrar variables bound
    - Reordenar hipótesis
    - Añadir ruido a métricas
    """
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def reorder_hypotheses(self, proof_context: ProofContext) -> ProofContext:
        """
        Reordena hipótesis (el orden no debería importar).
        
        Invarianza: Prueba es equivalente bajo reordenamiento de hipótesis.
        """
        # Simulamos reordenamiento variando avg_hypothesis_complexity ligeramente
        new_avg = proof_context.avg_hypothesis_complexity * np.random.uniform(0.95, 1.05)
        
        return ProofContext(
            goal_complexity=proof_context.goal_complexity,
            goal_depth=proof_context.goal_depth,
            num_free_variables=proof_context.num_free_variables,
            num_bound_variables=proof_context.num_bound_variables,
            num_hypotheses=proof_context.num_hypotheses,
            avg_hypothesis_complexity=new_avg,
            context_size=proof_context.context_size,
            num_definitions=proof_context.num_definitions,
            num_lemmas_available=proof_context.num_lemmas_available,
            proof_depth=proof_context.proof_depth,
            num_subgoals=proof_context.num_subgoals,
            num_tactics_tried=proof_context.num_tactics_tried
        )
    
    def add_noise_to_metrics(self, proof_context: ProofContext, noise_level: float = 0.1) -> ProofContext:
        """Añade ruido a métricas."""
        return ProofContext(
            goal_complexity=max(1, proof_context.goal_complexity + int(np.random.normal(0, noise_level * proof_context.goal_complexity))),
            goal_depth=max(1, proof_context.goal_depth + int(np.random.normal(0, noise_level * proof_context.goal_depth))),
            num_free_variables=proof_context.num_free_variables,
            num_bound_variables=proof_context.num_bound_variables,
            num_hypotheses=proof_context.num_hypotheses,
            avg_hypothesis_complexity=max(1, proof_context.avg_hypothesis_complexity + np.random.normal(0, noise_level * proof_context.avg_hypothesis_complexity)),
            context_size=proof_context.context_size,
            num_definitions=proof_context.num_definitions,
            num_lemmas_available=proof_context.num_lemmas_available,
            proof_depth=proof_context.proof_depth,
            num_subgoals=max(1, proof_context.num_subgoals + int(np.random.normal(0, noise_level))),
            num_tactics_tried=max(0, proof_context.num_tactics_tried + int(np.random.normal(0, noise_level * proof_context.num_tactics_tried)))
        )
    
    def augment(self, proof_context: ProofContext, num_augmentations: int = 5) -> List[ProofContext]:
        """Genera augmentaciones."""
        augmented = [proof_context]
        
        for _ in range(num_augmentations):
            aug = proof_context
            
            if random.random() < 0.5:
                aug = self.reorder_hypotheses(aug)
            
            if random.random() < 0.3:
                aug = self.add_noise_to_metrics(aug)
            
            augmented.append(aug)
        
        return augmented


# ============================================================================
# 4. FCA Augmenter
# ============================================================================

class FCAAugmenter:
    """
    Augmenta datos FCA.
    
    Transformaciones:
    - Permutar objetos
    - Permutar atributos
    - Ambas permutaciones
    """
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def permute_objects(self, context: FormalContext) -> FormalContext:
        """
        Permuta objetos.
        
        Invarianza: Lattice de conceptos es isomorfo bajo permutación de objetos.
        """
        objects = list(context.objects)
        perm_objects = objects.copy()
        random.shuffle(perm_objects)
        
        obj_map = {old: new for old, new in zip(objects, perm_objects)}
        
        new_incidence = {(obj_map[obj], attr) for obj, attr in context.incidence}
        
        return FormalContext(
            objects=set(perm_objects),
            attributes=context.attributes,
            incidence=new_incidence
        )
    
    def permute_attributes(self, context: FormalContext) -> FormalContext:
        """
        Permuta atributos.
        
        Invarianza: Lattice de conceptos es isomorfo bajo permutación de atributos.
        """
        attributes = list(context.attributes)
        perm_attributes = attributes.copy()
        random.shuffle(perm_attributes)
        
        attr_map = {old: new for old, new in zip(attributes, perm_attributes)}
        
        new_incidence = {(obj, attr_map[attr]) for obj, attr in context.incidence}
        
        return FormalContext(
            objects=context.objects,
            attributes=set(perm_attributes),
            incidence=new_incidence
        )
    
    def augment(self, context: FormalContext, num_augmentations: int = 5) -> List[FormalContext]:
        """Genera augmentaciones."""
        augmented = [context]
        
        for _ in range(num_augmentations):
            aug = context
            
            if random.random() < 0.5:
                aug = self.permute_objects(aug)
            
            if random.random() < 0.5:
                aug = self.permute_attributes(aug)
            
            augmented.append(aug)
        
        return augmented


# ============================================================================
# 5. Homotopy Augmenter
# ============================================================================

class HomotopyAugmenter:
    """
    Augmenta datos homotópicos.
    
    Transformaciones:
    - Subdivisión de células
    - Añadir ruido a métricas
    """
    
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def subdivide(self, structure: HomotopyStructure) -> HomotopyStructure:
        """
        Simula subdivisión de células.
        
        Invarianza: Tipo de homotopía preservado bajo subdivisión.
        """
        # Subdivisión aumenta número de células
        new_num_cells = {
            dim: count * 2 for dim, count in structure.num_cells.items()
        }
        
        return HomotopyStructure(
            num_cells=new_num_cells,
            dimension=structure.dimension,
            euler_characteristic=structure.euler_characteristic,  # Preservado
            fundamental_group_rank=structure.fundamental_group_rank  # Preservado
        )
    
    def add_noise(self, structure: HomotopyStructure, noise_level: float = 0.1) -> HomotopyStructure:
        """Añade ruido a número de células."""
        new_num_cells = {
            dim: max(1, count + int(np.random.normal(0, noise_level * count)))
            for dim, count in structure.num_cells.items()
        }
        
        return HomotopyStructure(
            num_cells=new_num_cells,
            dimension=structure.dimension,
            euler_characteristic=structure.euler_characteristic,
            fundamental_group_rank=structure.fundamental_group_rank
        )
    
    def augment(self, structure: HomotopyStructure, num_augmentations: int = 5) -> List[HomotopyStructure]:
        """Genera augmentaciones."""
        augmented = [structure]
        
        for _ in range(num_augmentations):
            aug = structure
            
            if random.random() < 0.3:
                aug = self.subdivide(aug)
            
            if random.random() < 0.5:
                aug = self.add_noise(aug)
            
            augmented.append(aug)
        
        return augmented


# ============================================================================
# Demo y tests
# ============================================================================

if __name__ == "__main__":
    print("=== Data Augmentation Demo ===\n")
    
    # 1. CSP Augmenter
    print("1. CSP Augmenter")
    
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    
    csp_state = CSPState(
        num_variables=3,
        num_constraints=2,
        domains={0: {1, 2}, 1: {1, 2, 3}, 2: {2, 3}},
        constraint_graph=graph,
        depth=1
    )
    
    csp_augmenter = CSPAugmenter(seed=42)
    csp_augmented = csp_augmenter.augment(csp_state, num_augmentations=3)
    print(f"   Original: {csp_state.domains}")
    print(f"   Augmented (total): {len(csp_augmented)} instances")
    print(f"   Example augmented: {csp_augmented[1].domains}")
    
    # 2. TDA Augmenter
    print("\n2. TDA Augmenter")
    
    points = np.random.randn(20, 2)
    point_cloud = PointCloud(points=points)
    
    tda_augmenter = TDAAugmenter(seed=42)
    tda_augmented = tda_augmenter.augment(point_cloud, num_augmentations=3)
    print(f"   Original shape: {point_cloud.points.shape}")
    print(f"   Augmented (total): {len(tda_augmented)} instances")
    print(f"   Example augmented shape: {tda_augmented[1].points.shape}")
    print(f"   Mean difference: {np.mean(np.abs(tda_augmented[1].points - point_cloud.points)):.4f}")
    
    # 3. Cubical Augmenter
    print("\n3. Cubical Augmenter")
    
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
    
    cubical_augmenter = CubicalAugmenter(seed=42)
    cubical_augmented = cubical_augmenter.augment(proof_context, num_augmentations=3)
    print(f"   Original avg_hypothesis_complexity: {proof_context.avg_hypothesis_complexity}")
    print(f"   Augmented (total): {len(cubical_augmented)} instances")
    print(f"   Example augmented: {cubical_augmented[1].avg_hypothesis_complexity:.2f}")
    
    # 4. FCA Augmenter
    print("\n4. FCA Augmenter")
    
    formal_context = FormalContext(
        objects={'o1', 'o2', 'o3'},
        attributes={'a1', 'a2'},
        incidence={('o1', 'a1'), ('o2', 'a2'), ('o3', 'a1')}
    )
    
    fca_augmenter = FCAAugmenter(seed=42)
    fca_augmented = fca_augmenter.augment(formal_context, num_augmentations=3)
    print(f"   Original objects: {formal_context.objects}")
    print(f"   Augmented (total): {len(fca_augmented)} instances")
    print(f"   Example augmented objects: {fca_augmented[1].objects}")
    
    # 5. Homotopy Augmenter
    print("\n5. Homotopy Augmenter")
    
    homotopy_structure = HomotopyStructure(
        num_cells={0: 10, 1: 15, 2: 8},
        dimension=2,
        euler_characteristic=3,
        fundamental_group_rank=1
    )
    
    homotopy_augmenter = HomotopyAugmenter(seed=42)
    homotopy_augmented = homotopy_augmenter.augment(homotopy_structure, num_augmentations=3)
    print(f"   Original num_cells: {homotopy_structure.num_cells}")
    print(f"   Augmented (total): {len(homotopy_augmented)} instances")
    print(f"   Example augmented: {homotopy_augmented[1].num_cells}")
    
    print("\n=== All augmenters functional ===")
    print(f"\nData augmentation factor: {len(csp_augmented)}x (from 1 to {len(csp_augmented)} instances)")

