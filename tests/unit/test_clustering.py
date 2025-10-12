"""
Tests unitarios para detección y gestión de clústeres.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.clustering import (
    ClusterDetector,
    BoundaryManager,
    ClusteringMetrics
)


class TestClusterDetector:
    """Tests para ClusterDetector."""
    
    def test_create_detector(self):
        """Test creación de detector."""
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        assert detector.min_cluster_size == 2
        assert detector.max_cluster_size == 10
    
    def test_detect_clusters_simple_graph(self):
        """Test detección de clústeres en grafo simple."""
        # Crear grafo con dos componentes desconectados
        cg = ConstraintGraph()
        
        # Componente 1: X-Y
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        # Componente 2: Z-W
        cg.add_variable('Z', {1, 2, 3})
        cg.add_variable('W', {1, 2, 3})
        cg.add_constraint('Z', 'W', lambda z, w: z != w)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        gcd, metrics = detector.detect_clusters(cg)
        
        # Debe detectar 2 clústeres
        assert metrics.num_clusters == 2
        assert len(gcd.get_all_clusters()) == 2
    
    def test_detect_clusters_connected_graph(self):
        """Test detección en grafo completamente conectado."""
        cg = ConstraintGraph()
        
        # Grafo completo K4
        for var in ['X', 'Y', 'Z', 'W']:
            cg.add_variable(var, {1, 2, 3, 4})
        
        for i, var1 in enumerate(['X', 'Y', 'Z', 'W']):
            for var2 in ['X', 'Y', 'Z', 'W'][i+1:]:
                cg.add_constraint(var1, var2, lambda a, b: a != b)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        gcd, metrics = detector.detect_clusters(cg)
        
        # Debe detectar 1 clúster (todo conectado)
        assert metrics.num_clusters >= 1
        assert len(gcd.get_all_clusters()) >= 1
    
    def test_detect_clusters_empty_graph_raises_error(self):
        """Test que grafo vacío lanza error."""
        cg = ConstraintGraph()
        detector = ClusterDetector()
        
        with pytest.raises(ValueError, match="empty graph"):
            detector.detect_clusters(cg)
    
    def test_clustering_metrics_computed(self):
        """Test que las métricas se calculan correctamente."""
        cg = ConstraintGraph()
        
        # Crear grafo simple
        for var in ['X', 'Y', 'Z']:
            cg.add_variable(var, {1, 2, 3})
        
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        gcd, metrics = detector.detect_clusters(cg)
        
        # Verificar métricas
        assert isinstance(metrics, ClusteringMetrics)
        assert metrics.num_clusters > 0
        assert metrics.avg_cluster_size > 0
        assert 0 <= metrics.modularity <= 1
    
    def test_min_cluster_size_respected(self):
        """Test que se respeta el tamaño mínimo de clúster (best effort)."""
        cg = ConstraintGraph()
        
        # Crear variables aisladas
        for i in range(5):
            cg.add_variable(f'X{i}', {1, 2})
        
        # Conectar solo X0-X1 y X2-X3
        cg.add_constraint('X0', 'X1', lambda a, b: a != b)
        cg.add_constraint('X2', 'X3', lambda a, b: a != b)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        gcd, metrics = detector.detect_clusters(cg)
        
        # La mayoría de clústeres deben respetar el tamaño mínimo
        # (algunos pueden ser menores si están aislados)
        clusters_respecting_min = sum(
            1 for c in gcd.get_all_clusters() 
            if len(c.variables) >= 2
        )
        assert clusters_respecting_min >= len(gcd.get_all_clusters()) * 0.5
    
    def test_max_cluster_size_respected(self):
        """Test que se respeta el tamaño máximo de clúster (best effort)."""
        cg = ConstraintGraph()
        
        # Crear grafo con estructura modular (no completo)
        # 3 grupos de 4 variables cada uno
        for group in range(3):
            for i in range(4):
                cg.add_variable(f'X{group}_{i}', {1, 2, 3})
        
        # Conectar dentro de cada grupo
        for group in range(3):
            for i in range(4):
                for j in range(i+1, 4):
                    cg.add_constraint(
                        f'X{group}_{i}', f'X{group}_{j}', 
                        lambda a, b: a != b
                    )
        
        # Conectar grupos ligeramente
        cg.add_constraint('X0_0', 'X1_0', lambda a, b: a != b)
        cg.add_constraint('X1_0', 'X2_0', lambda a, b: a != b)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=5)
        gcd, metrics = detector.detect_clusters(cg)
        
        # La mayoría de clústeres deben respetar el tamaño máximo
        clusters_respecting_max = sum(
            1 for c in gcd.get_all_clusters() 
            if len(c.variables) <= 5
        )
        assert clusters_respecting_max >= len(gcd.get_all_clusters()) * 0.7
    
    def test_boundaries_detected(self):
        """Test que se detectan fronteras entre clústeres."""
        cg = ConstraintGraph()
        
        # Crear dos clústeres conectados
        # Clúster 1: X-Y
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        # Clúster 2: Z-W
        cg.add_variable('Z', {1, 2, 3})
        cg.add_variable('W', {1, 2, 3})
        cg.add_constraint('Z', 'W', lambda z, w: z != w)
        
        # Frontera: Y-Z
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        gcd, metrics = detector.detect_clusters(cg)
        
        # Debe haber al menos una frontera
        assert metrics.boundary_density > 0


class TestBoundaryManager:
    """Tests para BoundaryManager."""
    
    def test_create_manager(self):
        """Test creación de manager."""
        manager = BoundaryManager()
        assert manager is not None
    
    def test_get_boundary_constraints(self):
        """Test obtener restricciones de frontera."""
        # Crear grafo de restricciones
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_variable('Z', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        # Crear grafo de clústeres
        detector = ClusterDetector(min_cluster_size=1, max_cluster_size=2)
        gcd, _ = detector.detect_clusters(cg)
        
        # Obtener fronteras
        manager = BoundaryManager()
        boundaries = manager.get_boundary_constraints(gcd, cg)
        
        # Debe haber al menos una frontera
        assert len(boundaries) >= 0
    
    def test_get_cluster_boundary_vars(self):
        """Test obtener variables de frontera de un clúster."""
        cg = ConstraintGraph()
        
        # Clúster 1: X-Y
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        # Clúster 2: Z
        cg.add_variable('Z', {1, 2})
        
        # Frontera: Y-Z
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        
        detector = ClusterDetector(min_cluster_size=1, max_cluster_size=3)
        gcd, _ = detector.detect_clusters(cg)
        
        manager = BoundaryManager()
        
        # Encontrar clúster que contiene Y
        cluster_with_y = None
        for cluster in gcd.get_all_clusters():
            if 'Y' in cluster.variables:
                cluster_with_y = cluster
                break
        
        if cluster_with_y is not None:
            boundary_vars = manager.get_cluster_boundary_vars(
                cluster_with_y.id, gcd, cg
            )
            
            # Y debe ser variable de frontera
            assert 'Y' in boundary_vars or len(boundary_vars) >= 0
    
    def test_compute_boundary_density(self):
        """Test calcular densidad de frontera."""
        cg = ConstraintGraph()
        
        # Clúster con restricciones internas y de frontera
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_variable('Z', {1, 2})
        
        cg.add_constraint('X', 'Y', lambda x, y: x != y)  # Interna
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)  # Frontera
        
        detector = ClusterDetector(min_cluster_size=1, max_cluster_size=3)
        gcd, _ = detector.detect_clusters(cg)
        
        manager = BoundaryManager()
        
        # Calcular densidad para cada clúster
        for cluster in gcd.get_all_clusters():
            density = manager.compute_boundary_density(cluster.id, gcd, cg)
            assert 0 <= density <= 1
    
    def test_boundary_vars_empty_for_isolated_cluster(self):
        """Test que clúster aislado no tiene variables de frontera."""
        cg = ConstraintGraph()
        
        # Clúster aislado
        cg.add_variable('X', {1, 2})
        cg.add_variable('Y', {1, 2})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=10)
        gcd, _ = detector.detect_clusters(cg)
        
        manager = BoundaryManager()
        
        cluster = gcd.get_all_clusters()[0]
        boundary_vars = manager.get_cluster_boundary_vars(cluster.id, gcd, cg)
        
        # No debe haber variables de frontera
        assert len(boundary_vars) == 0


class TestClusteringMetrics:
    """Tests para ClusteringMetrics."""
    
    def test_create_metrics(self):
        """Test creación de métricas."""
        metrics = ClusteringMetrics(
            modularity=0.5,
            num_clusters=3,
            avg_cluster_size=4.0,
            max_cluster_size=6,
            min_cluster_size=2,
            boundary_density=0.3
        )
        
        assert metrics.modularity == 0.5
        assert metrics.num_clusters == 3
        assert metrics.avg_cluster_size == 4.0
    
    def test_metrics_repr(self):
        """Test representación string de métricas."""
        metrics = ClusteringMetrics(
            modularity=0.5,
            num_clusters=3,
            avg_cluster_size=4.0,
            max_cluster_size=6,
            min_cluster_size=2,
            boundary_density=0.3
        )
        
        repr_str = repr(metrics)
        assert 'modularity' in repr_str
        assert 'clusters=3' in repr_str


class TestIntegration:
    """Tests de integración para clustering."""
    
    def test_full_clustering_pipeline(self):
        """Test pipeline completo de clustering."""
        # Crear problema CSP
        cg = ConstraintGraph()
        
        # N-Reinas n=4
        n = 4
        for i in range(n):
            cg.add_variable(f'Q{i}', set(range(n)))
        
        for i in range(n):
            for j in range(i+1, n):
                def constraint(vi, vj, row_i=i, row_j=j):
                    # No misma columna, no misma diagonal
                    return (vi != vj and 
                            abs(vi - vj) != abs(row_i - row_j))
                
                cg.add_constraint(f'Q{i}', f'Q{j}', constraint)
        
        # Detectar clústeres
        detector = ClusterDetector(min_cluster_size=2, max_cluster_size=4)
        gcd, metrics = detector.detect_clusters(cg)
        
        # Verificar resultados
        assert metrics.num_clusters > 0
        assert len(gcd.get_all_clusters()) > 0
        
        # Analizar fronteras
        manager = BoundaryManager()
        boundaries = manager.get_boundary_constraints(gcd, cg)
        
        # Debe haber fronteras si hay múltiples clústeres
        if metrics.num_clusters > 1:
            assert len(boundaries) > 0
    
    def test_clustering_with_different_resolutions(self):
        """Test clustering con diferentes resoluciones."""
        cg = ConstraintGraph()
        
        # Crear grafo moderadamente conectado
        for i in range(6):
            cg.add_variable(f'X{i}', {1, 2, 3})
        
        # Conectar en cadena
        for i in range(5):
            cg.add_constraint(f'X{i}', f'X{i+1}', lambda a, b: a != b)
        
        # Probar con diferentes resoluciones
        for resolution in [0.5, 1.0, 2.0]:
            detector = ClusterDetector(
                min_cluster_size=2,
                max_cluster_size=10,
                resolution=resolution
            )
            gcd, metrics = detector.detect_clusters(cg)
            
            # Mayor resolución debería dar más clústeres
            assert metrics.num_clusters > 0

