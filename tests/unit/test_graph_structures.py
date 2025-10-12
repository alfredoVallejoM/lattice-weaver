"""
Tests unitarios para las estructuras de grafo del ACE.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.arc_weaver.graph_structures import (
    ConstraintGraph,
    ConstraintEdge,
    DynamicClusterGraph,
    Cluster
)


class TestConstraintEdge:
    """Tests para ConstraintEdge."""
    
    def test_create_constraint_edge(self):
        """Test creación de arista de restricción."""
        edge = ConstraintEdge('X', 'Y', lambda x, y: x != y)
        assert edge.var1 == 'X'
        assert edge.var2 == 'Y'
        assert edge.weight == 1.0
    
    def test_evaluate_constraint(self):
        """Test evaluación de restricción."""
        edge = ConstraintEdge('X', 'Y', lambda x, y: x != y)
        assert edge.evaluate(1, 2) == True
        assert edge.evaluate(1, 1) == False


class TestConstraintGraph:
    """Tests para ConstraintGraph."""
    
    def test_create_empty_graph(self):
        """Test creación de grafo vacío."""
        cg = ConstraintGraph()
        assert len(cg.get_all_variables()) == 0
        assert len(cg.get_all_constraints()) == 0
    
    def test_add_variable(self):
        """Test añadir variable."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        
        assert 'X' in cg.get_all_variables()
        assert cg.get_domain('X') == {1, 2, 3}
    
    def test_add_duplicate_variable_raises_error(self):
        """Test que añadir variable duplicada lanza error."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        
        with pytest.raises(ValueError, match="already exists"):
            cg.add_variable('X', {4, 5, 6})
    
    def test_add_constraint(self):
        """Test añadir restricción."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        assert cg.has_constraint('X', 'Y')
        assert cg.has_constraint('Y', 'X')  # Bidireccional
    
    def test_add_constraint_nonexistent_variable_raises_error(self):
        """Test que añadir restricción con variable inexistente lanza error."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        
        with pytest.raises(ValueError, match="must exist"):
            cg.add_constraint('X', 'Y', lambda x, y: x != y)
    
    def test_get_domain(self):
        """Test obtener dominio."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        
        domain = cg.get_domain('X')
        assert domain == {1, 2, 3}
    
    def test_get_domain_nonexistent_variable_raises_error(self):
        """Test que obtener dominio de variable inexistente lanza error."""
        cg = ConstraintGraph()
        
        with pytest.raises(KeyError, match="does not exist"):
            cg.get_domain('X')
    
    def test_update_domain(self):
        """Test actualizar dominio."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.update_domain('X', {1, 2})
        
        assert cg.get_domain('X') == {1, 2}
    
    def test_update_domain_empty_raises_error(self):
        """Test que actualizar con dominio vacío lanza error."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        
        with pytest.raises(ValueError, match="cannot be empty"):
            cg.update_domain('X', set())
    
    def test_get_neighbors(self):
        """Test obtener vecinos."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_variable('Z', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('X', 'Z', lambda x, z: x != z)
        
        neighbors = cg.get_neighbors('X')
        assert set(neighbors) == {'Y', 'Z'}
    
    def test_get_constraint(self):
        """Test obtener restricción."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        constraint = cg.get_constraint('X', 'Y')
        assert constraint is not None
        assert constraint.var1 == 'X'
        assert constraint.var2 == 'Y'
    
    def test_reset_domains(self):
        """Test restaurar dominios."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.update_domain('X', {1})
        
        cg.reset_domains()
        assert cg.get_domain('X') == {1, 2, 3}
    
    def test_is_consistent(self):
        """Test verificar consistencia."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        
        assert cg.is_consistent() == True
    
    def test_repr(self):
        """Test representación string."""
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        
        repr_str = repr(cg)
        assert 'vars=2' in repr_str
        assert 'constraints=1' in repr_str


class TestCluster:
    """Tests para Cluster."""
    
    def test_create_cluster(self):
        """Test creación de clúster."""
        cluster = Cluster(id=0, variables={'X', 'Y'})
        assert cluster.id == 0
        assert cluster.variables == {'X', 'Y'}
        assert cluster.state == "ACTIVE"
    
    def test_cluster_states(self):
        """Test estados del clúster."""
        cluster = Cluster(id=0, variables={'X', 'Y'})
        
        assert cluster.is_active() == True
        assert cluster.is_stable() == False
        
        cluster.mark_stable()
        assert cluster.is_stable() == True
        assert cluster.is_active() == False
        
        cluster.mark_solved()
        assert cluster.is_solved() == True
        
        cluster.mark_inconsistent()
        assert cluster.is_inconsistent() == True
    
    def test_mark_active_updates_iteration(self):
        """Test que mark_active actualiza la iteración."""
        cluster = Cluster(id=0, variables={'X', 'Y'})
        cluster.mark_active(10)
        
        assert cluster.last_update_iteration == 10
        assert cluster.state == "ACTIVE"


class TestDynamicClusterGraph:
    """Tests para DynamicClusterGraph."""
    
    def test_create_empty_graph(self):
        """Test creación de grafo vacío."""
        gcd = DynamicClusterGraph()
        assert gcd.is_empty() == True
        assert len(gcd.get_all_clusters()) == 0
    
    def test_add_cluster(self):
        """Test añadir clúster."""
        gcd = DynamicClusterGraph()
        cluster_id = gcd.add_cluster({'X', 'Y'})
        
        assert cluster_id == 0
        assert len(gcd.get_all_clusters()) == 1
        
        cluster = gcd.get_cluster(cluster_id)
        assert cluster.variables == {'X', 'Y'}
    
    def test_add_empty_cluster_raises_error(self):
        """Test que añadir clúster vacío lanza error."""
        gcd = DynamicClusterGraph()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            gcd.add_cluster(set())
    
    def test_add_boundary_constraint(self):
        """Test añadir restricción de frontera."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y'})
        c2 = gcd.add_cluster({'Z', 'W'})
        gcd.add_boundary_constraint(c1, c2)
        
        neighbors = gcd.get_neighbors(c1)
        assert c2 in neighbors
    
    def test_get_active_clusters(self):
        """Test obtener clústeres activos."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y'})
        c2 = gcd.add_cluster({'Z', 'W'})
        
        # Marcar c2 como estable
        gcd.get_cluster(c2).mark_stable()
        
        active = gcd.get_active_clusters()
        assert len(active) == 1
        assert active[0].id == c1
    
    def test_merge_clusters(self):
        """Test fusionar clústeres."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y'})
        c2 = gcd.add_cluster({'Z', 'W'})
        c3 = gcd.add_cluster({'A', 'B'})
        
        # Añadir fronteras
        gcd.add_boundary_constraint(c1, c2)
        gcd.add_boundary_constraint(c2, c3)
        
        # Fusionar c1 y c2
        new_id = gcd.merge_clusters(c1, c2)
        
        # Verificar que el nuevo clúster contiene todas las variables
        new_cluster = gcd.get_cluster(new_id)
        assert new_cluster.variables == {'X', 'Y', 'Z', 'W'}
        
        # Verificar que los clústeres antiguos fueron eliminados
        assert c1 not in gcd.clusters
        assert c2 not in gcd.clusters
        
        # Verificar que las fronteras se transfirieron
        neighbors = gcd.get_neighbors(new_id)
        assert c3 in neighbors
    
    def test_split_cluster(self):
        """Test dividir clúster."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y', 'Z', 'W'})
        
        # Dividir en dos
        new_id1, new_id2 = gcd.split_cluster(c1, {'X', 'Y'}, {'Z', 'W'})
        
        # Verificar que los nuevos clústeres tienen las variables correctas
        assert gcd.get_cluster(new_id1).variables == {'X', 'Y'}
        assert gcd.get_cluster(new_id2).variables == {'Z', 'W'}
        
        # Verificar que están conectados
        assert new_id2 in gcd.get_neighbors(new_id1)
        
        # Verificar que el clúster antiguo fue eliminado
        assert c1 not in gcd.clusters
    
    def test_split_cluster_invalid_partition_raises_error(self):
        """Test que dividir con particiones inválidas lanza error."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y', 'Z'})
        
        # Particiones no cubren todas las variables
        with pytest.raises(ValueError, match="must cover all variables"):
            gcd.split_cluster(c1, {'X'}, {'Y'})
        
        # Particiones no son disjuntas
        with pytest.raises(ValueError, match="must be disjoint"):
            gcd.split_cluster(c1, {'X', 'Y'}, {'Y', 'Z'})
    
    def test_prune_cluster(self):
        """Test podar clúster."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y'})
        c2 = gcd.add_cluster({'Z', 'W'})
        gcd.add_boundary_constraint(c1, c2)
        
        # Podar c1
        gcd.prune_cluster(c1)
        
        # Verificar que fue eliminado
        assert c1 not in gcd.clusters
        assert len(gcd.get_all_clusters()) == 1
    
    def test_get_cluster_nonexistent_raises_error(self):
        """Test que obtener clúster inexistente lanza error."""
        gcd = DynamicClusterGraph()
        
        with pytest.raises(KeyError, match="does not exist"):
            gcd.get_cluster(999)
    
    def test_repr(self):
        """Test representación string."""
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y'})
        c2 = gcd.add_cluster({'Z', 'W'})
        gcd.add_boundary_constraint(c1, c2)
        
        repr_str = repr(gcd)
        assert 'clusters=2' in repr_str
        assert 'boundaries=1' in repr_str


class TestIntegration:
    """Tests de integración entre ConstraintGraph y DynamicClusterGraph."""
    
    def test_build_cluster_graph_from_constraint_graph(self):
        """Test construir GCD a partir de GR."""
        # Crear grafo de restricciones
        cg = ConstraintGraph()
        cg.add_variable('X', {1, 2, 3})
        cg.add_variable('Y', {1, 2, 3})
        cg.add_variable('Z', {1, 2, 3})
        cg.add_variable('W', {1, 2, 3})
        
        cg.add_constraint('X', 'Y', lambda x, y: x != y)
        cg.add_constraint('Y', 'Z', lambda y, z: y != z)
        cg.add_constraint('Z', 'W', lambda z, w: z != w)
        
        # Crear grafo de clústeres
        gcd = DynamicClusterGraph()
        c1 = gcd.add_cluster({'X', 'Y'})
        c2 = gcd.add_cluster({'Z', 'W'})
        
        # Verificar que hay restricciones entre clústeres
        c1_vars = gcd.get_cluster(c1).variables
        c2_vars = gcd.get_cluster(c2).variables
        
        has_boundary = any(
            cg.has_constraint(v1, v2)
            for v1 in c1_vars for v2 in c2_vars
        )
        
        assert has_boundary == True  # Y-Z conecta los clústeres
        
        # Añadir frontera
        gcd.add_boundary_constraint(c1, c2)
        assert c2 in gcd.get_neighbors(c1)

