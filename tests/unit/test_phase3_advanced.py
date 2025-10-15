"""
Tests para Fase 3: Advanced Optimizations

Tests para:
- Hacification Incremental
- Compilación JIT
- Vectorización NumPy

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import pytest
import numpy as np

from lattice_weaver.fibration.hacification_incremental import (
    IncrementalHacificationEngine, HacificationDelta, HacificationSnapshot
)
from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy
from lattice_weaver.utils.jit_compiler import (
    domain_intersection_jit, domain_difference_jit, domain_size_jit,
    mrv_score_jit, degree_score_jit, weighted_degree_score_jit,
    get_jit_compiler
)
from lattice_weaver.utils.numpy_vectorization import (
    NumpyVectorizer, VectorizedDomains, get_numpy_vectorizer
)


class TestHacificationIncremental:
    """Tests para Hacification Incremental."""
    
    def test_initialization(self):
        """Test inicialización."""
        hierarchy = ConstraintHierarchy()
        
        engine = IncrementalHacificationEngine(hierarchy)
        
        assert engine.hierarchy == hierarchy
        assert engine.timestamp == 0
        assert len(engine.snapshots) == 0
    
    def test_full_hacification(self):
        """Test hacification completa."""
        hierarchy = ConstraintHierarchy()
        hierarchy.add_local_constraint(
            "A", "B",
            lambda a: a.get("A", 0) != a.get("B", 0)
        )
        
        engine = IncrementalHacificationEngine(hierarchy)
        
        domains = {"A": [1, 2, 3], "B": [1, 2, 3]}
        result = engine.hacify(domains, force_full=True)
        
        assert result is not None
        assert engine.stats['full_hacifications'] == 1
    
    def test_incremental_hacification(self):
        """Test hacification incremental."""
        hierarchy = ConstraintHierarchy()
        
        engine = IncrementalHacificationEngine(hierarchy)
        
        # Primera hacification (full)
        domains1 = {"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3]}
        result1 = engine.hacify(domains1)
        
        # Segunda hacification con cambio pequeño (incremental)
        domains2 = {"A": [1, 2], "B": [1, 2, 3], "C": [1, 2, 3]}
        result2 = engine.hacify(domains2)
        
        assert engine.stats['incremental_updates'] >= 1
    
    def test_snapshot_creation(self):
        """Test creación de snapshots."""
        hierarchy = ConstraintHierarchy()
        
        engine = IncrementalHacificationEngine(hierarchy, snapshot_interval=2)
        
        domains = {"A": [1, 2, 3]}
        
        # Hacer varias hacifications para trigger snapshots
        for i in range(5):
            engine.hacify(domains, force_full=True)
        
        assert len(engine.snapshots) > 0
        assert engine.stats['snapshots_created'] > 0


class TestJITCompiler:
    """Tests para JIT Compiler."""
    
    def test_domain_intersection(self):
        """Test intersección de dominios JIT."""
        domain1 = np.array([1, 2, 3, 4])
        domain2 = np.array([2, 3, 4, 5])
        
        result = domain_intersection_jit(domain1, domain2)
        
        expected = np.array([2, 3, 4])
        assert np.array_equal(result, expected)
    
    def test_domain_difference(self):
        """Test diferencia de dominios JIT."""
        domain1 = np.array([1, 2, 3, 4])
        domain2 = np.array([2, 3])
        
        result = domain_difference_jit(domain1, domain2)
        
        expected = np.array([1, 4])
        assert np.array_equal(result, expected)
    
    def test_domain_size(self):
        """Test tamaño de dominio JIT."""
        domain = np.array([1, 2, 3, 4, 5])
        
        size = domain_size_jit(domain)
        
        assert size == 5
    
    def test_mrv_score(self):
        """Test MRV score JIT."""
        domain_sizes = np.array([3, 1, 5, 2])
        
        mrv_idx = mrv_score_jit(domain_sizes)
        
        assert mrv_idx == 1  # Variable con dominio más pequeño
    
    def test_degree_score(self):
        """Test Degree score JIT."""
        constraint_counts = np.array([2, 5, 3, 1])
        
        degree_idx = degree_score_jit(constraint_counts)
        
        assert degree_idx == 1  # Variable con más restricciones
    
    def test_weighted_degree_score(self):
        """Test Weighted Degree score JIT."""
        domain_sizes = np.array([2, 4, 3])
        weighted_degrees = np.array([1.0, 2.0, 1.5])
        
        wdeg_idx = weighted_degree_score_jit(domain_sizes, weighted_degrees)
        
        # Scores: 2/1=2, 4/2=2, 3/1.5=2
        # Todos iguales, retorna el primero
        assert wdeg_idx in [0, 1, 2]
    
    def test_jit_compiler_manager(self):
        """Test JIT Compiler manager."""
        compiler = get_jit_compiler()
        
        def test_func(x):
            return x * 2
        
        compiled = compiler.compile_function(test_func, mode='jit')
        
        assert compiled is not None
        assert compiler.stats['functions_compiled'] > 0


class TestNumpyVectorization:
    """Tests para NumPy Vectorization."""
    
    def test_vectorize_domains(self):
        """Test vectorización de dominios."""
        vectorizer = NumpyVectorizer(max_domain_size=10)
        
        domains = {
            "A": [1, 2, 3],
            "B": [4, 5, 6, 7],
            "C": [8, 9]
        }
        
        vectorized = vectorizer.vectorize_domains(domains)
        
        assert vectorized.domains.shape[0] == 3  # 3 variables
        assert vectorized.sizes[0] == 3  # A tiene 3 valores
        assert vectorized.sizes[1] == 4  # B tiene 4 valores
        assert vectorized.sizes[2] == 2  # C tiene 2 valores
    
    def test_devectorize_domains(self):
        """Test devectorización de dominios."""
        vectorizer = NumpyVectorizer(max_domain_size=10)
        
        original_domains = {
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        }
        
        vectorized = vectorizer.vectorize_domains(original_domains)
        devectorized = vectorizer.devectorize_domains(vectorized)
        
        assert devectorized == original_domains
    
    def test_intersection_vectorized(self):
        """Test intersección vectorizada."""
        vectorizer = NumpyVectorizer(max_domain_size=10)
        
        domains1 = {"A": [1, 2, 3, 4], "B": [5, 6, 7]}
        domains2 = {"A": [2, 3, 4, 5], "B": [6, 7, 8]}
        
        vec1 = vectorizer.vectorize_domains(domains1)
        vec2 = vectorizer.vectorize_domains(domains2)
        
        intersection = vectorizer.intersection_vectorized(vec1, vec2)
        result = vectorizer.devectorize_domains(intersection)
        
        assert result["A"] == [2, 3, 4]
        assert result["B"] == [6, 7]
    
    def test_compute_mrv_vectorized(self):
        """Test MRV vectorizado."""
        vectorizer = NumpyVectorizer(max_domain_size=10)
        
        domains = {
            "A": [1, 2, 3],
            "B": [4, 5],
            "C": [6, 7, 8, 9]
        }
        
        vectorized = vectorizer.vectorize_domains(domains)
        unassigned_mask = np.array([True, True, True])
        
        mrv_idx = vectorizer.compute_mrv_vectorized(vectorized, unassigned_mask)
        
        # B tiene el dominio más pequeño (2 valores)
        assert mrv_idx == 1
    
    def test_compute_degree_vectorized(self):
        """Test Degree vectorizado."""
        vectorizer = NumpyVectorizer(max_domain_size=10)
        
        # Matriz de restricciones (3x3)
        constraint_matrix = np.array([
            [0, 1, 1],  # A tiene 2 restricciones
            [1, 0, 1],  # B tiene 2 restricciones
            [1, 1, 0]   # C tiene 2 restricciones
        ])
        
        unassigned_mask = np.array([True, True, True])
        
        degree_idx = vectorizer.compute_degree_vectorized(
            constraint_matrix,
            unassigned_mask
        )
        
        # Todos tienen el mismo degree, retorna el primero
        assert degree_idx in [0, 1, 2]
    
    def test_filter_by_predicate_vectorized(self):
        """Test filtrado por predicado vectorizado."""
        vectorizer = NumpyVectorizer(max_domain_size=10)
        
        domains = {"A": [1, 2, 3, 4, 5]}
        vectorized = vectorizer.vectorize_domains(domains)
        
        # Predicado: mantener solo valores pares
        predicate_matrix = np.array([[False, True, False, True, False, False, False, False, False, False]])
        
        filtered = vectorizer.filter_by_predicate_vectorized(vectorized, predicate_matrix)
        result = vectorizer.devectorize_domains(filtered)
        
        assert result["A"] == [2, 4]
    
    def test_stats(self):
        """Test estadísticas."""
        vectorizer = get_numpy_vectorizer()
        vectorizer.reset_stats()
        
        domains = {"A": [1, 2, 3]}
        vectorized = vectorizer.vectorize_domains(domains)
        
        stats = vectorizer.get_stats()
        
        assert 'vectorized_operations' in stats
        assert 'elements_processed' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

