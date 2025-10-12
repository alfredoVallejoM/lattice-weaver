"""
Tests para TDA Engine (Topological Data Analysis)

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from lattice_weaver.topology.tda_engine import *


def test_simplex_creation():
    """Test: Creación de simplices."""
    print("\n" + "="*60)
    print("TEST 1: Creación de Simplices")
    print("="*60)
    
    # 0-simplex (punto)
    s0 = Simplex(frozenset([0]), 0)
    print(f"0-simplex: {s0.vertices}, dim={s0.dimension}")
    assert s0.dimension == 0
    assert len(s0.faces()) == 0
    
    # 1-simplex (arista)
    s1 = Simplex(frozenset([0, 1]), 1)
    print(f"1-simplex: {s1.vertices}, dim={s1.dimension}")
    assert s1.dimension == 1
    assert len(s1.faces()) == 2
    
    # 2-simplex (triángulo)
    s2 = Simplex(frozenset([0, 1, 2]), 2)
    print(f"2-simplex: {s2.vertices}, dim={s2.dimension}")
    assert s2.dimension == 2
    assert len(s2.faces()) == 3
    
    print("✅ Test pasado")


def test_simplicial_complex():
    """Test: Complejo simplicial."""
    print("\n" + "="*60)
    print("TEST 2: Complejo Simplicial")
    print("="*60)
    
    complex = SimplicialComplex()
    
    # Añadir triángulo (añade automáticamente aristas y vértices)
    triangle = Simplex(frozenset([0, 1, 2]), 2)
    complex.add_simplex(triangle)
    
    print(f"Simplices totales: {len(complex.simplices)}")
    print(f"Dimensión: {complex.dimension}")
    
    # Debe tener: 1 triángulo + 3 aristas + 3 vértices = 7 simplices
    assert len(complex.simplices) == 7
    assert complex.dimension == 2
    
    # Verificar por dimensión
    vertices = complex.get_simplices_by_dimension(0)
    edges = complex.get_simplices_by_dimension(1)
    triangles = complex.get_simplices_by_dimension(2)
    
    print(f"  Vértices: {len(vertices)}")
    print(f"  Aristas: {len(edges)}")
    print(f"  Triángulos: {len(triangles)}")
    
    assert len(vertices) == 3
    assert len(edges) == 3
    assert len(triangles) == 1
    
    print("✅ Test pasado")


def test_boundary_matrix():
    """Test: Matriz de frontera."""
    print("\n" + "="*60)
    print("TEST 3: Matriz de Frontera")
    print("="*60)
    
    complex = SimplicialComplex()
    
    # Añadir triángulo
    triangle = Simplex(frozenset([0, 1, 2]), 2)
    complex.add_simplex(triangle)
    
    # Matriz de frontera ∂_1 (aristas → vértices)
    boundary_1 = complex.get_boundary_matrix(1)
    print(f"Matriz ∂_1 shape: {boundary_1.shape}")
    print(f"Matriz ∂_1:\n{boundary_1}")
    
    assert boundary_1.shape == (3, 3)  # 3 vértices x 3 aristas
    
    # Matriz de frontera ∂_2 (triángulos → aristas)
    boundary_2 = complex.get_boundary_matrix(2)
    print(f"Matriz ∂_2 shape: {boundary_2.shape}")
    print(f"Matriz ∂_2:\n{boundary_2}")
    
    assert boundary_2.shape == (3, 1)  # 3 aristas x 1 triángulo
    
    print("✅ Test pasado")


def test_vietoris_rips_simple():
    """Test: Complejo de Vietoris-Rips simple."""
    print("\n" + "="*60)
    print("TEST 4: Vietoris-Rips Simple")
    print("="*60)
    
    # Tres puntos en línea
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0]
    ])
    
    engine = create_tda_engine()
    complex = engine.build_vietoris_rips(points, max_epsilon=1.5, max_dimension=1)
    
    print(f"Puntos: {len(points)}")
    print(f"Simplices: {len(complex.simplices)}")
    print(f"Dimensión: {complex.dimension}")
    
    vertices = complex.get_simplices_by_dimension(0)
    edges = complex.get_simplices_by_dimension(1)
    
    print(f"  Vértices: {len(vertices)}")
    print(f"  Aristas: {len(edges)}")
    
    assert len(vertices) == 3
    assert len(edges) >= 2  # Al menos (0,1) y (1,2)
    
    print("✅ Test pasado")


def test_vietoris_rips_circle():
    """Test: Complejo VR de puntos en círculo."""
    print("\n" + "="*60)
    print("TEST 5: Vietoris-Rips Círculo")
    print("="*60)
    
    # Puntos en círculo
    n_points = 8
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    engine = create_tda_engine()
    complex = engine.build_vietoris_rips(points, max_epsilon=1.5, max_dimension=2)
    
    print(f"Puntos: {len(points)}")
    print(f"Simplices: {len(complex.simplices)}")
    
    stats = engine.get_statistics()
    print(f"Vértices: {stats['n_vertices']}")
    print(f"Aristas: {stats['n_edges']}")
    print(f"Triángulos: {stats['n_triangles']}")
    
    assert stats['n_vertices'] == n_points
    assert stats['n_edges'] > 0
    
    print("✅ Test pasado")


def test_persistent_homology():
    """Test: Homología persistente."""
    print("\n" + "="*60)
    print("TEST 6: Homología Persistente")
    print("="*60)
    
    # Puntos simples
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866]  # Triángulo equilátero
    ])
    
    engine = create_tda_engine()
    engine.build_vietoris_rips(points, max_epsilon=2.0, max_dimension=2)
    
    intervals = engine.compute_persistent_homology(max_dimension=2)
    
    print(f"Intervalos de persistencia: {len(intervals)}")
    for i, interval in enumerate(intervals[:5]):  # Mostrar primeros 5
        print(f"  [{i}] dim={interval.dimension}, "
              f"birth={interval.birth:.3f}, death={interval.death:.3f}, "
              f"persistence={interval.persistence:.3f}")
    
    assert len(intervals) > 0
    
    print("✅ Test pasado")


def test_topological_features():
    """Test: Extracción de características topológicas."""
    print("\n" + "="*60)
    print("TEST 7: Características Topológicas")
    print("="*60)
    
    # Puntos en dos componentes
    points = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [5.0, 0.0],  # Componente separada
        [5.1, 0.0]
    ])
    
    engine = create_tda_engine()
    engine.build_vietoris_rips(points, max_epsilon=0.5, max_dimension=1)
    engine.compute_persistent_homology(max_dimension=1)
    
    features = engine.get_topological_features()
    
    print(f"Componentes: {features['n_components']}")
    print(f"Ciclos: {features['n_cycles']}")
    print(f"Números de Betti: {features['betti_numbers']}")
    print(f"Característica de Euler: {features['euler_characteristic']}")
    
    assert 'n_components' in features
    assert 'betti_numbers' in features
    
    print("✅ Test pasado")


def test_formal_context_extraction():
    """Test: Extracción de contexto formal desde topología."""
    print("\n" + "="*60)
    print("TEST 8: Extracción de Contexto Formal")
    print("="*60)
    
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866]
    ])
    
    engine = create_tda_engine()
    engine.build_vietoris_rips(points, max_epsilon=2.0, max_dimension=2)
    engine.compute_persistent_homology(max_dimension=2)
    
    objects, attributes, relation = engine.extract_formal_context_from_topology()
    
    print(f"Objetos: {len(objects)}")
    print(f"Atributos: {len(attributes)}")
    print(f"Relación: {len(relation)} pares")
    print(f"Atributos: {attributes}")
    
    assert len(objects) > 0
    assert len(attributes) > 0
    assert len(relation) > 0
    
    print("✅ Test pasado")


def test_analyze_point_cloud():
    """Test: Análisis completo de nube de puntos."""
    print("\n" + "="*60)
    print("TEST 9: Análisis Completo de Nube de Puntos")
    print("="*60)
    
    # Puntos aleatorios
    np.random.seed(42)
    points = np.random.rand(10, 2)
    
    results = analyze_point_cloud(points, max_epsilon=0.5, max_dimension=2)
    
    print(f"Complejo: {len(results['complex'].simplices)} simplices")
    print(f"Intervalos: {len(results['persistence_intervals'])}")
    print(f"Características:")
    for key, value in results['features'].items():
        if key != 'persistence_diagram':
            print(f"  {key}: {value}")
    
    assert 'complex' in results
    assert 'features' in results
    assert 'statistics' in results
    
    print("✅ Test pasado")


def test_tda_statistics():
    """Test: Estadísticas del motor TDA."""
    print("\n" + "="*60)
    print("TEST 10: Estadísticas TDA")
    print("="*60)
    
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866],
        [0.5, 0.289]
    ])
    
    engine = create_tda_engine()
    engine.build_vietoris_rips(points, max_epsilon=1.5, max_dimension=2)
    engine.compute_persistent_homology(max_dimension=2)
    
    stats = engine.get_statistics()
    
    print("Estadísticas:")
    for key, value in stats.items():
        if key != 'topological_features':
            print(f"  {key}: {value}")
    
    assert stats['n_simplices'] > 0
    assert stats['n_vertices'] == 4
    
    print("✅ Test pasado")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("Tests de TDA Engine (Topological Data Analysis)")
    print("LatticeWeaver v4")
    print("="*60)
    
    tests = [
        test_simplex_creation,
        test_simplicial_complex,
        test_boundary_matrix,
        test_vietoris_rips_simple,
        test_vietoris_rips_circle,
        test_persistent_homology,
        test_topological_features,
        test_formal_context_extraction,
        test_analyze_point_cloud,
        test_tda_statistics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test falló: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Resultados: {passed}/{len(tests)} tests pasados")
    if failed == 0:
        print("✅ TODOS LOS TESTS PASARON")
    else:
        print(f"❌ {failed} tests fallaron")
    print("="*60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

