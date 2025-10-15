"""
Tests para la integración topológica (Fase 4).

Este módulo valida:
- CSPTopologyAdapter
- Estrategias topológicas (TopologyGuidedSelector, ComponentBasedSelector)
- Estrategias híbridas (HybridFCATopologySelector, AdaptiveMultiscaleSelector)

Autor: Manus AI
Fecha: 15 de Octubre de 2025
"""

import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.topology_adapter import CSPTopologyAdapter, analyze_csp_topology
from lattice_weaver.core.csp_engine.strategies.topology_guided import (
    TopologyGuidedSelector,
    ComponentBasedSelector
)
from lattice_weaver.core.csp_engine.strategies.hybrid_multiescala import (
    HybridFCATopologySelector,
    AdaptiveMultiscaleSelector
)


# ============================================================================
# Tests para CSPTopologyAdapter
# ============================================================================

def test_adapter_build_consistency_graph():
    """El adaptador debe construir el grafo de consistencia correctamente"""
    csp = CSP(
        variables=['A', 'B', 'C'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2]),
            'C': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
            Constraint(scope=('B', 'C'), relation=lambda b, c: b != c)
        ]
    )
    
    adapter = CSPTopologyAdapter(csp)
    graph = adapter.build_consistency_graph()
    
    # Verificar nodos (3 variables × 2 valores = 6 nodos)
    assert graph.number_of_nodes() == 6
    
    # Verificar que hay aristas
    assert graph.number_of_edges() > 0
    
    # Verificar mapeos
    assert len(adapter.node_to_id) == 6
    assert len(adapter.id_to_node) == 6


def test_adapter_find_connected_components():
    """El adaptador debe encontrar componentes conexas"""
    # CSP con dos componentes independientes
    csp = CSP(
        variables=['A', 'B', 'C', 'D'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2]),
            'C': frozenset([1, 2]),
            'D': frozenset([1, 2])
        },
        constraints=[
            # Componente 1: A-B
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
            # Componente 2: C-D
            Constraint(scope=('C', 'D'), relation=lambda c, d: c != d)
        ]
    )
    
    adapter = CSPTopologyAdapter(csp)
    components = adapter.find_connected_components()
    
    # Debe haber al menos 2 componentes (A-B y C-D están desconectados)
    assert len(components) >= 2


def test_adapter_compute_graph_metrics():
    """El adaptador debe calcular métricas del grafo"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
        ]
    )
    
    adapter = CSPTopologyAdapter(csp)
    metrics = adapter.compute_graph_metrics()
    
    # Verificar que todas las métricas están presentes
    assert 'num_nodes' in metrics
    assert 'num_edges' in metrics
    assert 'density' in metrics
    assert 'num_components' in metrics
    assert 'largest_component_size' in metrics
    assert 'average_degree' in metrics
    assert 'clustering_coefficient' in metrics
    
    # Verificar valores razonables
    assert metrics['num_nodes'] == 4  # 2 variables × 2 valores
    assert metrics['num_edges'] >= 0
    assert 0 <= metrics['density'] <= 1
    assert metrics['num_components'] >= 1


def test_adapter_find_critical_nodes():
    """El adaptador debe encontrar nodos críticos"""
    csp = CSP(
        variables=['A', 'B', 'C'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2]),
            'C': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
            Constraint(scope=('B', 'C'), relation=lambda b, c: b != c)
        ]
    )
    
    adapter = CSPTopologyAdapter(csp)
    critical_nodes = adapter.find_critical_nodes(top_k=3)
    
    # Debe retornar lista de (nodo, centralidad)
    assert isinstance(critical_nodes, list)
    assert len(critical_nodes) <= 3
    
    if critical_nodes:
        node, centrality = critical_nodes[0]
        assert isinstance(node, tuple)
        assert len(node) == 2  # (variable, valor)
        assert isinstance(centrality, float)
        assert centrality >= 0


def test_adapter_analyze_structure():
    """El adaptador debe realizar análisis estructural completo"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
        ]
    )
    
    adapter = CSPTopologyAdapter(csp)
    analysis = adapter.analyze_structure()
    
    # Verificar estructura del análisis
    assert 'metrics' in analysis
    assert 'components' in analysis
    assert 'critical_nodes' in analysis
    assert 'summary' in analysis
    
    # Verificar que summary es texto
    assert isinstance(analysis['summary'], str)
    assert len(analysis['summary']) > 0


def test_analyze_csp_topology_function():
    """La función de conveniencia debe funcionar"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1]),
            'B': frozenset([1])
        },
        constraints=[]
    )
    
    analysis = analyze_csp_topology(csp)
    
    assert 'metrics' in analysis
    assert 'summary' in analysis


# ============================================================================
# Tests para TopologyGuidedSelector
# ============================================================================

def test_topology_guided_selector_basic():
    """TopologyGuidedSelector debe seleccionar variables"""
    csp = CSP(
        variables=['A', 'B', 'C'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2]),
            'C': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
            Constraint(scope=('B', 'C'), relation=lambda b, c: b != c)
        ]
    )
    
    selector = TopologyGuidedSelector()
    current_domains = {
        'A': [1, 2],
        'B': [1, 2],
        'C': [1, 2]
    }
    
    selected = selector.select(csp, {}, current_domains)
    assert selected in ['A', 'B', 'C']


def test_topology_guided_selector_all_assigned():
    """TopologyGuidedSelector debe retornar None cuando todas están asignadas"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1]),
            'B': frozenset([1])
        },
        constraints=[]
    )
    
    selector = TopologyGuidedSelector()
    assignment = {'A': 1, 'B': 1}
    current_domains = {'A': [1], 'B': [1]}
    
    selected = selector.select(csp, assignment, current_domains)
    assert selected is None


def test_topology_guided_selector_single_variable():
    """TopologyGuidedSelector debe manejar CSP con una sola variable"""
    csp = CSP(
        variables=['A'],
        domains={'A': frozenset([1, 2])},
        constraints=[]
    )
    
    selector = TopologyGuidedSelector()
    current_domains = {'A': [1, 2]}
    
    selected = selector.select(csp, {}, current_domains)
    assert selected == 'A'


def test_topology_guided_selector_cache():
    """TopologyGuidedSelector debe cachear análisis topológico"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
        ]
    )
    
    selector = TopologyGuidedSelector()
    current_domains = {'A': [1, 2], 'B': [1, 2]}
    
    # Primera selección (construye caché)
    selected1 = selector.select(csp, {}, current_domains)
    
    # Segunda selección (usa caché)
    selected2 = selector.select(csp, {}, current_domains)
    
    # Debe usar caché
    csp_id = id(csp)
    assert csp_id in selector._critical_vars_cache


def test_topology_guided_selector_reset_cache():
    """TopologyGuidedSelector debe permitir limpiar caché"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1]),
            'B': frozenset([1])
        },
        constraints=[]
    )
    
    selector = TopologyGuidedSelector()
    current_domains = {'A': [1], 'B': [1]}
    
    # Construir caché
    selector.select(csp, {}, current_domains)
    assert len(selector._critical_vars_cache) > 0
    
    # Limpiar caché
    selector.reset_cache()
    assert len(selector._critical_vars_cache) == 0
    assert len(selector._topology_cache) == 0


# ============================================================================
# Tests para ComponentBasedSelector
# ============================================================================

def test_component_based_selector_basic():
    """ComponentBasedSelector debe seleccionar variables"""
    csp = CSP(
        variables=['A', 'B', 'C', 'D'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2]),
            'C': frozenset([1, 2]),
            'D': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
            Constraint(scope=('C', 'D'), relation=lambda c, d: c != d)
        ]
    )
    
    selector = ComponentBasedSelector()
    current_domains = {
        'A': [1, 2],
        'B': [1, 2],
        'C': [1, 2],
        'D': [1, 2]
    }
    
    selected = selector.select(csp, {}, current_domains)
    assert selected in ['A', 'B', 'C', 'D']


def test_component_based_selector_processes_smallest_first():
    """ComponentBasedSelector debe procesar componentes pequeñas primero"""
    # CSP con componentes de tamaños diferentes
    csp = CSP(
        variables=['A', 'B', 'C', 'D', 'E'],
        domains={
            'A': frozenset([1]),
            'B': frozenset([1]),
            'C': frozenset([1]),
            'D': frozenset([1]),
            'E': frozenset([1])
        },
        constraints=[
            # Componente grande: A-B-C
            Constraint(scope=('A', 'B'), relation=lambda a, b: True),
            Constraint(scope=('B', 'C'), relation=lambda b, c: True),
            # Componente pequeña: D-E
            Constraint(scope=('D', 'E'), relation=lambda d, e: True)
        ]
    )
    
    selector = ComponentBasedSelector()
    current_domains = {
        'A': [1],
        'B': [1],
        'C': [1],
        'D': [1],
        'E': [1]
    }
    
    # Debe seleccionar de la componente más pequeña
    selected = selector.select(csp, {}, current_domains)
    assert selected is not None


# ============================================================================
# Tests para HybridFCATopologySelector
# ============================================================================

def test_hybrid_fca_topology_selector_basic():
    """HybridFCATopologySelector debe seleccionar variables"""
    csp = CSP(
        variables=['A', 'B', 'C'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2]),
            'C': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b),
            Constraint(scope=('B', 'C'), relation=lambda b, c: b != c)
        ]
    )
    
    selector = HybridFCATopologySelector(fca_weight=0.5, topology_weight=0.5)
    current_domains = {
        'A': [1, 2],
        'B': [1, 2],
        'C': [1, 2]
    }
    
    selected = selector.select(csp, {}, current_domains)
    assert selected in ['A', 'B', 'C']


def test_hybrid_fca_topology_selector_weights():
    """HybridFCATopologySelector debe normalizar pesos"""
    selector = HybridFCATopologySelector(fca_weight=2.0, topology_weight=3.0)
    
    # Pesos deben estar normalizados
    assert abs(selector.fca_weight + selector.topology_weight - 1.0) < 0.001
    assert abs(selector.fca_weight - 0.4) < 0.001  # 2/(2+3)
    assert abs(selector.topology_weight - 0.6) < 0.001  # 3/(2+3)


def test_hybrid_fca_topology_selector_cache():
    """HybridFCATopologySelector debe cachear análisis"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1]),
            'B': frozenset([1])
        },
        constraints=[]
    )
    
    selector = HybridFCATopologySelector()
    current_domains = {'A': [1], 'B': [1]}
    
    # Primera selección
    selector.select(csp, {}, current_domains)
    
    # Verificar caché
    csp_id = id(csp)
    assert csp_id in selector._fca_priorities or csp_id in selector._topology_priorities


def test_hybrid_fca_topology_selector_reset_cache():
    """HybridFCATopologySelector debe permitir limpiar caché"""
    csp = CSP(
        variables=['A'],
        domains={'A': frozenset([1])},
        constraints=[]
    )
    
    selector = HybridFCATopologySelector()
    current_domains = {'A': [1]}
    
    # Construir caché
    selector.select(csp, {}, current_domains)
    
    # Limpiar caché
    selector.reset_cache()
    assert len(selector._fca_cache) == 0
    assert len(selector._topology_cache) == 0
    assert len(selector._fca_priorities) == 0
    assert len(selector._topology_priorities) == 0


# ============================================================================
# Tests para AdaptiveMultiscaleSelector
# ============================================================================

def test_adaptive_multiscale_selector_basic():
    """AdaptiveMultiscaleSelector debe seleccionar variables"""
    csp = CSP(
        variables=['A', 'B'],
        domains={
            'A': frozenset([1, 2]),
            'B': frozenset([1, 2])
        },
        constraints=[
            Constraint(scope=('A', 'B'), relation=lambda a, b: a != b)
        ]
    )
    
    selector = AdaptiveMultiscaleSelector(initial_fca_weight=0.5)
    current_domains = {'A': [1, 2], 'B': [1, 2]}
    
    selected = selector.select(csp, {}, current_domains)
    assert selected in ['A', 'B']


def test_adaptive_multiscale_selector_adapts_weights():
    """AdaptiveMultiscaleSelector debe ajustar pesos dinámicamente"""
    selector = AdaptiveMultiscaleSelector(initial_fca_weight=0.5)
    
    # Pesos iniciales
    initial_fca = selector.fca_weight
    initial_topo = selector.topology_weight
    
    # Registrar éxitos de FCA
    for _ in range(10):
        selector.record_success('fca')
    
    # Ajustar pesos (se hace cada 10 selecciones)
    selector._adjust_weights()
    
    # FCA debe tener mayor peso ahora
    assert selector.fca_weight > initial_fca


def test_adaptive_multiscale_selector_reset_adaptation():
    """AdaptiveMultiscaleSelector debe permitir reiniciar adaptación"""
    selector = AdaptiveMultiscaleSelector()
    
    # Registrar éxitos
    selector.record_success('fca')
    selector.record_success('topology')
    
    # Reiniciar
    selector.reset_adaptation()
    
    assert selector._fca_successes == 0
    assert selector._topology_successes == 0
    assert selector._total_selections == 0
    assert selector.fca_weight == 0.5
    assert selector.topology_weight == 0.5


# ============================================================================
# Tests de Edge Cases
# ============================================================================

def test_empty_csp():
    """Todas las estrategias deben manejar CSP vacío"""
    csp = CSP(variables=[], domains={}, constraints=[])
    
    selectors = [
        TopologyGuidedSelector(),
        ComponentBasedSelector(),
        HybridFCATopologySelector(),
        AdaptiveMultiscaleSelector()
    ]
    
    for selector in selectors:
        selected = selector.select(csp, {}, {})
        assert selected is None


def test_single_variable_csp():
    """Todas las estrategias deben manejar CSP con una variable"""
    csp = CSP(
        variables=['A'],
        domains={'A': frozenset([1, 2, 3])},
        constraints=[]
    )
    
    selectors = [
        TopologyGuidedSelector(),
        ComponentBasedSelector(),
        HybridFCATopologySelector(),
        AdaptiveMultiscaleSelector()
    ]
    
    current_domains = {'A': [1, 2, 3]}
    
    for selector in selectors:
        selected = selector.select(csp, {}, current_domains)
        assert selected == 'A'


def test_no_constraints_csp():
    """Todas las estrategias deben manejar CSP sin restricciones"""
    csp = CSP(
        variables=['A', 'B', 'C'],
        domains={
            'A': frozenset([1]),
            'B': frozenset([1]),
            'C': frozenset([1])
        },
        constraints=[]
    )
    
    selectors = [
        TopologyGuidedSelector(),
        ComponentBasedSelector(),
        HybridFCATopologySelector(),
        AdaptiveMultiscaleSelector()
    ]
    
    current_domains = {'A': [1], 'B': [1], 'C': [1]}
    
    for selector in selectors:
        selected = selector.select(csp, {}, current_domains)
        assert selected in ['A', 'B', 'C']

