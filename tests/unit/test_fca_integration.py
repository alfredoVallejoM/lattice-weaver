"""
Tests para integración FCA con CSPSolver

Este módulo prueba:
1. Adaptador CSP-to-FCA
2. Analizador FCA
3. Estrategias FCA-guided
4. Integración completa con CSPSolver

Autor: Manus AI
Fecha: 15 de Octubre, 2025
"""

import pytest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.fca_adapter import CSPToFCAAdapter, analyze_csp_structure
from lattice_weaver.core.csp_engine.fca_analyzer import FCAAnalyzer, analyze_csp_with_fca
from lattice_weaver.core.csp_engine.strategies.fca_guided import (
    FCAGuidedSelector,
    FCAOnlySelector,
    FCAClusterSelector
)


# ============================================================================
# Fixtures: Problemas CSP de prueba
# ============================================================================

@pytest.fixture
def simple_csp():
    """CSP simple: 3 variables, dominios {1,2,3}, restricciones de desigualdad."""
    variables = ['A', 'B', 'C']
    domains = {var: frozenset([1, 2, 3]) for var in variables}
    constraints = [
        Constraint(scope=frozenset(['A', 'B']), relation=lambda a, b: a != b),
        Constraint(scope=frozenset(['B', 'C']), relation=lambda b, c: b != c),
    ]
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


@pytest.fixture
def varied_domains_csp():
    """CSP con dominios de diferentes tamaños."""
    variables = ['A', 'B', 'C', 'D']
    domains = {
        'A': frozenset([1]),  # domain_size_1
        'B': frozenset([1, 2]),  # domain_size_small
        'C': frozenset(range(10)),  # domain_size_medium
        'D': frozenset(range(50))  # domain_size_large
    }
    constraints = [
        Constraint(scope=frozenset(['A', 'B']), relation=lambda a, b: a != b),
        Constraint(scope=frozenset(['B', 'C']), relation=lambda b, c: b < c),
        Constraint(scope=frozenset(['C', 'D']), relation=lambda c, d: c < d),
    ]
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


@pytest.fixture
def nqueens_4():
    """N-Queens 4x4."""
    n = 4
    variables = [f'Q{i}' for i in range(n)]
    domains = {var: frozenset(range(n)) for var in variables}
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            # No misma columna
            constraints.append(
                Constraint(
                    scope=frozenset([f'Q{i}', f'Q{j}']),
                    relation=lambda vi, vj, i=i, j=j: vi != vj
                )
            )
            # No misma diagonal
            constraints.append(
                Constraint(
                    scope=frozenset([f'Q{i}', f'Q{j}']),
                    relation=lambda vi, vj, i=i, j=j: abs(vi - vj) != abs(i - j)
                )
            )
    
    return CSP(variables=frozenset(variables), domains=domains, constraints=constraints)


# ============================================================================
# Tests: CSPToFCAAdapter
# ============================================================================

def test_adapter_build_context(simple_csp):
    """El adaptador debe construir un contexto formal válido."""
    adapter = CSPToFCAAdapter(simple_csp)
    context = adapter.build_context()
    
    # Verificar que se crearon objetos (variables)
    assert len(context.objects) == 3
    assert 'A' in context.objects
    assert 'B' in context.objects
    assert 'C' in context.objects
    
    # Verificar que se crearon atributos
    assert len(context.attributes) > 0
    
    # Verificar que hay incidencias
    assert len(context.incidences) > 0


def test_adapter_domain_attributes(varied_domains_csp):
    """El adaptador debe clasificar correctamente los dominios."""
    adapter = CSPToFCAAdapter(varied_domains_csp)
    context = adapter.build_context()
    
    # Verificar atributos de dominio
    assert 'domain_size_1' in context.attributes
    assert 'domain_size_small' in context.attributes
    assert 'domain_size_medium' in context.attributes
    assert 'domain_size_large' in context.attributes
    
    # Verificar incidencias correctas
    assert ('A', 'domain_size_1') in context.incidences
    assert ('B', 'domain_size_small') in context.incidences
    assert ('C', 'domain_size_medium') in context.incidences
    assert ('D', 'domain_size_large') in context.incidences


def test_adapter_degree_attributes(simple_csp):
    """El adaptador debe calcular correctamente los grados."""
    adapter = CSPToFCAAdapter(simple_csp)
    context = adapter.build_context()
    
    # Verificar que se calcularon los grados
    assert adapter.get_degree('A') == 1  # Solo en restricción con B
    assert adapter.get_degree('B') == 2  # En restricciones con A y C
    assert adapter.get_degree('C') == 1  # Solo en restricción con B


def test_adapter_build_lattice(simple_csp):
    """El adaptador debe construir un retículo de conceptos."""
    adapter = CSPToFCAAdapter(simple_csp)
    concepts = adapter.build_lattice()
    
    # Debe haber al menos un concepto
    assert len(concepts) > 0
    
    # Cada concepto debe ser una tupla (extent, intent)
    for extent, intent in concepts:
        assert isinstance(extent, frozenset)
        assert isinstance(intent, frozenset)


def test_adapter_extract_implications(simple_csp):
    """El adaptador debe extraer implicaciones del retículo."""
    adapter = CSPToFCAAdapter(simple_csp)
    implications = adapter.extract_implications()
    
    # Debe retornar una lista (puede estar vacía)
    assert isinstance(implications, list)
    
    # Cada implicación debe ser una tupla (antecedente, consecuente)
    for antecedent, consequent in implications:
        assert isinstance(antecedent, frozenset)
        assert isinstance(consequent, frozenset)


def test_adapter_get_summary(simple_csp):
    """El adaptador debe generar un resumen del análisis."""
    adapter = CSPToFCAAdapter(simple_csp)
    summary = adapter.get_summary()
    
    # Verificar campos del resumen
    assert 'num_variables' in summary
    assert 'num_attributes' in summary
    assert 'num_concepts' in summary
    assert 'num_implications' in summary
    assert 'avg_degree' in summary
    assert 'max_degree' in summary
    assert 'min_degree' in summary
    
    # Verificar valores
    assert summary['num_variables'] == 3
    assert summary['avg_degree'] > 0


def test_analyze_csp_structure_function(simple_csp):
    """La función de conveniencia debe funcionar correctamente."""
    summary = analyze_csp_structure(simple_csp)
    
    assert isinstance(summary, dict)
    assert 'num_variables' in summary
    assert summary['num_variables'] == 3


# ============================================================================
# Tests: FCAAnalyzer
# ============================================================================

def test_analyzer_analyze(simple_csp):
    """El analizador debe realizar análisis completo."""
    analyzer = FCAAnalyzer(simple_csp)
    analysis = analyzer.analyze()
    
    # Verificar campos del análisis
    assert 'num_concepts' in analysis
    assert 'num_implications' in analysis
    assert 'implications' in analysis
    assert 'variable_clusters' in analysis
    assert 'critical_variables' in analysis
    assert 'redundant_pairs' in analysis
    assert 'variable_priorities' in analysis
    assert 'summary' in analysis


def test_analyzer_cluster_variables(varied_domains_csp):
    """El analizador debe agrupar variables similares."""
    analyzer = FCAAnalyzer(varied_domains_csp)
    analyzer.analyze()
    
    clusters = analyzer.get_variable_clusters()
    
    # Debe retornar una lista
    assert isinstance(clusters, list)
    
    # Cada cluster debe ser un frozenset
    for cluster in clusters:
        assert isinstance(cluster, frozenset)


def test_analyzer_identify_critical_variables(varied_domains_csp):
    """El analizador debe identificar variables críticas."""
    analyzer = FCAAnalyzer(varied_domains_csp)
    analyzer.analyze()
    
    critical_vars = analyzer.get_critical_variables()
    
    # Debe retornar una lista
    assert isinstance(critical_vars, list)
    assert len(critical_vars) == 4  # Todas las variables
    
    # 'A' debe ser la más crítica (dominio de tamaño 1)
    assert critical_vars[0] == 'A'


def test_analyzer_compute_priorities(simple_csp):
    """El analizador debe calcular prioridades de variables."""
    analyzer = FCAAnalyzer(simple_csp)
    analyzer.analyze()
    
    # Obtener prioridad de cada variable
    priority_a = analyzer.get_variable_priority('A')
    priority_b = analyzer.get_variable_priority('B')
    priority_c = analyzer.get_variable_priority('C')
    
    # Todas deben tener prioridad > 0
    assert priority_a > 0
    assert priority_b > 0
    assert priority_c > 0
    
    # B debe tener mayor prioridad (mayor grado)
    assert priority_b > priority_a
    assert priority_b > priority_c


def test_analyzer_suggest_ordering(simple_csp):
    """El analizador debe sugerir un ordenamiento de variables."""
    analyzer = FCAAnalyzer(simple_csp)
    analyzer.analyze()
    
    ordering = analyzer.suggest_variable_ordering()
    
    # Debe retornar todas las variables
    assert len(ordering) == 3
    assert set(ordering) == {'A', 'B', 'C'}
    
    # B debe estar primero (mayor prioridad)
    assert ordering[0] == 'B'


def test_analyzer_get_analysis_summary(simple_csp):
    """El analizador debe generar un resumen textual."""
    analyzer = FCAAnalyzer(simple_csp)
    analyzer.analyze()
    
    summary = analyzer.get_analysis_summary()
    
    # Debe retornar un string
    assert isinstance(summary, str)
    assert len(summary) > 0
    
    # Debe contener información clave
    assert 'Variables:' in summary
    assert 'Conceptos formales:' in summary


def test_analyze_csp_with_fca_function(simple_csp):
    """La función de conveniencia debe funcionar correctamente."""
    analyzer = analyze_csp_with_fca(simple_csp)
    
    assert isinstance(analyzer, FCAAnalyzer)
    assert analyzer._analysis_cache is not None


# ============================================================================
# Tests: FCAGuidedSelector
# ============================================================================

def test_fca_guided_selector_basic(simple_csp):
    """FCAGuidedSelector debe seleccionar una variable válida."""
    selector = FCAGuidedSelector()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    var = selector.select(simple_csp, assignment, domains)
    
    # Debe retornar una variable válida
    assert var in simple_csp.variables
    assert var not in assignment


def test_fca_guided_selector_all_assigned(simple_csp):
    """FCAGuidedSelector debe retornar None cuando todas están asignadas."""
    selector = FCAGuidedSelector()
    assignment = {'A': 1, 'B': 2, 'C': 3}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    var = selector.select(simple_csp, assignment, domains)
    
    assert var is None


def test_fca_guided_selector_prioritizes_small_domains(varied_domains_csp):
    """FCAGuidedSelector debe priorizar dominios pequeños."""
    selector = FCAGuidedSelector()
    assignment = {}
    domains = {var: list(varied_domains_csp.domains[var]) for var in varied_domains_csp.variables}
    
    var = selector.select(varied_domains_csp, assignment, domains)
    
    # Debe seleccionar 'A' (dominio de tamaño 1)
    assert var == 'A'


def test_fca_guided_selector_cache(simple_csp):
    """FCAGuidedSelector debe cachear el análisis FCA."""
    selector = FCAGuidedSelector(use_cache=True)
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    # Primera llamada
    var1 = selector.select(simple_csp, assignment, domains)
    analyzer1 = selector._analyzer_cache
    
    # Segunda llamada
    var2 = selector.select(simple_csp, assignment, domains)
    analyzer2 = selector._analyzer_cache
    
    # Debe reutilizar el mismo analizador
    assert analyzer1 is analyzer2


def test_fca_guided_selector_reset_cache(simple_csp):
    """FCAGuidedSelector debe poder resetear el cache."""
    selector = FCAGuidedSelector()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    # Realizar selección
    selector.select(simple_csp, assignment, domains)
    assert selector._analyzer_cache is not None
    
    # Resetear cache
    selector.reset_cache()
    assert selector._analyzer_cache is None


# ============================================================================
# Tests: FCAOnlySelector
# ============================================================================

def test_fca_only_selector_basic(simple_csp):
    """FCAOnlySelector debe seleccionar una variable válida."""
    selector = FCAOnlySelector()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    var = selector.select(simple_csp, assignment, domains)
    
    assert var in simple_csp.variables
    assert var not in assignment


def test_fca_only_selector_uses_fca_priorities(simple_csp):
    """FCAOnlySelector debe usar solo prioridades FCA."""
    selector = FCAOnlySelector()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    var = selector.select(simple_csp, assignment, domains)
    
    # Debe seleccionar 'B' (mayor grado, mayor prioridad FCA)
    assert var == 'B'


# ============================================================================
# Tests: FCAClusterSelector
# ============================================================================

def test_fca_cluster_selector_basic(simple_csp):
    """FCAClusterSelector debe seleccionar una variable válida."""
    selector = FCAClusterSelector()
    assignment = {}
    domains = {var: list(simple_csp.domains[var]) for var in simple_csp.variables}
    
    var = selector.select(simple_csp, assignment, domains)
    
    assert var in simple_csp.variables
    assert var not in assignment


def test_fca_cluster_selector_processes_clusters(nqueens_4):
    """FCAClusterSelector debe procesar clusters secuencialmente."""
    selector = FCAClusterSelector()
    assignment = {}
    domains = {var: list(nqueens_4.domains[var]) for var in nqueens_4.variables}
    
    # Seleccionar varias variables
    selected = []
    for _ in range(4):
        var = selector.select(nqueens_4, assignment, domains)
        if var:
            selected.append(var)
            assignment[var] = 0  # Asignar dummy value
    
    # Debe haber seleccionado todas las variables
    assert len(selected) == 4


# ============================================================================
# Tests: Edge Cases
# ============================================================================

def test_empty_csp():
    """Debe manejar CSP sin variables."""
    variables = []
    domains = {}
    constraints = []
    csp = CSP(variables=frozenset(variables), domains=domains, constraints=constraints)
    
    adapter = CSPToFCAAdapter(csp)
    context = adapter.build_context()
    
    assert len(context.objects) == 0


def test_single_variable_csp():
    """Debe manejar CSP de una sola variable."""
    variables = ['A']
    domains = {'A': frozenset([1, 2, 3])}
    constraints = []
    csp = CSP(variables=frozenset(variables), domains=domains, constraints=constraints)
    
    selector = FCAGuidedSelector()
    assignment = {}
    domains_list = {var: list(csp.domains[var]) for var in csp.variables}
    
    var = selector.select(csp, assignment, domains_list)
    
    assert var == 'A'


def test_no_constraints_csp():
    """Debe manejar CSP sin restricciones."""
    variables = ['A', 'B', 'C']
    domains = {var: frozenset([1, 2, 3]) for var in variables}
    constraints = []
    csp = CSP(variables=frozenset(variables), domains=domains, constraints=constraints)
    
    analyzer = FCAAnalyzer(csp)
    analysis = analyzer.analyze()
    
    # Todas las variables deben tener grado 0
    for var in variables:
        assert analyzer.adapter.get_degree(var) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

