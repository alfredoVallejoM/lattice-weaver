"""
Tests para Interpretación Lógica Completa de CSP en HoTT

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lattice_weaver.formal.csp_logic_interpretation import *
from lattice_weaver.formal.csp_integration import CSPProblem, CSPSolution


def test_domain_interpretation_propositional():
    """Test: Interpretación proposicional de dominios."""
    print("\n" + "="*60)
    print("TEST 1: Interpretación Proposicional de Dominios")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROPOSITIONAL)
    
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2, 3}, 'y': {1, 2}},
        constraints=[]
    )
    
    # Interpretar dominio
    x_interp = interpreter.interpret_domain('x', {1, 2, 3}, problem)
    y_interp = interpreter.interpret_domain('y', {1, 2}, problem)
    
    print(f"Dominio x: {x_interp.domain_type}")
    print(f"  Elementos: {len(x_interp.elements)}")
    print(f"  Axiomas: {len(x_interp.axioms)}")
    print(f"  Semántica: {x_interp.semantics.value}")
    
    print(f"\nDominio y: {y_interp.domain_type}")
    print(f"  Elementos: {len(y_interp.elements)}")
    
    assert len(x_interp.elements) == 3
    assert len(y_interp.elements) == 2
    assert x_interp.semantics == CSPSemantics.PROPOSITIONAL
    
    print("✅ Test pasado")


def test_domain_interpretation_proof_relevant():
    """Test: Interpretación proof-relevant de dominios."""
    print("\n" + "="*60)
    print("TEST 2: Interpretación Proof-Relevant de Dominios")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROOF_RELEVANT)
    
    problem = CSPProblem(
        variables=['x'],
        domains={'x': {1, 2, 3}},
        constraints=[]
    )
    
    x_interp = interpreter.interpret_domain('x', {1, 2, 3}, problem)
    
    print(f"Dominio x: {x_interp.domain_type}")
    print(f"  Tipo: Sigma type con pruebas de pertenencia")
    print(f"  Elementos: {len(x_interp.elements)}")
    print(f"  Semántica: {x_interp.semantics.value}")
    
    # Verificar que los elementos son pares (valor, prueba)
    for val, term in x_interp.elements.items():
        print(f"  {val} → {term}")
        assert isinstance(term, Pair)
    
    assert x_interp.semantics == CSPSemantics.PROOF_RELEVANT
    
    print("✅ Test pasado")


def test_constraint_interpretation():
    """Test: Interpretación de restricciones."""
    print("\n" + "="*60)
    print("TEST 3: Interpretación de Restricciones")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROPOSITIONAL)
    
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2}, 'y': {1, 2}},
        constraints=[('x', 'y', lambda a, b: a != b)]
    )
    
    # Interpretar restricción
    constraint_interp = interpreter.interpret_constraint(
        'x', 'y', lambda a, b: a != b, problem
    )
    
    print(f"Restricción x ≠ y:")
    print(f"  Tipo: {constraint_interp.constraint_type}")
    print(f"  Semántica: {constraint_interp.semantics.value}")
    
    # Verificar constructor de pruebas
    proof_valid = constraint_interp.proof_constructor(1, 2)
    proof_invalid = constraint_interp.proof_constructor(1, 1)
    
    print(f"\n  Prueba (1, 2): {proof_valid}")
    print(f"  Prueba (1, 1): {proof_invalid}")
    
    assert proof_valid is not None
    assert proof_invalid is None
    
    print("✅ Test pasado")


def test_curry_howard_correspondence():
    """Test: Correspondencia Curry-Howard."""
    print("\n" + "="*60)
    print("TEST 4: Correspondencia Curry-Howard")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROOF_RELEVANT)
    
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2}, 'y': {1, 2}},
        constraints=[('x', 'y', lambda a, b: a < b)]
    )
    
    # Obtener correspondencia
    correspondence = interpreter.curry_howard_correspondence(problem)
    
    print("Correspondencia Curry-Howard:")
    print(f"  Semántica: {correspondence['semantics']}")
    print(f"  Dominios → Tipos: {len(correspondence['domains_to_types'])}")
    print(f"  Restricciones → Proposiciones: {len(correspondence['constraints_to_propositions'])}")
    print(f"  Soluciones → Pruebas: {correspondence['solutions_to_proofs']}")
    print(f"  Arc-consistency → {correspondence['arc_consistency_to']}")
    print(f"  Backtracking → {correspondence['backtracking_to']}")
    
    assert 'x' in correspondence['domains_to_types']
    assert 'y' in correspondence['domains_to_types']
    assert 'x_y' in correspondence['constraints_to_propositions']
    
    print("✅ Test pasado")


def test_propagation_interpretation():
    """Test: Interpretación de propagación."""
    print("\n" + "="*60)
    print("TEST 5: Interpretación de Propagación")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROPOSITIONAL)
    
    problem = CSPProblem(
        variables=['x'],
        domains={'x': {1, 2, 3}},
        constraints=[]
    )
    
    # Simular propagación: {1, 2, 3} → {1, 2}
    domain_before = {1, 2, 3}
    domain_after = {1, 2}
    
    prop_interp = interpreter.interpret_arc_consistency_step(
        'x', domain_before, domain_after, problem
    )
    
    print("Propagación de arc-consistency:")
    print(f"  Tipo antes: {prop_interp.before_type}")
    print(f"  Tipo después: {prop_interp.after_type}")
    print(f"  Transformación: {prop_interp.transformation}")
    print(f"  Prueba de correctitud: {prop_interp.correctness_proof is not None}")
    
    assert prop_interp.correctness_proof is not None
    
    print("✅ Test pasado")


def test_compare_semantics():
    """Test: Comparación de semánticas."""
    print("\n" + "="*60)
    print("TEST 6: Comparación de Semánticas")
    print("="*60)
    
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2}, 'y': {1, 2}},
        constraints=[('x', 'y', lambda a, b: a != b)]
    )
    
    # Comparar todas las semánticas
    comparison = compare_semantics(problem)
    
    print("Comparación de semánticas:")
    for semantics_name, data in comparison.items():
        print(f"\n{semantics_name.upper()}:")
        stats = data['statistics']
        print(f"  Dominios interpretados: {stats['domains_interpreted']}")
        print(f"  Restricciones interpretadas: {stats['constraints_interpreted']}")
        print(f"  Axiomas totales: {stats['total_axioms']}")
    
    assert len(comparison) == 4  # 4 semánticas
    assert all('statistics' in data for data in comparison.values())
    
    print("\n✅ Test pasado")


def test_homotopical_interpretation():
    """Test: Interpretación homotópica."""
    print("\n" + "="*60)
    print("TEST 7: Interpretación Homotópica")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.HOMOTOPICAL)
    
    problem = CSPProblem(
        variables=['x'],
        domains={'x': {1, 2, 3}},
        constraints=[]
    )
    
    x_interp = interpreter.interpret_domain('x', {1, 2, 3}, problem)
    
    print(f"Interpretación homotópica de dominio:")
    print(f"  Tipo: {x_interp.domain_type}")
    print(f"  Elementos como puntos: {len(x_interp.elements)}")
    print(f"  Axiomas (espacio discreto): {len(x_interp.axioms)}")
    
    # Verificar axioma de decidibilidad de igualdad
    assert len(x_interp.axioms) > 0
    assert x_interp.semantics == CSPSemantics.HOMOTOPICAL
    
    print("✅ Test pasado")


def test_categorical_interpretation():
    """Test: Interpretación categórica."""
    print("\n" + "="*60)
    print("TEST 8: Interpretación Categórica")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.CATEGORICAL)
    
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2}, 'y': {1, 2}},
        constraints=[('x', 'y', lambda a, b: a != b)]
    )
    
    # Interpretar dominio como objeto
    x_interp = interpreter.interpret_domain('x', {1, 2}, problem)
    
    print(f"Interpretación categórica:")
    print(f"  Dominio como objeto: {x_interp.domain_type}")
    print(f"  Elementos como morfismos: {len(x_interp.elements)}")
    
    # Interpretar restricción como morfismo
    constraint_interp = interpreter.interpret_constraint(
        'x', 'y', lambda a, b: a != b, problem
    )
    
    print(f"  Restricción como morfismo: {constraint_interp.constraint_type}")
    
    # Constructor devuelve términos para ambos casos
    proof_true = constraint_interp.proof_constructor(1, 2)
    proof_false = constraint_interp.proof_constructor(1, 1)
    
    print(f"  Morfismo (1, 2): {proof_true}")
    print(f"  Morfismo (1, 1): {proof_false}")
    
    assert proof_true is not None
    assert proof_false is not None  # En categórica, siempre hay morfismo
    
    print("✅ Test pasado")


def test_interpretation_statistics():
    """Test: Estadísticas de interpretación."""
    print("\n" + "="*60)
    print("TEST 9: Estadísticas de Interpretación")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROOF_RELEVANT)
    
    problem = CSPProblem(
        variables=['x', 'y', 'z'],
        domains={'x': {1, 2}, 'y': {1, 2}, 'z': {1, 2}},
        constraints=[
            ('x', 'y', lambda a, b: a != b),
            ('y', 'z', lambda a, b: a != b)
        ]
    )
    
    # Interpretar todo
    for var in problem.variables:
        interpreter.interpret_domain(var, problem.domains[var], problem)
    
    for var1, var2, relation in problem.constraints:
        interpreter.interpret_constraint(var1, var2, relation, problem)
    
    # Obtener estadísticas
    stats = interpreter.get_interpretation_statistics()
    
    print("Estadísticas de interpretación:")
    print(f"  Semántica: {stats['semantics']}")
    print(f"  Dominios interpretados: {stats['domains_interpreted']}")
    print(f"  Restricciones interpretadas: {stats['constraints_interpreted']}")
    print(f"  Propagaciones interpretadas: {stats['propagations_interpreted']}")
    print(f"  Axiomas totales: {stats['total_axioms']}")
    
    assert stats['domains_interpreted'] == 3
    assert stats['constraints_interpreted'] == 2
    
    print("✅ Test pasado")


def test_explain_interpretation():
    """Test: Explicación de interpretación."""
    print("\n" + "="*60)
    print("TEST 10: Explicación de Interpretación")
    print("="*60)
    
    interpreter = create_logic_interpreter(CSPSemantics.PROOF_RELEVANT)
    
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2}, 'y': {1, 2}},
        constraints=[('x', 'y', lambda a, b: a < b)]
    )
    
    # Generar explicación
    explanation = interpreter.explain_interpretation(problem)
    
    print("Explicación generada:")
    print("-" * 60)
    print(explanation)
    print("-" * 60)
    
    assert 'Interpretación Lógica' in explanation
    assert 'proof_relevant' in explanation
    assert 'Curry-Howard' in explanation
    
    print("✅ Test pasado")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("Tests de Interpretación Lógica Completa de CSP en HoTT")
    print("LatticeWeaver v4")
    print("="*60)
    
    tests = [
        test_domain_interpretation_propositional,
        test_domain_interpretation_proof_relevant,
        test_constraint_interpretation,
        test_curry_howard_correspondence,
        test_propagation_interpretation,
        test_compare_semantics,
        test_homotopical_interpretation,
        test_categorical_interpretation,
        test_interpretation_statistics,
        test_explain_interpretation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test falló: {e}")
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

