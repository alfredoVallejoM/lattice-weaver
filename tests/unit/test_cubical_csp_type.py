"""
Tests Unitarios para CubicalCSPType

Verifica la correcta traducción de problemas CSP a tipos cúbicos.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import pytest
from lattice_weaver.formal.cubical_csp_type import (
    CubicalCSPType,
    FiniteType,
    PropositionType,
    UnitType
)


class TestFiniteType:
    """Tests para FiniteType."""
    
    def test_create_finite_type(self):
        """Test: Crear tipo finito básico."""
        ft = FiniteType("Domain_X", frozenset({1, 2, 3}))
        assert ft.name == "Domain_X"
        assert len(ft.values) == 3
        assert 1 in ft.values
        assert 2 in ft.values
        assert 3 in ft.values
    
    def test_finite_type_str(self):
        """Test: Representación en string de tipo finito."""
        ft = FiniteType("Domain_X", frozenset({1, 2, 3}))
        str_repr = str(ft)
        assert "{" in str_repr
        assert "}" in str_repr
        # Debe contener todos los valores
        assert "1" in str_repr
        assert "2" in str_repr
        assert "3" in str_repr
    
    def test_finite_type_immutable(self):
        """Test: Tipo finito es inmutable."""
        ft = FiniteType("Domain_X", frozenset({1, 2, 3}))
        with pytest.raises(Exception):  # dataclass frozen
            ft.name = "NewName"
    
    def test_finite_type_hashable(self):
        """Test: Tipo finito es hashable."""
        ft1 = FiniteType("Domain_X", frozenset({1, 2, 3}))
        ft2 = FiniteType("Domain_X", frozenset({1, 2, 3}))
        assert hash(ft1) == hash(ft2)
        
        # Puede usarse en sets
        type_set = {ft1, ft2}
        assert len(type_set) == 1


class TestPropositionType:
    """Tests para PropositionType."""
    
    def test_create_proposition(self):
        """Test: Crear proposición básica."""
        prop = PropositionType("X_lt_Y", ("X", "Y"), lambda x, y: x < y)
        assert prop.constraint_name == "X_lt_Y"
        assert prop.variables == ("X", "Y")
    
    def test_proposition_check_true(self):
        """Test: Verificar proposición satisfecha."""
        prop = PropositionType("X_lt_Y", ("X", "Y"), lambda x, y: x < y)
        assert prop.check(1, 2) is True
        assert prop.check(1, 3) is True
    
    def test_proposition_check_false(self):
        """Test: Verificar proposición no satisfecha."""
        prop = PropositionType("X_lt_Y", ("X", "Y"), lambda x, y: x < y)
        assert prop.check(2, 1) is False
        assert prop.check(2, 2) is False
    
    def test_proposition_str(self):
        """Test: Representación en string de proposición."""
        prop = PropositionType("X_lt_Y", ("X", "Y"), lambda x, y: x < y)
        str_repr = str(prop)
        assert "X_lt_Y" in str_repr
        assert "X" in str_repr
        assert "Y" in str_repr


class TestCubicalCSPType:
    """Tests para CubicalCSPType."""
    
    def test_construct_from_simple_csp(self):
        """Test: Construir tipo desde CSP simple."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y, 'name': 'X_lt_Y'}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert csp_type.variables == ['X', 'Y']
        assert len(csp_type.domain_types) == 2
        assert len(csp_type.constraint_props) == 1
        assert csp_type.solution_type is not None
    
    def test_construct_from_complex_csp(self):
        """Test: Construir tipo desde CSP complejo (3 variables)."""
        variables = ['X', 'Y', 'Z']
        domains = {
            'X': {1, 2, 3},
            'Y': {1, 2, 3},
            'Z': {1, 2, 3}
        }
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y, 'name': 'X_lt_Y'},
            {'variables': ['Y', 'Z'], 'predicate': lambda y, z: y < z, 'name': 'Y_lt_Z'}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert len(csp_type.variables) == 3
        assert len(csp_type.constraint_props) == 2
        assert csp_type.get_domain_size() == 27  # 3^3
    
    def test_verify_solution_valid(self):
        """Test: Verificar solución válida."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        solution = {'X': 1, 'Y': 2}
        assert csp_type.verify_solution(solution) is True
        
        solution = {'X': 1, 'Y': 3}
        assert csp_type.verify_solution(solution) is True
    
    def test_verify_solution_invalid(self):
        """Test: Rechazar solución inválida."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        solution = {'X': 2, 'Y': 1}
        assert csp_type.verify_solution(solution) is False
        
        solution = {'X': 2, 'Y': 2}
        assert csp_type.verify_solution(solution) is False
    
    def test_synthesize_term_from_solution(self):
        """Test: Sintetizar término desde solución."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        solution = {'X': 1, 'Y': 2}
        term = csp_type.synthesize_term(solution)
        
        assert term is not None
        # El término debe ser un par anidado
        assert str(term) == "(1, (2, ()))"
    
    def test_synthesize_term_invalid_solution(self):
        """Test: Sintetizar término desde solución inválida debe fallar."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        solution = {'X': 2, 'Y': 1}  # Inválida
        with pytest.raises(ValueError):
            csp_type.synthesize_term(solution)
    
    def test_domain_types_correct(self):
        """Test: Tipos de dominios correctos."""
        variables = ['X', 'Y']
        domains = {
            'X': {1, 2, 3},
            'Y': {4, 5, 6}
        }
        constraints = []
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert 'X' in csp_type.domain_types
        assert 'Y' in csp_type.domain_types
        
        domain_x = csp_type.domain_types['X']
        assert 1 in domain_x.values
        assert 2 in domain_x.values
        assert 3 in domain_x.values
        
        domain_y = csp_type.domain_types['Y']
        assert 4 in domain_y.values
        assert 5 in domain_y.values
        assert 6 in domain_y.values
    
    def test_constraint_props_correct(self):
        """Test: Proposiciones de restricciones correctas."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y, 'name': 'X_lt_Y'},
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x != y, 'name': 'X_ne_Y'}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert len(csp_type.constraint_props) == 2
        assert csp_type.constraint_props[0].constraint_name == 'X_lt_Y'
        assert csp_type.constraint_props[1].constraint_name == 'X_ne_Y'
    
    def test_solution_type_structure(self):
        """Test: Estructura del tipo Sigma."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2}, 'Y': {1, 2}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x != y}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        # El tipo solución debe ser un SigmaType
        from lattice_weaver.formal.cubical_syntax import SigmaType
        assert isinstance(csp_type.solution_type, SigmaType)
        
        # Debe tener la estructura correcta
        str_repr = str(csp_type.solution_type)
        assert "Σ" in str_repr or "(" in str_repr  # Sigma o producto
    
    def test_get_domain_size(self):
        """Test: Cálculo del tamaño del espacio de búsqueda."""
        variables = ['X', 'Y', 'Z']
        domains = {
            'X': {1, 2},
            'Y': {1, 2, 3},
            'Z': {1, 2, 3, 4}
        }
        constraints = []
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert csp_type.get_domain_size() == 2 * 3 * 4  # 24
    
    def test_get_constraint_count(self):
        """Test: Conteo de restricciones."""
        variables = ['X', 'Y', 'Z']
        domains = {'X': {1, 2}, 'Y': {1, 2}, 'Z': {1, 2}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x != y},
            {'variables': ['Y', 'Z'], 'predicate': lambda y, z: y != z},
            {'variables': ['X', 'Z'], 'predicate': lambda x, z: x != z}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert csp_type.get_constraint_count() == 3
    
    def test_empty_csp(self):
        """Test: CSP vacío (sin variables)."""
        variables = []
        domains = {}
        constraints = []
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert len(csp_type.variables) == 0
        assert csp_type.get_domain_size() == 1  # Espacio trivial
        assert isinstance(csp_type.solution_type, UnitType)
    
    def test_csp_without_constraints(self):
        """Test: CSP sin restricciones."""
        variables = ['X', 'Y']
        domains = {'X': {1, 2}, 'Y': {1, 2}}
        constraints = []
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        assert len(csp_type.constraint_props) == 0
        
        # Todas las soluciones deben ser válidas
        assert csp_type.verify_solution({'X': 1, 'Y': 1}) is True
        assert csp_type.verify_solution({'X': 1, 'Y': 2}) is True
        assert csp_type.verify_solution({'X': 2, 'Y': 1}) is True
        assert csp_type.verify_solution({'X': 2, 'Y': 2}) is True
    
    def test_multiple_constraints_all_satisfied(self):
        """Test: Múltiples restricciones todas satisfechas."""
        variables = ['X', 'Y', 'Z']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y},
            {'variables': ['Y', 'Z'], 'predicate': lambda y, z: y < z}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        solution = {'X': 1, 'Y': 2, 'Z': 3}
        assert csp_type.verify_solution(solution) is True
    
    def test_multiple_constraints_one_fails(self):
        """Test: Múltiples restricciones, una falla."""
        variables = ['X', 'Y', 'Z']
        domains = {'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}}
        constraints = [
            {'variables': ['X', 'Y'], 'predicate': lambda x, y: x < y},
            {'variables': ['Y', 'Z'], 'predicate': lambda y, z: y < z}
        ]
        
        csp_type = CubicalCSPType.from_csp_problem(variables, domains, constraints)
        
        # Primera restricción OK, segunda falla
        solution = {'X': 1, 'Y': 2, 'Z': 1}
        assert csp_type.verify_solution(solution) is False



if __name__ == '__main__':
    pytest.main([__file__, '-v'])

