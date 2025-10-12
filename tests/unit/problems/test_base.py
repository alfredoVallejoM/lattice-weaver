"""
Tests unitarios para la clase base ProblemFamily.
"""

import pytest
from lattice_weaver.problems.base import ProblemFamily


class MockProblemFamily(ProblemFamily):
    """Implementación mock de ProblemFamily para testing."""
    
    def __init__(self):
        super().__init__(
            name='mock_problem',
            description='Mock problem for testing'
        )
    
    def generate(self, **params):
        """Mock implementation."""
        return None
    
    def validate_solution(self, solution, **params):
        """Mock implementation."""
        return True
    
    def get_metadata(self, **params):
        """Mock implementation."""
        return {'family': self.name}
    
    def get_param_schema(self):
        """Schema for testing."""
        return {
            'n': {
                'type': int,
                'required': True,
                'min': 1,
                'max': 100,
                'description': 'Test parameter'
            },
            'mode': {
                'type': str,
                'required': False,
                'default': 'normal',
                'choices': ['easy', 'normal', 'hard'],
                'description': 'Difficulty mode'
            }
        }


class TestProblemFamily:
    """Tests para la clase base ProblemFamily."""
    
    def test_initialization(self):
        """Test que la inicialización funciona correctamente."""
        family = MockProblemFamily()
        assert family.name == 'mock_problem'
        assert family.description == 'Mock problem for testing'
    
    def test_get_default_params(self):
        """Test que get_default_params retorna dict vacío por defecto."""
        family = MockProblemFamily()
        defaults = family.get_default_params()
        assert isinstance(defaults, dict)
        assert len(defaults) == 0
    
    def test_validate_params_valid(self):
        """Test que validate_params acepta parámetros válidos."""
        family = MockProblemFamily()
        # No debe lanzar excepción
        family.validate_params(n=50, mode='normal')
    
    def test_validate_params_missing_required(self):
        """Test que validate_params rechaza parámetros requeridos faltantes."""
        family = MockProblemFamily()
        with pytest.raises(ValueError, match="Parámetro requerido faltante: n"):
            family.validate_params(mode='easy')
    
    def test_validate_params_wrong_type(self):
        """Test que validate_params rechaza tipos incorrectos."""
        family = MockProblemFamily()
        with pytest.raises(ValueError, match="debe ser de tipo int"):
            family.validate_params(n='invalid')
    
    def test_validate_params_out_of_range_min(self):
        """Test que validate_params rechaza valores fuera de rango (mínimo)."""
        family = MockProblemFamily()
        with pytest.raises(ValueError, match="debe ser >= 1"):
            family.validate_params(n=0)
    
    def test_validate_params_out_of_range_max(self):
        """Test que validate_params rechaza valores fuera de rango (máximo)."""
        family = MockProblemFamily()
        with pytest.raises(ValueError, match="debe ser <= 100"):
            family.validate_params(n=101)
    
    def test_validate_params_invalid_choice(self):
        """Test que validate_params rechaza opciones inválidas."""
        family = MockProblemFamily()
        with pytest.raises(ValueError, match="debe ser uno de"):
            family.validate_params(n=50, mode='invalid')
    
    def test_validate_params_valid_choice(self):
        """Test que validate_params acepta opciones válidas."""
        family = MockProblemFamily()
        family.validate_params(n=50, mode='easy')
        family.validate_params(n=50, mode='normal')
        family.validate_params(n=50, mode='hard')
    
    def test_repr(self):
        """Test que __repr__ retorna string correcto."""
        family = MockProblemFamily()
        assert repr(family) == '<ProblemFamily: mock_problem>'
    
    def test_str(self):
        """Test que __str__ retorna string legible."""
        family = MockProblemFamily()
        assert str(family) == 'mock_problem: Mock problem for testing'


class TestProblemFamilyAbstract:
    """Tests para verificar que ProblemFamily es abstracta."""
    
    def test_cannot_instantiate_directly(self):
        """Test que no se puede instanciar ProblemFamily directamente."""
        with pytest.raises(TypeError):
            ProblemFamily('test', 'test description')
    
    def test_must_implement_generate(self):
        """Test que subclases deben implementar generate()."""
        class IncompleteFamily(ProblemFamily):
            def validate_solution(self, solution, **params):
                return True
            def get_metadata(self, **params):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteFamily('test', 'test')
    
    def test_must_implement_validate_solution(self):
        """Test que subclases deben implementar validate_solution()."""
        class IncompleteFamily(ProblemFamily):
            def generate(self, **params):
                return None
            def get_metadata(self, **params):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteFamily('test', 'test')
    
    def test_must_implement_get_metadata(self):
        """Test que subclases deben implementar get_metadata()."""
        class IncompleteFamily(ProblemFamily):
            def generate(self, **params):
                return None
            def validate_solution(self, solution, **params):
                return True
        
        with pytest.raises(TypeError):
            IncompleteFamily('test', 'test')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

