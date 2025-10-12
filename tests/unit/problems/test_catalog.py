"""
Tests unitarios para ProblemCatalog.
"""

import pytest
from lattice_weaver.problems.catalog import ProblemCatalog, get_catalog
from lattice_weaver.problems.base import ProblemFamily


class MockFamily1(ProblemFamily):
    """Mock family 1 para testing."""
    
    def __init__(self):
        super().__init__('mock1', 'Mock family 1')
    
    def generate(self, **params):
        return None
    
    def validate_solution(self, solution, **params):
        return True
    
    def get_metadata(self, **params):
        return {'family': self.name}


class MockFamily2(ProblemFamily):
    """Mock family 2 para testing."""
    
    def __init__(self):
        super().__init__('mock2', 'Mock family 2')
    
    def generate(self, **params):
        return None
    
    def validate_solution(self, solution, **params):
        return True
    
    def get_metadata(self, **params):
        return {'family': self.name}


class TestProblemCatalog:
    """Tests para ProblemCatalog."""
    
    def setup_method(self):
        """Setup para cada test - crear catálogo limpio."""
        self.catalog = ProblemCatalog()
    
    def test_initialization(self):
        """Test que el catálogo se inicializa vacío."""
        assert len(self.catalog) == 0
        assert self.catalog.list_families() == []
    
    def test_register_family(self):
        """Test que register() añade una familia."""
        family = MockFamily1()
        self.catalog.register(family)
        
        assert len(self.catalog) == 1
        assert 'mock1' in self.catalog
        assert self.catalog.get('mock1') == family
    
    def test_register_duplicate_raises_error(self):
        """Test que registrar duplicado lanza ValueError."""
        family1 = MockFamily1()
        family2 = MockFamily1()
        
        self.catalog.register(family1)
        
        with pytest.raises(ValueError, match="ya está registrada"):
            self.catalog.register(family2)
    
    def test_register_invalid_type_raises_error(self):
        """Test que registrar tipo inválido lanza TypeError."""
        with pytest.raises(TypeError, match="Se esperaba una instancia de ProblemFamily"):
            self.catalog.register("not a family")
    
    def test_unregister_family(self):
        """Test que unregister() elimina una familia."""
        family = MockFamily1()
        self.catalog.register(family)
        
        assert 'mock1' in self.catalog
        
        self.catalog.unregister('mock1')
        
        assert 'mock1' not in self.catalog
        assert len(self.catalog) == 0
    
    def test_unregister_nonexistent_raises_error(self):
        """Test que unregister() de familia inexistente lanza KeyError."""
        with pytest.raises(KeyError, match="no encontrada"):
            self.catalog.unregister('nonexistent')
    
    def test_get_existing_family(self):
        """Test que get() retorna familia existente."""
        family = MockFamily1()
        self.catalog.register(family)
        
        retrieved = self.catalog.get('mock1')
        assert retrieved == family
    
    def test_get_nonexistent_family(self):
        """Test que get() retorna None para familia inexistente."""
        result = self.catalog.get('nonexistent')
        assert result is None
    
    def test_has_family(self):
        """Test que has() verifica existencia correctamente."""
        family = MockFamily1()
        self.catalog.register(family)
        
        assert self.catalog.has('mock1') is True
        assert self.catalog.has('nonexistent') is False
    
    def test_list_families(self):
        """Test que list_families() retorna lista ordenada."""
        family1 = MockFamily1()
        family2 = MockFamily2()
        
        self.catalog.register(family2)  # Registrar en orden inverso
        self.catalog.register(family1)
        
        families = self.catalog.list_families()
        assert families == ['mock1', 'mock2']  # Ordenado alfabéticamente
    
    def test_get_all_families(self):
        """Test que get_all_families() retorna dict completo."""
        family1 = MockFamily1()
        family2 = MockFamily2()
        
        self.catalog.register(family1)
        self.catalog.register(family2)
        
        all_families = self.catalog.get_all_families()
        assert len(all_families) == 2
        assert all_families['mock1'] == family1
        assert all_families['mock2'] == family2
    
    def test_generate_problem(self):
        """Test que generate_problem() llama a family.generate()."""
        family = MockFamily1()
        self.catalog.register(family)
        
        # generate() retorna None en el mock
        result = self.catalog.generate_problem('mock1')
        assert result is None
    
    def test_generate_problem_nonexistent_raises_error(self):
        """Test que generate_problem() con familia inexistente lanza ValueError."""
        with pytest.raises(ValueError, match="Familia desconocida"):
            self.catalog.generate_problem('nonexistent')
    
    def test_validate_solution(self):
        """Test que validate_solution() llama a family.validate_solution()."""
        family = MockFamily1()
        self.catalog.register(family)
        
        result = self.catalog.validate_solution('mock1', {})
        assert result is True
    
    def test_validate_solution_nonexistent_raises_error(self):
        """Test que validate_solution() con familia inexistente lanza ValueError."""
        with pytest.raises(ValueError, match="Familia desconocida"):
            self.catalog.validate_solution('nonexistent', {})
    
    def test_get_metadata(self):
        """Test que get_metadata() llama a family.get_metadata()."""
        family = MockFamily1()
        self.catalog.register(family)
        
        metadata = self.catalog.get_metadata('mock1')
        assert metadata == {'family': 'mock1'}
    
    def test_get_metadata_nonexistent_raises_error(self):
        """Test que get_metadata() con familia inexistente lanza ValueError."""
        with pytest.raises(ValueError, match="Familia desconocida"):
            self.catalog.get_metadata('nonexistent')
    
    def test_clear(self):
        """Test que clear() elimina todas las familias."""
        family1 = MockFamily1()
        family2 = MockFamily2()
        
        self.catalog.register(family1)
        self.catalog.register(family2)
        
        assert len(self.catalog) == 2
        
        self.catalog.clear()
        
        assert len(self.catalog) == 0
        assert self.catalog.list_families() == []
    
    def test_len(self):
        """Test que len() retorna número correcto de familias."""
        assert len(self.catalog) == 0
        
        self.catalog.register(MockFamily1())
        assert len(self.catalog) == 1
        
        self.catalog.register(MockFamily2())
        assert len(self.catalog) == 2
    
    def test_contains(self):
        """Test que 'in' operator funciona correctamente."""
        family = MockFamily1()
        self.catalog.register(family)
        
        assert 'mock1' in self.catalog
        assert 'nonexistent' not in self.catalog
    
    def test_repr(self):
        """Test que __repr__ retorna string correcto."""
        self.catalog.register(MockFamily1())
        self.catalog.register(MockFamily2())
        
        assert repr(self.catalog) == '<ProblemCatalog: 2 familias>'


class TestGlobalCatalog:
    """Tests para el catálogo global singleton."""
    
    def test_get_catalog_returns_singleton(self):
        """Test que get_catalog() siempre retorna la misma instancia."""
        catalog1 = get_catalog()
        catalog2 = get_catalog()
        
        assert catalog1 is catalog2
    
    def test_get_catalog_is_problem_catalog(self):
        """Test que get_catalog() retorna instancia de ProblemCatalog."""
        catalog = get_catalog()
        assert isinstance(catalog, ProblemCatalog)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

