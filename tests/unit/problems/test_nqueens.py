import pytest
from lattice_weaver.problems.generators.nqueens import NQueensProblem
from lattice_weaver.core.csp_problem import CSP


class TestNQueensProblem:
    """Tests para NQueensProblem."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.family = NQueensProblem()
    
    def test_initialization(self):
        """Test que la inicialización es correcta."""
        assert self.family.name == 'nqueens'
        assert 'N-Reinas' in self.family.description
    
    def test_get_default_params(self):
        """Test que los parámetros por defecto son correctos."""
        defaults = self.family.get_default_params()
        assert defaults == {'n': 8}
    
    def test_get_param_schema(self):
        """Test que el esquema de parámetros es correcto."""
        schema = self.family.get_param_schema()
        
        assert 'n' in schema
        assert schema['n']['type'] == int
        assert schema['n']['required'] is True
        assert schema['n']['min'] == 4
        assert schema['n']['max'] == 1000
    
    def test_generate_creates_csp(self):
        """Test que generate() crea un CSP."""
        csp = self.family.generate(n=4)
        assert isinstance(csp, CSP)

    
    def test_generate_correct_number_of_variables(self):
        """Test que generate() crea el número correcto de variables."""
        for n in [4, 8, 16]:
            csp = self.family.generate(n=n)
            assert len(csp.variables) == n
    
    def test_generate_correct_variable_names(self):
        """Test que las variables tienen nombres correctos."""
        csp = self.family.generate(n=4)
        expected_vars = {'Q0', 'Q1', 'Q2', 'Q3'}
        assert set(csp.variables) == expected_vars
    
    def test_generate_correct_domains(self):
        """Test que los dominios son correctos."""
        n = 4
        csp = self.family.generate(n=n)
        
        for i in range(n):
            var_name = f'Q{i}'
            domain = csp.domains[var_name]
            assert domain == frozenset(range(n))
    
    def test_generate_correct_number_of_constraints(self):
        """Test que generate() crea el número correcto de restricciones."""
        n = 4
        csp = self.family.generate(n=n)
        
        # Número de restricciones = C(n, 2) = n*(n-1)/2
        expected_constraints = n * (n - 1) // 2
        assert len(csp.constraints) == expected_constraints
    
    def test_generate_with_different_sizes(self):
        """Test que generate() funciona con diferentes tamaños."""
        for n in [4, 5, 8, 10, 16]:
            csp = self.family.generate(n=n)
            assert len(csp.variables) == n
            assert len(csp.constraints) == n * (n - 1) // 2
    
    def test_generate_invalid_n_too_small(self):
        """Test que generate() rechaza n < 4."""
        with pytest.raises(ValueError):
            self.family.generate(n=3)
    
    def test_generate_invalid_n_too_large(self):
        """Test que generate() rechaza n > 1000."""
        with pytest.raises(ValueError):
            self.family.generate(n=1001)
    
    def test_validate_solution_correct(self):
        """Test que validate_solution() acepta solución correcta."""
        # Solución conocida para 4-Queens
        solution = {
            'Q0': 1,
            'Q1': 3,
            'Q2': 0,
            'Q3': 2
        }
        assert self.family.validate_solution(solution, n=4) is True
    
    def test_validate_solution_incorrect_same_column(self):
        """Test que validate_solution() rechaza reinas en misma columna."""
        solution = {
            'Q0': 0,
            'Q1': 0,  # Misma columna que Q0
            'Q2': 2,
            'Q3': 3
        }
        assert self.family.validate_solution(solution, n=4) is False
    
    def test_validate_solution_incorrect_same_diagonal(self):
        """Test que validate_solution() rechaza reinas en misma diagonal."""
        solution = {
            'Q0': 0,
            'Q1': 1,  # Diagonal con Q0
            'Q2': 2,
            'Q3': 3
        }
        assert self.family.validate_solution(solution, n=4) is False
    
    def test_validate_solution_incomplete(self):
        """Test que validate_solution() rechaza solución incompleta."""
        solution = {
            'Q0': 0,
            'Q1': 2
            # Faltan Q2 y Q3
        }
        assert self.family.validate_solution(solution, n=4) is False
    
    def test_get_metadata(self):
        """Test que get_metadata() retorna información correcta."""
        metadata = self.family.get_metadata(n=8)
        
        assert metadata['family'] == 'nqueens'
        assert metadata['n'] == 8
        assert metadata['n_variables'] == 8
        assert metadata['n_constraints'] == 28  # 8*7/2
        assert metadata['domain_size'] == 8
        assert metadata['complexity'] == 'O(n^2)'
        assert metadata['problem_type'] == 'combinatorial'
    
    def test_get_metadata_difficulty_estimation(self):
        """Test que la estimación de dificultad es correcta."""
        assert self.family.get_metadata(n=4)['difficulty'] == 'easy'
        assert self.family.get_metadata(n=8)['difficulty'] == 'easy'
        assert self.family.get_metadata(n=12)['difficulty'] == 'medium'
        assert self.family.get_metadata(n=30)['difficulty'] == 'hard'
        assert self.family.get_metadata(n=100)['difficulty'] == 'very_hard'
    
    def test_known_solution_8queens(self):
        """Test con solución conocida de 8-Queens."""
        solution = {
            'Q0': 0,
            'Q1': 4,
            'Q2': 7,
            'Q3': 5,
            'Q4': 2,
            'Q5': 6,
            'Q6': 1,
            'Q7': 3
        }
        assert self.family.validate_solution(solution, n=8) is True


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
