"""
Tests unitarios para SudokuProblem.
"""

import pytest
from lattice_weaver.problems.generators.sudoku import SudokuProblem
from lattice_weaver.arc_engine import ArcEngine


class TestSudokuProblem:
    """Tests para SudokuProblem."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.family = SudokuProblem()
    
    def test_initialization(self):
        """Test que la inicialización es correcta."""
        assert self.family.name == 'sudoku'
        assert 'Sudoku' in self.family.description
    
    def test_get_default_params(self):
        """Test que los parámetros por defecto son correctos."""
        defaults = self.family.get_default_params()
        assert defaults['size'] == 9
        assert defaults['difficulty'] == 'medium'
    
    def test_get_param_schema(self):
        """Test que el esquema de parámetros es correcto."""
        schema = self.family.get_param_schema()
        
        assert 'size' in schema
        assert 'difficulty' in schema
        assert 'n_clues' in schema
        assert 'seed' in schema
        
        assert schema['size']['choices'] == [4, 9, 16, 25]
        assert 'easy' in schema['difficulty']['choices']
    
    def test_generate_creates_arc_engine(self):
        """Test que generate() crea un ArcEngine."""
        engine = self.family.generate(size=4, difficulty='easy')
        assert isinstance(engine, ArcEngine)
    
    def test_generate_4x4_correct_variables(self):
        """Test generación de Sudoku 4x4."""
        engine = self.family.generate(size=4, difficulty='empty')
        
        # 4x4 = 16 celdas
        assert len(engine.variables) == 16
        
        # Verificar nombres de variables
        for row in range(4):
            for col in range(4):
                var_name = f'C_{row}_{col}'
                assert var_name in engine.variables
    
    def test_generate_9x9_correct_variables(self):
        """Test generación de Sudoku 9x9."""
        engine = self.family.generate(size=9, difficulty='empty')
        
        # 9x9 = 81 celdas
        assert len(engine.variables) == 81
    
    def test_generate_correct_domains(self):
        """Test que los dominios son correctos."""
        size = 4
        engine = self.family.generate(size=size, difficulty='empty')
        
        for row in range(size):
            for col in range(size):
                var_name = f'C_{row}_{col}'
                domain = list(engine.variables[var_name].get_values())
                assert domain == list(range(1, size + 1))
    
    def test_generate_with_clues(self):
        """Test que las pistas reducen dominios."""
        engine = self.family.generate(size=4, n_clues=5, seed=42)
        
        # Contar celdas con dominio reducido (pistas)
        clues_count = 0
        for var_name, domain in engine.variables.items():
            if len(list(domain.get_values())) == 1:
                clues_count += 1
        
        assert clues_count == 5
    
    def test_generate_difficulty_easy(self):
        """Test que dificultad 'easy' genera más pistas."""
        engine = self.family.generate(size=9, difficulty='easy', seed=42)
        
        clues_count = sum(
            1 for domain in engine.variables.values()
            if len(list(domain.get_values())) == 1
        )
        
        # Easy debería tener ~60% de celdas llenas (48-49 pistas)
        assert clues_count >= 40
    
    def test_generate_difficulty_hard(self):
        """Test que dificultad 'hard' genera menos pistas."""
        engine = self.family.generate(size=9, difficulty='hard', seed=42)
        
        clues_count = sum(
            1 for domain in engine.variables.values()
            if len(list(domain.get_values())) == 1
        )
        
        # Hard debería tener ~30% de celdas llenas (24-25 pistas)
        assert clues_count <= 30
    
    def test_generate_difficulty_empty(self):
        """Test que dificultad 'empty' no genera pistas."""
        engine = self.family.generate(size=4, difficulty='empty')
        
        for var_name, domain in engine.variables.items():
            assert len(list(domain.get_values())) == 4
    
    def test_generate_with_seed_reproducible(self):
        """Test que la semilla produce resultados reproducibles."""
        engine1 = self.family.generate(size=4, n_clues=5, seed=42)
        engine2 = self.family.generate(size=4, n_clues=5, seed=42)
        
        # Verificar que las mismas celdas tienen pistas
        for var_name in engine1.variables:
            domain1 = list(engine1.variables[var_name].get_values())
            domain2 = list(engine2.variables[var_name].get_values())
            assert domain1 == domain2
    
    def test_validate_solution_4x4_correct(self):
        """Test validación de solución correcta 4x4."""
        solution = {
            'C_0_0': 1, 'C_0_1': 2, 'C_0_2': 3, 'C_0_3': 4,
            'C_1_0': 3, 'C_1_1': 4, 'C_1_2': 1, 'C_1_3': 2,
            'C_2_0': 2, 'C_2_1': 1, 'C_2_2': 4, 'C_2_3': 3,
            'C_3_0': 4, 'C_3_1': 3, 'C_3_2': 2, 'C_3_3': 1
        }
        
        assert self.family.validate_solution(solution, size=4) is True
    
    def test_validate_solution_row_violation(self):
        """Test que detecta violación de fila."""
        solution = {
            'C_0_0': 1, 'C_0_1': 1, 'C_0_2': 3, 'C_0_3': 4,  # Dos 1s en fila 0
            'C_1_0': 3, 'C_1_1': 4, 'C_1_2': 1, 'C_1_3': 2,
            'C_2_0': 2, 'C_2_1': 1, 'C_2_2': 4, 'C_2_3': 3,
            'C_3_0': 4, 'C_3_1': 3, 'C_3_2': 2, 'C_3_3': 1
        }
        
        assert self.family.validate_solution(solution, size=4) is False
    
    def test_validate_solution_column_violation(self):
        """Test que detecta violación de columna."""
        solution = {
            'C_0_0': 1, 'C_0_1': 2, 'C_0_2': 3, 'C_0_3': 4,
            'C_1_0': 1, 'C_1_1': 4, 'C_1_2': 1, 'C_1_3': 2,  # Dos 1s en columna 0
            'C_2_0': 2, 'C_2_1': 1, 'C_2_2': 4, 'C_2_3': 3,
            'C_3_0': 4, 'C_3_1': 3, 'C_3_2': 2, 'C_3_3': 1
        }
        
        assert self.family.validate_solution(solution, size=4) is False
    
    def test_validate_solution_block_violation(self):
        """Test que detecta violación de bloque."""
        solution = {
            'C_0_0': 1, 'C_0_1': 1, 'C_0_2': 3, 'C_0_3': 4,  # Dos 1s en bloque superior izquierdo
            'C_1_0': 3, 'C_1_1': 4, 'C_1_2': 1, 'C_1_3': 2,
            'C_2_0': 2, 'C_2_1': 3, 'C_2_2': 4, 'C_2_3': 1,
            'C_3_0': 4, 'C_3_1': 2, 'C_3_2': 2, 'C_3_3': 3
        }
        
        assert self.family.validate_solution(solution, size=4) is False
    
    def test_validate_solution_incomplete(self):
        """Test que rechaza solución incompleta."""
        solution = {
            'C_0_0': 1, 'C_0_1': 2
            # Faltan muchas celdas
        }
        
        assert self.family.validate_solution(solution, size=4) is False
    
    def test_validate_solution_out_of_range(self):
        """Test que rechaza valores fuera de rango."""
        solution = {
            'C_0_0': 5, 'C_0_1': 2, 'C_0_2': 3, 'C_0_3': 4,  # 5 fuera de rango [1,4]
            'C_1_0': 3, 'C_1_1': 4, 'C_1_2': 1, 'C_1_3': 2,
            'C_2_0': 2, 'C_2_1': 1, 'C_2_2': 4, 'C_2_3': 3,
            'C_3_0': 4, 'C_3_1': 3, 'C_3_2': 2, 'C_3_3': 1
        }
        
        assert self.family.validate_solution(solution, size=4) is False
    
    def test_get_metadata_4x4(self):
        """Test metadatos para Sudoku 4x4."""
        metadata = self.family.get_metadata(size=4, difficulty='medium')
        
        assert metadata['family'] == 'sudoku'
        assert metadata['size'] == 4
        assert metadata['block_size'] == 2
        assert metadata['n_variables'] == 16
        assert metadata['domain_size'] == 4
        assert metadata['difficulty'] == 'medium'
    
    def test_get_metadata_9x9(self):
        """Test metadatos para Sudoku 9x9."""
        metadata = self.family.get_metadata(size=9, difficulty='hard')
        
        assert metadata['size'] == 9
        assert metadata['block_size'] == 3
        assert metadata['n_variables'] == 81
        assert metadata['domain_size'] == 9
    
    def test_invalid_size(self):
        """Test que tamaño inválido lanza error."""
        with pytest.raises(ValueError, match="debe ser uno de"):
            self.family.generate(size=10)  # 10 no es válido
    
    def test_invalid_difficulty(self):
        """Test que dificultad inválida lanza error."""
        with pytest.raises(ValueError, match="debe ser uno de"):
            self.family.generate(size=9, difficulty='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

