"""
Tests unitarios para Latin Square Problem.
"""

import pytest
from lattice_weaver.problems import get_catalog
from lattice_weaver.problems.generators.latin_square import LatinSquareProblem


class TestLatinSquareProblem:
    """Tests para la familia Latin Square."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.catalog = get_catalog()
        self.family = LatinSquareProblem()
    
    def test_initialization(self):
        """Test de inicialización de la familia."""
        assert self.family.name == 'latin_square'
        assert 'latin' in self.family.description.lower()
    
    def test_default_params(self):
        """Test de parámetros por defecto."""
        defaults = self.family.get_default_params()
        assert 'size' in defaults
        assert 'difficulty' in defaults
        assert defaults['size'] == 5
        assert defaults['difficulty'] == 'medium'
    
    def test_param_schema(self):
        """Test del esquema de parámetros."""
        schema = self.family.get_param_schema()
        assert 'size' in schema
        assert 'difficulty' in schema
        assert 'n_clues' in schema
        assert schema['size']['type'] == int
        assert schema['difficulty']['type'] == str
    
    def test_generate_size_5(self):
        """Test de generación de cuadrado latino 5×5."""
        engine = self.catalog.generate_problem('latin_square', size=5, difficulty='medium', seed=42)
        
        # Verificar variables (25 celdas)
        assert len(engine.variables) == 25
        
        # Verificar nombres de variables
        for row in range(5):
            for col in range(5):
                var_name = f'C_{row}_{col}'
                assert var_name in engine.variables
    
    def test_generate_different_sizes(self):
        """Test de generación con diferentes tamaños."""
        for size in [3, 5, 7, 10]:
            engine = self.catalog.generate_problem('latin_square', size=size, difficulty='empty')
            assert len(engine.variables) == size * size
    
    def test_difficulty_levels(self):
        """Test de niveles de dificultad."""
        size = 5
        
        difficulties = {
            'empty': 0.0,
            'easy': 0.6,
            'medium': 0.4,
            'hard': 0.25,
            'expert': 0.15
        }
        
        for difficulty, expected_ratio in difficulties.items():
            metadata = self.catalog.get_metadata('latin_square', size=size, difficulty=difficulty)
            actual_ratio = metadata['fill_ratio']
            
            # Permitir pequeña tolerancia por redondeo
            assert abs(actual_ratio - expected_ratio) < 0.05, \
                f"Dificultad {difficulty}: esperado {expected_ratio}, obtenido {actual_ratio}"
    
    def test_clues_with_seed(self):
        """Test de reproducibilidad de pistas con semilla."""
        engine1 = self.catalog.generate_problem('latin_square', size=5, difficulty='medium', seed=42)
        engine2 = self.catalog.generate_problem('latin_square', size=5, difficulty='medium', seed=42)
        
        # Verificar que las pistas son las mismas (comparar contenido)
        for var_name in engine1.variables:
            domain1 = list(engine1.variables[var_name])
            domain2 = list(engine2.variables[var_name])
            assert domain1 == domain2
    
    def test_validate_valid_solution_3x3(self):
        """Test de validación de solución válida 3×3."""
        # Cuadrado latino 3×3 válido
        solution = {
            'C_0_0': 1, 'C_0_1': 2, 'C_0_2': 3,
            'C_1_0': 2, 'C_1_1': 3, 'C_1_2': 1,
            'C_2_0': 3, 'C_2_1': 1, 'C_2_2': 2,
        }
        
        is_valid = self.catalog.validate_solution('latin_square', solution, size=3)
        assert is_valid
    
    def test_validate_valid_solution_5x5(self):
        """Test de validación de solución válida 5×5."""
        # Cuadrado latino 5×5 válido (permutaciones cíclicas)
        solution = {}
        for row in range(5):
            for col in range(5):
                value = ((col - row) % 5) + 1
                solution[f'C_{row}_{col}'] = value
        
        is_valid = self.catalog.validate_solution('latin_square', solution, size=5)
        assert is_valid
    
    def test_validate_invalid_row(self):
        """Test de validación con fila inválida."""
        # Fila 0 tiene dos 1s
        solution = {
            'C_0_0': 1, 'C_0_1': 1, 'C_0_2': 3,  # Dos 1s en fila 0
            'C_1_0': 2, 'C_1_1': 3, 'C_1_2': 1,
            'C_2_0': 3, 'C_2_1': 2, 'C_2_2': 2,
        }
        
        is_valid = self.catalog.validate_solution('latin_square', solution, size=3)
        assert not is_valid
    
    def test_validate_invalid_column(self):
        """Test de validación con columna inválida."""
        # Columna 0 tiene dos 1s
        solution = {
            'C_0_0': 1, 'C_0_1': 2, 'C_0_2': 3,
            'C_1_0': 1, 'C_1_1': 3, 'C_1_2': 2,  # Dos 1s en columna 0
            'C_2_0': 3, 'C_2_1': 1, 'C_2_2': 2,
        }
        
        is_valid = self.catalog.validate_solution('latin_square', solution, size=3)
        assert not is_valid
    
    def test_validate_incomplete_solution(self):
        """Test de validación de solución incompleta."""
        # Falta una celda
        solution = {
            'C_0_0': 1, 'C_0_1': 2, 'C_0_2': 3,
            'C_1_0': 2, 'C_1_1': 3, 'C_1_2': 1,
            'C_2_0': 3, 'C_2_1': 1,  # Falta C_2_2
        }
        
        is_valid = self.catalog.validate_solution('latin_square', solution, size=3)
        assert not is_valid
    
    def test_metadata_size_5(self):
        """Test de metadatos para tamaño 5."""
        metadata = self.catalog.get_metadata('latin_square', size=5, difficulty='medium')
        
        assert metadata['family'] == 'latin_square'
        assert metadata['size'] == 5
        assert metadata['n_variables'] == 25
        assert metadata['domain_size'] == 5
        assert metadata['difficulty'] == 'medium'
        assert 'n_constraints' in metadata
        assert 'fill_ratio' in metadata
    
    def test_metadata_different_sizes(self):
        """Test de metadatos para diferentes tamaños."""
        for size in [3, 5, 10]:
            metadata = self.catalog.get_metadata('latin_square', size=size, difficulty='easy')
            
            assert metadata['size'] == size
            assert metadata['n_variables'] == size * size
            assert metadata['domain_size'] == size
            
            # Verificar número de restricciones
            # Restricciones de fila: size * (size * (size-1) / 2)
            # Restricciones de columna: size * (size * (size-1) / 2)
            expected_constraints = 2 * size * (size * (size - 1) // 2)
            assert metadata['n_constraints'] == expected_constraints
    
    def test_constraints_count(self):
        """Test de conteo de restricciones."""
        engine = self.catalog.generate_problem('latin_square', size=5, difficulty='empty')
        
        # 5×5: restricciones de fila + columna
        # Fila: 5 filas × (5 × 4 / 2) = 5 × 10 = 50
        # Columna: 5 columnas × (5 × 4 / 2) = 5 × 10 = 50
        # Total: 100
        expected_constraints = 100
        assert len(engine.constraints) == expected_constraints
    
    def test_row_constraints(self):
        """Test de restricciones de fila."""
        engine = self.catalog.generate_problem('latin_square', size=3, difficulty='empty')
        
        # Verificar que hay restricciones de fila
        row_constraints = [cid for cid in engine.constraints.keys() if 'row_' in cid]
        
        # 3 filas × (3 × 2 / 2) = 3 × 3 = 9
        assert len(row_constraints) == 9
    
    def test_column_constraints(self):
        """Test de restricciones de columna."""
        engine = self.catalog.generate_problem('latin_square', size=3, difficulty='empty')
        
        # Verificar que hay restricciones de columna
        col_constraints = [cid for cid in engine.constraints.keys() if 'col_' in cid]
        
        # 3 columnas × (3 × 2 / 2) = 3 × 3 = 9
        assert len(col_constraints) == 9
    
    def test_invalid_size(self):
        """Test de tamaño inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('latin_square', size=2)
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem('latin_square', size=26)
    
    def test_invalid_difficulty(self):
        """Test de dificultad inválida."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('latin_square', size=5, difficulty='invalid')
    
    def test_catalog_registration(self):
        """Test de registro en catálogo."""
        assert self.catalog.has('latin_square')
        family = self.catalog.get('latin_square')
        assert isinstance(family, LatinSquareProblem)
    
    def test_custom_n_clues(self):
        """Test de número personalizado de pistas."""
        metadata = self.catalog.get_metadata('latin_square', size=5, n_clues=10)
        
        assert metadata['n_clues'] == 10
        assert metadata['fill_ratio'] == 0.4  # 10/25

