"""
Tests unitarios para Magic Square Problem.
"""

import pytest
from lattice_weaver.problems import get_catalog
from lattice_weaver.problems.generators.magic_square import MagicSquareProblem


class TestMagicSquareProblem:
    """Tests para la familia Magic Square."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.catalog = get_catalog()
        self.family = MagicSquareProblem()
    
    def test_initialization(self):
        """Test de inicialización de la familia."""
        assert self.family.name == 'magic_square'
        assert 'magic' in self.family.description.lower()
    
    def test_default_params(self):
        """Test de parámetros por defecto."""
        defaults = self.family.get_default_params()
        assert 'size' in defaults
        assert 'difficulty' in defaults
        assert defaults['size'] == 3
        assert defaults['difficulty'] == 'medium'
    
    def test_param_schema(self):
        """Test del esquema de parámetros."""
        schema = self.family.get_param_schema()
        assert 'size' in schema
        assert 'difficulty' in schema
        assert 'n_clues' in schema
        assert schema['size']['type'] == int
        assert schema['difficulty']['type'] == str
    
    def test_calculate_magic_sum(self):
        """Test del cálculo de la suma mágica."""
        # Para n=3: M = 3(9+1)/2 = 15
        assert self.family._calculate_magic_sum(3) == 15
        
        # Para n=4: M = 4(16+1)/2 = 34
        assert self.family._calculate_magic_sum(4) == 34
        
        # Para n=5: M = 5(25+1)/2 = 65
        assert self.family._calculate_magic_sum(5) == 65
    
    def test_generate_size_3(self):
        """Test de generación de cuadrado mágico 3×3."""
        engine = self.catalog.generate_problem('magic_square', size=3, difficulty='empty')
        
        # Verificar variables (9 celdas)
        assert len(engine.variables) == 9
        
        # Verificar nombres de variables
        for row in range(3):
            for col in range(3):
                var_name = f'M_{row}_{col}'
                assert var_name in engine.variables
    
    def test_generate_size_4(self):
        """Test de generación de cuadrado mágico 4×4."""
        engine = self.catalog.generate_problem('magic_square', size=4, difficulty='empty')
        
        # Verificar variables (16 celdas)
        assert len(engine.variables) == 16
    
    def test_generate_with_clues(self):
        """Test de generación con pistas."""
        engine = self.catalog.generate_problem('magic_square', size=3, n_clues=5, seed=42)
        
        # Verificar que hay exactamente 5 celdas con dominios de un solo valor
        clue_count = sum(1 for domain in engine.variables.values() if len(domain) == 1)
        assert clue_count == 5
    
    def test_difficulty_levels(self):
        """Test de niveles de dificultad."""
        size = 3
        
        difficulties = {
            'empty': 0.0,
            'easy': 0.5,
            'medium': 0.3,
            'hard': 0.2,
            'expert': 0.1
        }
        
        for difficulty, expected_ratio in difficulties.items():
            metadata = self.catalog.get_metadata('magic_square', size=size, difficulty=difficulty)
            actual_ratio = metadata['fill_ratio']
            
            # Permitir pequeña tolerancia por redondeo
            assert abs(actual_ratio - expected_ratio) < 0.15, \
                f"Dificultad {difficulty}: esperado {expected_ratio}, obtenido {actual_ratio}"
    
    def test_validate_valid_solution_3x3(self):
        """Test de validación de solución válida 3×3 (Lo Shu)."""
        # Cuadrado mágico 3×3 clásico
        solution = {
            'M_0_0': 2, 'M_0_1': 7, 'M_0_2': 6,
            'M_1_0': 9, 'M_1_1': 5, 'M_1_2': 1,
            'M_2_0': 4, 'M_2_1': 3, 'M_2_2': 8,
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=3)
        assert is_valid
    
    def test_validate_valid_solution_4x4(self):
        """Test de validación de solución válida 4×4 (Durero)."""
        # Cuadrado mágico 4×4 de Durero
        solution = {
            'M_0_0': 16, 'M_0_1': 3, 'M_0_2': 2, 'M_0_3': 13,
            'M_1_0': 5, 'M_1_1': 10, 'M_1_2': 11, 'M_1_3': 8,
            'M_2_0': 9, 'M_2_1': 6, 'M_2_2': 7, 'M_2_3': 12,
            'M_3_0': 4, 'M_3_1': 15, 'M_3_2': 14, 'M_3_3': 1,
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=4)
        assert is_valid
    
    def test_validate_invalid_row_sum(self):
        """Test de validación con suma de fila incorrecta."""
        # Fila 0 suma 16 en lugar de 15
        solution = {
            'M_0_0': 2, 'M_0_1': 7, 'M_0_2': 7,  # Suma = 16
            'M_1_0': 9, 'M_1_1': 5, 'M_1_2': 1,
            'M_2_0': 4, 'M_2_1': 3, 'M_2_2': 8,
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=3)
        assert not is_valid
    
    def test_validate_invalid_column_sum(self):
        """Test de validación con suma de columna incorrecta."""
        # Columna 0 suma incorrecta
        solution = {
            'M_0_0': 1, 'M_0_1': 7, 'M_0_2': 6,
            'M_1_0': 9, 'M_1_1': 5, 'M_1_2': 1,
            'M_2_0': 4, 'M_2_1': 3, 'M_2_2': 8,
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=3)
        assert not is_valid
    
    def test_validate_invalid_diagonal_sum(self):
        """Test de validación con suma de diagonal incorrecta."""
        # Diagonal principal suma incorrecta
        solution = {
            'M_0_0': 1, 'M_0_1': 7, 'M_0_2': 7,
            'M_1_0': 9, 'M_1_1': 5, 'M_1_2': 1,
            'M_2_0': 5, 'M_2_1': 3, 'M_2_2': 7,
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=3)
        assert not is_valid
    
    def test_validate_duplicate_values(self):
        """Test de validación con valores duplicados."""
        # Dos celdas con valor 5
        solution = {
            'M_0_0': 2, 'M_0_1': 7, 'M_0_2': 6,
            'M_1_0': 9, 'M_1_1': 5, 'M_1_2': 1,
            'M_2_0': 4, 'M_2_1': 3, 'M_2_2': 5,  # Duplicado
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=3)
        assert not is_valid
    
    def test_validate_incomplete_solution(self):
        """Test de validación de solución incompleta."""
        # Falta una celda
        solution = {
            'M_0_0': 2, 'M_0_1': 7, 'M_0_2': 6,
            'M_1_0': 9, 'M_1_1': 5, 'M_1_2': 1,
            'M_2_0': 4, 'M_2_1': 3,  # Falta M_2_2
        }
        
        is_valid = self.catalog.validate_solution('magic_square', solution, size=3)
        assert not is_valid
    
    def test_metadata_size_3(self):
        """Test de metadatos para tamaño 3."""
        metadata = self.catalog.get_metadata('magic_square', size=3, difficulty='medium')
        
        assert metadata['family'] == 'magic_square'
        assert metadata['size'] == 3
        assert metadata['n_variables'] == 9
        assert metadata['domain_size'] == 9
        assert metadata['magic_sum'] == 15
        assert metadata['complexity'] == 'NP-complete'
    
    def test_metadata_size_4(self):
        """Test de metadatos para tamaño 4."""
        metadata = self.catalog.get_metadata('magic_square', size=4, difficulty='easy')
        
        assert metadata['size'] == 4
        assert metadata['n_variables'] == 16
        assert metadata['magic_sum'] == 34
    
    def test_invalid_size(self):
        """Test de tamaño inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('magic_square', size=2)
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem('magic_square', size=11)
    
    def test_invalid_difficulty(self):
        """Test de dificultad inválida."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('magic_square', size=3, difficulty='invalid')
    
    def test_catalog_registration(self):
        """Test de registro en catálogo."""
        assert self.catalog.has('magic_square')
        family = self.catalog.get('magic_square')
        assert isinstance(family, MagicSquareProblem)
    
    def test_reproducibility_with_seed(self):
        """Test de reproducibilidad con semilla."""
        engine1 = self.catalog.generate_problem('magic_square', size=3, n_clues=5, seed=42)
        engine2 = self.catalog.generate_problem('magic_square', size=3, n_clues=5, seed=42)
        
        # Verificar que las pistas son las mismas
        for var_name in engine1.variables:
            domain1 = list(engine1.variables[var_name])
            domain2 = list(engine2.variables[var_name])
            assert domain1 == domain2

