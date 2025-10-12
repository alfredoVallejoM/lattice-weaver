"""
Tests unitarios para Logic Puzzle Problem.
"""

import pytest
from lattice_weaver.problems import get_catalog
from lattice_weaver.problems.generators.logic_puzzle import LogicPuzzleProblem


class TestLogicPuzzleProblem:
    """Tests para la familia Logic Puzzle."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.catalog = get_catalog()
        self.family = LogicPuzzleProblem()
    
    def test_initialization(self):
        """Test de inicialización de la familia."""
        assert self.family.name == 'logic_puzzle'
        assert 'logic' in self.family.description.lower()
    
    def test_default_params(self):
        """Test de parámetros por defecto."""
        defaults = self.family.get_default_params()
        assert 'puzzle_type' in defaults
        assert 'n_entities' in defaults
        assert 'n_attributes' in defaults
        assert defaults['puzzle_type'] == 'simple'
        assert defaults['n_entities'] == 5
    
    def test_param_schema(self):
        """Test del esquema de parámetros."""
        schema = self.family.get_param_schema()
        assert 'puzzle_type' in schema
        assert 'n_entities' in schema
        assert 'n_attributes' in schema
        assert schema['puzzle_type']['type'] == str
        assert schema['n_entities']['type'] == int
    
    def test_generate_simple_puzzle(self):
        """Test de generación de puzzle simple."""
        engine = self.catalog.generate_problem(
            'logic_puzzle',
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        
        # Verificar variables (3 entidades × 2 atributos = 6 variables)
        assert len(engine.variables) == 6
        
        # Verificar nombres de variables
        for attr_idx in range(2):
            for entity_id in range(3):
                var_name = f'color_{entity_id}' if attr_idx == 0 else f'shape_{entity_id}'
                assert var_name in engine.variables
    
    def test_generate_zebra_puzzle(self):
        """Test de generación de Zebra puzzle."""
        engine = self.catalog.generate_problem('logic_puzzle', puzzle_type='zebra')
        
        # Zebra puzzle: 5 entidades × 5 categorías = 25 variables
        assert len(engine.variables) == 25
        
        # Verificar categorías
        categories = ['color', 'nationality', 'drink', 'smoke', 'pet']
        for category in categories:
            for entity_id in range(5):
                var_name = f'{category}_{entity_id}'
                assert var_name in engine.variables
    
    def test_generate_custom_puzzle(self):
        """Test de generación de puzzle personalizado."""
        custom_attributes = {
            'color': ['red', 'blue', 'green'],
            'size': ['small', 'medium', 'large']
        }
        
        engine = self.catalog.generate_problem(
            'logic_puzzle',
            puzzle_type='custom',
            n_entities=3,
            attributes=custom_attributes
        )
        
        # 3 entidades × 2 categorías = 6 variables
        assert len(engine.variables) == 6
    
    def test_validate_valid_solution_simple(self):
        """Test de validación de solución válida (puzzle simple)."""
        # 3 entidades, 2 atributos
        solution = {
            'color_0': 0, 'color_1': 1, 'color_2': 2,
            'shape_0': 1, 'shape_1': 2, 'shape_2': 0
        }
        
        is_valid = self.catalog.validate_solution(
            'logic_puzzle',
            solution,
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        assert is_valid
    
    def test_validate_invalid_duplicate_values(self):
        """Test de validación con valores duplicados en una categoría."""
        # color_0 y color_1 tienen el mismo valor
        solution = {
            'color_0': 0, 'color_1': 0, 'color_2': 2,  # Dos 0s
            'shape_0': 1, 'shape_1': 2, 'shape_2': 0
        }
        
        is_valid = self.catalog.validate_solution(
            'logic_puzzle',
            solution,
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        assert not is_valid
    
    def test_validate_invalid_out_of_range(self):
        """Test de validación con valor fuera de rango."""
        # color_0 = 3 (fuera de rango [0, 2])
        solution = {
            'color_0': 3, 'color_1': 1, 'color_2': 2,
            'shape_0': 1, 'shape_1': 2, 'shape_2': 0
        }
        
        is_valid = self.catalog.validate_solution(
            'logic_puzzle',
            solution,
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        assert not is_valid
    
    def test_validate_incomplete_solution(self):
        """Test de validación de solución incompleta."""
        # Falta shape_2
        solution = {
            'color_0': 0, 'color_1': 1, 'color_2': 2,
            'shape_0': 1, 'shape_1': 2
        }
        
        is_valid = self.catalog.validate_solution(
            'logic_puzzle',
            solution,
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        assert not is_valid
    
    def test_metadata_simple_puzzle(self):
        """Test de metadatos para puzzle simple."""
        metadata = self.catalog.get_metadata(
            'logic_puzzle',
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        
        assert metadata['family'] == 'logic_puzzle'
        assert metadata['puzzle_type'] == 'simple'
        assert metadata['n_entities'] == 3
        assert metadata['n_categories'] == 2
        assert metadata['n_variables'] == 6
        assert metadata['domain_size'] == 3
        assert metadata['complexity'] == 'NP-complete'
    
    def test_metadata_zebra_puzzle(self):
        """Test de metadatos para Zebra puzzle."""
        metadata = self.catalog.get_metadata('logic_puzzle', puzzle_type='zebra')
        
        assert metadata['puzzle_type'] == 'zebra'
        assert metadata['n_entities'] == 5
        assert metadata['n_categories'] == 5
        assert metadata['n_variables'] == 25
        assert metadata['difficulty'] == 'medium'
    
    def test_difficulty_calculation(self):
        """Test de cálculo de dificultad."""
        # 3 entidades = easy
        metadata = self.catalog.get_metadata(
            'logic_puzzle',
            puzzle_type='simple',
            n_entities=3,
            n_attributes=2
        )
        assert metadata['difficulty'] == 'easy'
        
        # 5 entidades = medium
        metadata = self.catalog.get_metadata(
            'logic_puzzle',
            puzzle_type='simple',
            n_entities=5,
            n_attributes=3
        )
        assert metadata['difficulty'] == 'medium'
        
        # 7 entidades = hard
        metadata = self.catalog.get_metadata(
            'logic_puzzle',
            puzzle_type='simple',
            n_entities=7,
            n_attributes=3
        )
        assert metadata['difficulty'] == 'hard'
    
    def test_invalid_n_entities(self):
        """Test de número de entidades inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'logic_puzzle',
                puzzle_type='simple',
                n_entities=2
            )
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'logic_puzzle',
                puzzle_type='simple',
                n_entities=11
            )
    
    def test_invalid_n_attributes(self):
        """Test de número de atributos inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'logic_puzzle',
                puzzle_type='simple',
                n_entities=5,
                n_attributes=1
            )
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'logic_puzzle',
                puzzle_type='simple',
                n_entities=5,
                n_attributes=7
            )
    
    def test_invalid_custom_attributes_length(self):
        """Test de atributos personalizados con longitud incorrecta."""
        custom_attributes = {
            'color': ['red', 'blue'],  # Solo 2 valores, necesita 3
            'size': ['small', 'medium', 'large']
        }
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'logic_puzzle',
                puzzle_type='custom',
                n_entities=3,
                attributes=custom_attributes
            )
    
    def test_catalog_registration(self):
        """Test de registro en catálogo."""
        assert self.catalog.has('logic_puzzle')
        family = self.catalog.get('logic_puzzle')
        assert isinstance(family, LogicPuzzleProblem)
    
    def test_zebra_attributes(self):
        """Test de atributos del Zebra puzzle."""
        attributes = self.family._get_zebra_attributes()
        
        assert 'color' in attributes
        assert 'nationality' in attributes
        assert 'drink' in attributes
        assert 'smoke' in attributes
        assert 'pet' in attributes
        
        # Cada categoría debe tener 5 valores
        for category, values in attributes.items():
            assert len(values) == 5

