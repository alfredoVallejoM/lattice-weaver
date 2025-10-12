"""
Tests unitarios para Knapsack Problem.
"""

import pytest
from lattice_weaver.problems import get_catalog
from lattice_weaver.problems.generators.knapsack import KnapsackProblem


class TestKnapsackProblem:
    """Tests para la familia Knapsack."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.catalog = get_catalog()
        self.family = KnapsackProblem()
    
    def test_initialization(self):
        """Test de inicialización de la familia."""
        assert self.family.name == 'knapsack'
        assert 'knapsack' in self.family.description.lower()
    
    def test_default_params(self):
        """Test de parámetros por defecto."""
        defaults = self.family.get_default_params()
        assert 'n_items' in defaults
        assert 'max_weight' in defaults
        assert 'max_value' in defaults
        assert defaults['n_items'] == 10
    
    def test_param_schema(self):
        """Test del esquema de parámetros."""
        schema = self.family.get_param_schema()
        assert 'n_items' in schema
        assert 'capacity' in schema
        assert 'max_weight' in schema
        assert 'max_value' in schema
        assert schema['n_items']['type'] == int
    
    def test_generate_random_instance(self):
        """Test de generación de instancia aleatoria."""
        engine = self.catalog.generate_problem('knapsack', n_items=10, seed=42)
        
        # Verificar variables (10 items)
        assert len(engine.variables) == 10
        
        # Verificar nombres de variables
        for i in range(10):
            var_name = f'item_{i}'
            assert var_name in engine.variables
            # Dominio binario
            assert list(engine.variables[var_name]) == [0, 1]
    
    def test_generate_with_custom_items(self):
        """Test de generación con items personalizados."""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        engine = self.catalog.generate_problem(
            'knapsack',
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        
        assert len(engine.variables) == 3
    
    def test_generate_reproducibility(self):
        """Test de reproducibilidad con semilla."""
        engine1 = self.catalog.generate_problem('knapsack', n_items=10, seed=42)
        engine2 = self.catalog.generate_problem('knapsack', n_items=10, seed=42)
        
        # Verificar que los dominios son iguales
        for var_name in engine1.variables:
            domain1 = list(engine1.variables[var_name])
            domain2 = list(engine2.variables[var_name])
            assert domain1 == domain2
    
    def test_validate_valid_solution(self):
        """Test de validación de solución válida."""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        # Solución válida: seleccionar items 0 y 1 (peso = 30)
        solution = {
            'item_0': 1,
            'item_1': 1,
            'item_2': 0
        }
        
        is_valid = self.catalog.validate_solution(
            'knapsack',
            solution,
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        assert is_valid
    
    def test_validate_capacity_violation(self):
        """Test de validación con violación de capacidad."""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        # Solución inválida: seleccionar todos los items (peso = 60 > 50)
        solution = {
            'item_0': 1,
            'item_1': 1,
            'item_2': 1
        }
        
        is_valid = self.catalog.validate_solution(
            'knapsack',
            solution,
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        assert not is_valid
    
    def test_validate_invalid_decision(self):
        """Test de validación con decisión inválida."""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        # Decisión inválida: valor 2 (debe ser 0 o 1)
        solution = {
            'item_0': 1,
            'item_1': 2,  # Inválido
            'item_2': 0
        }
        
        is_valid = self.catalog.validate_solution(
            'knapsack',
            solution,
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        assert not is_valid
    
    def test_validate_incomplete_solution(self):
        """Test de validación de solución incompleta."""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        # Falta item_2
        solution = {
            'item_0': 1,
            'item_1': 1
        }
        
        is_valid = self.catalog.validate_solution(
            'knapsack',
            solution,
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        assert not is_valid
    
    def test_metadata_custom_instance(self):
        """Test de metadatos para instancia personalizada."""
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 50
        
        metadata = self.catalog.get_metadata(
            'knapsack',
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        
        assert metadata['family'] == 'knapsack'
        assert metadata['n_items'] == 3
        assert metadata['n_variables'] == 3
        assert metadata['domain_size'] == 2
        assert metadata['capacity'] == 50
        assert metadata['total_weight'] == 60
        assert metadata['total_value'] == 280
        assert metadata['complexity'] == 'NP-complete'
    
    def test_metadata_random_instance(self):
        """Test de metadatos para instancia aleatoria."""
        metadata = self.catalog.get_metadata(
            'knapsack',
            n_items=10,
            max_weight=20,
            max_value=100,
            seed=42
        )
        
        assert metadata['n_items'] == 10
        assert 'capacity' in metadata
        assert 'total_weight' in metadata
        assert 'total_value' in metadata
        assert 'difficulty' in metadata
    
    def test_difficulty_calculation(self):
        """Test de cálculo de dificultad."""
        # Capacidad muy baja (< 30% del peso total) = hard
        weights = [10, 20, 30]
        values = [60, 100, 120]
        capacity = 15  # 15/60 = 0.25 < 0.3
        
        metadata = self.catalog.get_metadata(
            'knapsack',
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        assert metadata['difficulty'] == 'hard'
        
        # Capacidad alta (>= 50% del peso total) = easy
        capacity = 35  # 35/60 = 0.58 >= 0.5
        metadata = self.catalog.get_metadata(
            'knapsack',
            n_items=3,
            weights=weights,
            values=values,
            capacity=capacity
        )
        assert metadata['difficulty'] == 'easy'
    
    def test_invalid_n_items(self):
        """Test de número de items inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('knapsack', n_items=1)
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem('knapsack', n_items=101)
    
    def test_invalid_weights_values_length(self):
        """Test de longitud incorrecta de weights/values."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem(
                'knapsack',
                n_items=3,
                weights=[10, 20],  # Longitud incorrecta
                values=[60, 100, 120]
            )
    
    def test_catalog_registration(self):
        """Test de registro en catálogo."""
        assert self.catalog.has('knapsack')
        family = self.catalog.get('knapsack')
        assert isinstance(family, KnapsackProblem)
    
    def test_auto_capacity_calculation(self):
        """Test de cálculo automático de capacidad."""
        weights = [10, 20, 30, 40]
        values = [60, 100, 120, 150]
        
        # No especificar capacidad
        metadata = self.catalog.get_metadata(
            'knapsack',
            n_items=4,
            weights=weights,
            values=values
        )
        
        # Capacidad debe ser 50% del peso total
        expected_capacity = sum(weights) // 2
        assert metadata['capacity'] == expected_capacity

