"""
Tests unitarios para Map Coloring Problem.
"""

import pytest
from lattice_weaver.problems import get_catalog
from lattice_weaver.problems.generators.map_coloring import MapColoringProblem, PREDEFINED_MAPS


class TestMapColoringProblem:
    """Tests para la familia Map Coloring."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.catalog = get_catalog()
        self.family = MapColoringProblem()
    
    def test_initialization(self):
        """Test de inicialización de la familia."""
        assert self.family.name == 'map_coloring'
        assert 'map' in self.family.description.lower()
    
    def test_default_params(self):
        """Test de parámetros por defecto."""
        defaults = self.family.get_default_params()
        assert 'map_name' in defaults
        assert 'n_colors' in defaults
        assert defaults['n_colors'] == 4
    
    def test_param_schema(self):
        """Test del esquema de parámetros."""
        schema = self.family.get_param_schema()
        assert 'map_name' in schema
        assert 'n_colors' in schema
        assert schema['map_name']['type'] == str
        assert schema['n_colors']['type'] == int
    
    def test_generate_australia(self):
        """Test de generación de mapa de Australia."""
        engine = self.catalog.generate_problem('map_coloring', map_name='australia', n_colors=3)
        
        # Verificar variables (7 estados/territorios)
        assert len(engine.variables) == 7
        
        # Verificar dominios
        for var_name, domain in engine.variables.items():
            assert len(domain) == 3  # 3 colores
            # Convertir a lista para comparar
            assert list(domain) == [0, 1, 2]
    
    def test_generate_usa(self):
        """Test de generación de mapa de USA."""
        engine = self.catalog.generate_problem('map_coloring', map_name='usa', n_colors=4)
        
        # Verificar variables (48 estados contiguos)
        assert len(engine.variables) == 48
        
        # Verificar dominios
        for var_name, domain in engine.variables.items():
            assert len(domain) == 4  # 4 colores
    
    def test_generate_europe(self):
        """Test de generación de mapa de Europa."""
        engine = self.catalog.generate_problem('map_coloring', map_name='europe_simple', n_colors=4)
        
        # Verificar variables (25 países)
        assert len(engine.variables) == 25
    
    def test_generate_south_america(self):
        """Test de generación de mapa de Sudamérica."""
        engine = self.catalog.generate_problem('map_coloring', map_name='south_america', n_colors=4)
        
        # Verificar variables (13 países)
        assert len(engine.variables) == 13
    
    def test_generate_random_map(self):
        """Test de generación de mapa aleatorio."""
        engine = self.catalog.generate_problem(
            'map_coloring',
            map_name='random',
            n_regions=20,
            n_colors=4,
            seed=42
        )
        
        # Verificar variables
        assert len(engine.variables) == 20
        
        # Verificar reproducibilidad
        engine2 = self.catalog.generate_problem(
            'map_coloring',
            map_name='random',
            n_regions=20,
            n_colors=4,
            seed=42
        )
        assert len(engine2.variables) == 20
    
    def test_validate_australia_solution(self):
        """Test de validación de solución para Australia."""
        # Solución válida (3 colores)
        solution = {
            'WA': 0,
            'NT': 1,
            'SA': 2,
            'QLD': 0,
            'NSW': 1,
            'VIC': 0,
            'TAS': 0  # Tasmania es isla, puede ser cualquier color
        }
        
        is_valid = self.catalog.validate_solution(
            'map_coloring',
            solution,
            map_name='australia',
            n_colors=3
        )
        assert is_valid
    
    def test_validate_australia_invalid_solution(self):
        """Test de validación de solución inválida para Australia."""
        # Solución inválida (WA y NT adyacentes con mismo color)
        solution = {
            'WA': 0,
            'NT': 0,  # Mismo color que WA (adyacente)
            'SA': 1,
            'QLD': 2,
            'NSW': 0,
            'VIC': 2,
            'TAS': 1
        }
        
        is_valid = self.catalog.validate_solution(
            'map_coloring',
            solution,
            map_name='australia',
            n_colors=3
        )
        assert not is_valid
    
    def test_metadata_australia(self):
        """Test de metadatos para Australia."""
        metadata = self.catalog.get_metadata('map_coloring', map_name='australia', n_colors=3)
        
        assert metadata['family'] == 'map_coloring'
        assert metadata['map_name'] == 'australia'
        assert metadata['n_regions'] == 7
        assert metadata['n_colors'] == 3
        assert metadata['is_planar'] is True
        assert metadata['chromatic_number_upper_bound'] == 4  # Four Color Theorem
        assert 'difficulty' in metadata
    
    def test_metadata_usa(self):
        """Test de metadatos para USA."""
        metadata = self.catalog.get_metadata('map_coloring', map_name='usa', n_colors=4)
        
        assert metadata['n_regions'] == 48
        assert metadata['n_colors'] == 4
        assert metadata['difficulty'] == 'hard'  # 48 regiones es difícil
    
    def test_four_color_theorem(self):
        """Test del Four Color Theorem (cota superior = 4)."""
        # Todos los mapas planares deberían ser coloreables con 4 colores
        for map_name in PREDEFINED_MAPS.keys():
            metadata = self.catalog.get_metadata('map_coloring', map_name=map_name, n_colors=4)
            assert metadata['chromatic_number_upper_bound'] == 4
            assert metadata['is_planar'] is True
    
    def test_invalid_map_name(self):
        """Test de nombre de mapa inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('map_coloring', map_name='invalid_map', n_colors=4)
    
    def test_invalid_n_colors(self):
        """Test de número de colores inválido."""
        with pytest.raises(ValueError):
            self.catalog.generate_problem('map_coloring', map_name='australia', n_colors=1)
        
        with pytest.raises(ValueError):
            self.catalog.generate_problem('map_coloring', map_name='australia', n_colors=11)
    
    def test_catalog_registration(self):
        """Test de registro en catálogo."""
        assert self.catalog.has('map_coloring')
        family = self.catalog.get('map_coloring')
        assert isinstance(family, MapColoringProblem)
    
    def test_adjacency_symmetry(self):
        """Test de simetría en adyacencias de mapas predefinidos."""
        # Verificar que si A es adyacente a B, entonces B es adyacente a A
        for map_name, adjacency in PREDEFINED_MAPS.items():
            for region, neighbors in adjacency.items():
                for neighbor in neighbors:
                    assert region in adjacency[neighbor], \
                        f"Asimetría en {map_name}: {region} -> {neighbor} pero no {neighbor} -> {region}"

