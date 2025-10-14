import pytest
from lattice_weaver.problems.generators.graph_coloring import GraphColoringProblem
from lattice_weaver.core.csp_problem import CSP, Constraint


class TestGraphColoringProblem:
    """Tests para GraphColoringProblem."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.family = GraphColoringProblem()
    
    def test_initialization(self):
        """Test que la inicialización es correcta."""
        assert self.family.name == 'graph_coloring'
        assert 'coloración de grafos' in self.family.description
    
    def test_get_default_params(self):
        """Test que los parámetros por defecto son correctos."""
        defaults = self.family.get_default_params()
        assert defaults['graph_type'] == 'random'
        assert defaults['n_nodes'] == 10
        assert defaults['n_colors'] == 3
        assert defaults['edge_probability'] == 0.3
    
    def test_get_param_schema(self):
        """Test que el esquema de parámetros es correcto."""
        schema = self.family.get_param_schema()
        
        assert 'graph_type' in schema
        assert 'n_nodes' in schema
        assert 'n_colors' in schema
        assert 'edge_probability' in schema
        
        assert schema['n_nodes']['required'] is True
        assert schema['n_colors']['required'] is True
    
    def test_generate_creates_csp(self):
        """Test que generate() crea un CSP."""
        csp = self.family.generate(n_nodes=5, n_colors=3, graph_type='cycle')
        assert isinstance(csp, CSP)

    
    def test_generate_cycle_graph(self):
        """Test generación de grafo cíclico."""
        n = 5
        csp = self.family.generate(graph_type='cycle', n_nodes=n, n_colors=3)
        
        assert len(csp.variables) == n
        # Ciclo tiene n aristas
        assert len(csp.constraints) == n
    
    def test_generate_complete_graph(self):
        """Test generación de grafo completo."""
        n = 4
        csp = self.family.generate(graph_type='complete', n_nodes=n, n_colors=4)
        
        assert len(csp.variables) == n
        # Grafo completo tiene n*(n-1)/2 aristas
        expected_edges = n * (n - 1) // 2
        assert len(csp.constraints) == expected_edges
    
    def test_generate_path_graph(self):
        """Test generación de grafo camino."""
        n = 5
        csp = self.family.generate(graph_type='path', n_nodes=n, n_colors=2)
        
        assert len(csp.variables) == n
        # Camino tiene n-1 aristas
        assert len(csp.constraints) == n - 1
    
    def test_generate_star_graph(self):
        """Test generación de grafo estrella."""
        n = 6
        csp = self.family.generate(graph_type='star', n_nodes=n, n_colors=2)
        
        assert len(csp.variables) == n
        # Estrella tiene n-1 aristas
        assert len(csp.constraints) == n - 1
    
    def test_generate_wheel_graph(self):
        """Test generación de grafo rueda."""
        n = 5
        csp = self.family.generate(graph_type='wheel', n_nodes=n, n_colors=3)
        
        assert len(csp.variables) == n
        # Rueda tiene 2*(n-1) aristas
        assert len(csp.constraints) == 2 * (n - 1)

    
    def test_generate_grid_graph(self):
        """Test generación de grafo grid."""
        csp = self.family.generate(
            graph_type='grid',
            n_nodes=9,
            grid_rows=3,
            grid_cols=3,
            n_colors=3
        )
        
        assert len(csp.variables) == 9
        # Grid 3x3 tiene 12 aristas (6 horizontales + 6 verticales)
        assert len(csp.constraints) == 12
    
    def test_generate_bipartite_graph(self):
        """Test generación de grafo bipartito."""
        csp = self.family.generate(
            graph_type='bipartite',
            n_nodes=6,
            n_colors=2,
            edge_probability=1.0,  # Bipartito completo
            seed=42
        )
        
        assert len(csp.variables) == 6
        # Bipartito completo 3+3 tiene 3*3=9 aristas
        assert len(csp.constraints) == 9
    
    def test_generate_random_graph_with_seed(self):
        """Test que la semilla produce resultados reproducibles."""
        csp1 = self.family.generate(
            graph_type='random',
            n_nodes=10,
            n_colors=3,
            edge_probability=0.3,
            seed=42
        )
        
        csp2 = self.family.generate(
            graph_type='random',
            n_nodes=10,
            n_colors=3,
            edge_probability=0.3,
            seed=42
        )
        
        # Mismo número de restricciones con misma semilla
        assert len(csp1.constraints) == len(csp2.constraints)
    
    def test_generate_correct_variable_names(self):
        """Test que las variables tienen nombres correctos."""
        csp = self.family.generate(graph_type='cycle', n_nodes=4, n_colors=3)
        expected_vars = {'V0', 'V1', 'V2', 'V3'}
        assert set(csp.variables) == expected_vars
    
    def test_generate_correct_domains(self):
        """Test que los dominios son correctos."""
        n_colors = 3
        csp = self.family.generate(graph_type='cycle', n_nodes=5, n_colors=n_colors)
        
        for var_name in csp.variables:
            domain = csp.domains[var_name]
            assert domain == frozenset(range(n_colors))
    
    def test_validate_solution_cycle_correct(self):
        """Test validación de solución correcta para ciclo."""
        # Ciclo de 5 nodos necesita 3 colores
        solution = {
            'V0': 0,
            'V1': 1,
            'V2': 0,
            'V3': 1,
            'V4': 2
        }
        assert self.family.validate_solution(
            solution,
            graph_type='cycle',
            n_nodes=5,
            n_colors=3
        ) is True
    
    def test_validate_solution_cycle_incorrect(self):
        """Test validación de solución incorrecta para ciclo."""
        # Nodos adyacentes con mismo color
        solution = {
            'V0': 0,
            'V1': 0,  # Mismo color que V0 (adyacente)
            'V2': 1,
            'V3': 0,
            'V4': 1
        }
        assert self.family.validate_solution(
            solution,
            graph_type='cycle',
            n_nodes=5,
            n_colors=3
        ) is False
    
    def test_validate_solution_path_correct(self):
        """Test validación de solución correcta para camino."""
        # Camino solo necesita 2 colores
        solution = {
            'V0': 0,
            'V1': 1,
            'V2': 0,
            'V3': 1
        }
        assert self.family.validate_solution(
            solution,
            graph_type='path',
            n_nodes=4,
            n_colors=2
        ) is True
    
    def test_get_metadata(self):
        """Test que get_metadata() retorna información correcta."""
        metadata = self.family.get_metadata(
            graph_type='cycle',
            n_nodes=5,
            n_colors=3
        )
        
        assert metadata['family'] == 'graph_coloring'
        assert metadata['graph_type'] == 'cycle'
        assert metadata['n_nodes'] == 5
        assert metadata['n_colors'] == 3
        assert metadata['n_variables'] == 5
        assert metadata['n_edges'] == 5
        assert metadata['domain_size'] == 3
        assert metadata['complexity'] == 'O(|E|)'
    
    def test_get_metadata_chromatic_lower_bound(self):
        """Test que se calcula la cota inferior del número cromático."""
        metadata = self.family.get_metadata(
            graph_type='complete',
            n_nodes=5,
            n_colors=5
        )
        
        # Grafo completo K5 necesita 5 colores
        assert metadata['chromatic_lower_bound'] >= 4
    
    def test_invalid_graph_type(self):
        """Test que tipo de grafo inválido lanza error."""
        with pytest.raises(ValueError, match="debe ser uno de"):
            self.family.generate(
                graph_type='invalid_type',
                n_nodes=5,
                n_colors=3
            )
    
    def test_invalid_n_nodes_too_small(self):
        """Test que n_nodes < 2 lanza error."""
        with pytest.raises(ValueError):
            self.family.generate(graph_type='cycle', n_nodes=1, n_colors=3)
    
    def test_invalid_n_colors_too_small(self):
        """Test que n_colors < 2 lanza error."""
        with pytest.raises(ValueError):
            self.family.generate(graph_type='cycle', n_nodes=5, n_colors=1)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
