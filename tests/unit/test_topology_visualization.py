import pytest
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para pruebas
from lattice_weaver.topology.visualization import TopologyVisualizer
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube
import os


class TestTopologyVisualizer:

    @pytest.fixture
    def simple_cubical_complex(self):
        graph = nx.Graph()
        v0 = GeometricCube(dimensions=0, coordinates=(0,))
        graph.add_node(v0)
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    @pytest.fixture
    def square_complex(self):
        graph = nx.Graph()
        v0 = GeometricCube(dimensions=0, coordinates=(0, 0))
        v1 = GeometricCube(dimensions=0, coordinates=(1, 0))
        v2 = GeometricCube(dimensions=0, coordinates=(0, 1))
        v3 = GeometricCube(dimensions=0, coordinates=(1, 1))
        graph.add_edges_from([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    @pytest.fixture
    def sample_homology(self):
        return {'beta_0': 1, 'beta_1': 1, 'beta_2': 0}

    def test_visualize_cubical_complex(self, simple_cubical_complex, tmp_path):
        visualizer = TopologyVisualizer()
        save_path = tmp_path / "cubical_complex.png"
        fig = visualizer.visualize_cubical_complex(
            simple_cubical_complex,
            title="Test Complejo Cúbico",
            save_path=str(save_path)
        )
        assert fig is not None
        assert os.path.exists(save_path)
        visualizer.close()

    def test_visualize_homology(self, sample_homology, tmp_path):
        visualizer = TopologyVisualizer()
        save_path = tmp_path / "homology.png"
        fig = visualizer.visualize_homology(
            sample_homology,
            title="Test Números de Betti",
            save_path=str(save_path)
        )
        assert fig is not None
        assert os.path.exists(save_path)
        visualizer.close()

    def test_visualize_cubical_complex_invalid_input(self):
        visualizer = TopologyVisualizer()
        with pytest.raises(ValueError):
            visualizer.visualize_cubical_complex(object())

    def test_visualize_homology_invalid_input(self):
        visualizer = TopologyVisualizer()
        with pytest.raises(ValueError):
            visualizer.visualize_homology("invalid")

    def test_visualize_square_complex(self, square_complex, tmp_path):
        visualizer = TopologyVisualizer()
        save_path = tmp_path / "square_complex.png"
        fig = visualizer.visualize_cubical_complex(
            square_complex,
            title="Test Complejo Cuadrado",
            save_path=str(save_path)
        )
        assert fig is not None
        assert os.path.exists(save_path)
        visualizer.close()

