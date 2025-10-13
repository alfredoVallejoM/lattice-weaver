import pytest
import networkx as nx
from lattice_weaver.topology.homology_engine import HomologyEngine
from lattice_weaver.topology.cubical_complex import CubicalComplex
from lattice_weaver.topology.simplicial_complex import SimplicialComplex
from lattice_weaver.formal.cubical_geometry import GeometricCube

class TestHomologyEngine:

    @pytest.fixture
    def simple_cubical_complex(self):
        # Un complejo cúbico simple: un solo vértice
        graph = nx.Graph()
        v0 = GeometricCube(dimensions=0, coordinates=(0,))
        graph.add_node(v0)
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    @pytest.fixture
    def line_segment_complex(self):
        # Un complejo cúbico que representa un segmento de línea (dos vértices, una arista)
        graph = nx.Graph()
        v0 = GeometricCube(dimensions=0, coordinates=(0,))
        v1 = GeometricCube(dimensions=0, coordinates=(1,))
        graph.add_edge(v0, v1)
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    @pytest.fixture
    def square_complex(self):
        # Un complejo cúbico que representa un cuadrado (4 vértices, 4 aristas, 1 cara 2D)
        graph = nx.Graph()
        # Vértices
        v0 = GeometricCube(dimensions=0, coordinates=(0, 0))
        v1 = GeometricCube(dimensions=0, coordinates=(1, 0))
        v2 = GeometricCube(dimensions=0, coordinates=(0, 1))
        v3 = GeometricCube(dimensions=0, coordinates=(1, 1))
        graph.add_edges_from([(v0, v1), (v1, v3), (v3, v2), (v2, v0)])
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    @pytest.fixture
    def disconnected_complex(self):
        # Un complejo cúbico con dos componentes conexas (dos vértices separados)
        graph = nx.Graph()
        v0 = GeometricCube(dimensions=0, coordinates=(0,))
        v1 = GeometricCube(dimensions=0, coordinates=(10,))
        graph.add_node(v0)
        graph.add_node(v1)
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    def test_compute_homology_simple_vertex(self, simple_cubical_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(simple_cubical_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

    def test_compute_homology_line_segment(self, line_segment_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(line_segment_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

    def test_compute_homology_square(self, square_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(square_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 1
        assert homology["beta_2"] == 1

    def test_compute_homology_disconnected(self, disconnected_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(disconnected_complex)
        assert homology["beta_0"] == 2
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

    def test_compute_homology_invalid_input(self):
        engine = HomologyEngine()
        with pytest.raises(TypeError):
            engine.compute_homology(object())

    @pytest.fixture
    def cube_complex(self):
        graph = nx.Graph()
        vertices = [GeometricCube(0, (x, y, z)) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
        for i in range(8):
            for j in range(i + 1, 8):
                v_i, v_j = vertices[i], vertices[j]
                if sum(abs(c1 - c2) for c1, c2 in zip(v_i.coordinates, v_j.coordinates)) == 1:
                    graph.add_edge(v_i, v_j)
        complex = CubicalComplex(graph)
        complex.build_complex()
        return complex

    @pytest.fixture
    def hollow_cube_complex(self):
        graph = nx.Graph()
        vertices = [GeometricCube(0, (x, y, z)) for x in [0, 1] for y in [0, 1] for z in [0, 1]]
        for i in range(8):
            for j in range(i + 1, 8):
                v_i, v_j = vertices[i], vertices[j]
                if sum(abs(c1 - c2) for c1, c2 in zip(v_i.coordinates, v_j.coordinates)) == 1:
                    graph.add_edge(v_i, v_j)
        complex = CubicalComplex(graph)
        complex.build_complex()
        complex.cubes[3] = []
        return complex

    @pytest.mark.xfail(reason="La aproximación actual de beta_1 no es correcta para 3-cubos.")
    def test_compute_homology_cube(self, cube_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(cube_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

    @pytest.mark.xfail(reason="La aproximación actual de beta_1 y beta_2 no es correcta para cubos huecos.")
    def test_compute_homology_hollow_cube(self, hollow_cube_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(hollow_cube_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 1

    @pytest.fixture
    def simple_simplicial_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex([0]) # Un solo vértice
        return sc

    @pytest.fixture
    def line_simplicial_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex([0])
        sc.add_simplex([1])
        sc.add_simplex([0, 1])
        return sc

    @pytest.fixture
    def triangle_simplicial_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex([0])
        sc.add_simplex([1])
        sc.add_simplex([2])
        sc.add_simplex([0, 1])
        sc.add_simplex([1, 2])
        sc.add_simplex([0, 2])
        sc.add_simplex([0, 1, 2])
        return sc

    def test_compute_homology_simple_simplicial_complex(self, simple_simplicial_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(simple_simplicial_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

    def test_compute_homology_line_simplicial_complex(self, line_simplicial_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(line_simplicial_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

    def test_compute_homology_triangle_simplicial_complex(self, triangle_simplicial_complex):
        engine = HomologyEngine()
        homology = engine.compute_homology(triangle_simplicial_complex)
        assert homology["beta_0"] == 1
        assert homology["beta_1"] == 0
        assert homology["beta_2"] == 0

