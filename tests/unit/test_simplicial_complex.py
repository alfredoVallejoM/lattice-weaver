import pytest
import gudhi
from lattice_weaver.topology.simplicial_complex import SimplicialComplex

class TestSimplicialComplex:

    @pytest.fixture
    def empty_complex(self):
        return SimplicialComplex()

    @pytest.fixture
    def point_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex([0])
        return sc

    @pytest.fixture
    def line_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex([0])
        sc.add_simplex([1])
        sc.add_simplex([0, 1])
        return sc

    @pytest.fixture
    def triangle_complex(self):
        sc = SimplicialComplex()
        sc.add_simplex([0])
        sc.add_simplex([1])
        sc.add_simplex([2])
        sc.add_simplex([0, 1])
        sc.add_simplex([1, 2])
        sc.add_simplex([0, 2])
        sc.add_simplex([0, 1, 2])
        return sc

    def test_empty_complex_initialization(self, empty_complex):
        assert empty_complex.num_vertices() == 0
        assert empty_complex.num_simplices() == 0
        assert empty_complex.get_max_dimension() == -1
        assert not empty_complex.is_built()

    def test_add_simplex_point(self, point_complex):
        assert point_complex.num_vertices() == 1
        assert point_complex.num_simplices() == 1  # [0]
        assert point_complex.get_max_dimension() == 0
        assert point_complex.is_built()
        simplices = point_complex.get_simplices()
        assert len(simplices) == 1
        assert ([0], 0.0) in simplices

    def test_add_simplex_line(self, line_complex):
        assert line_complex.num_vertices() == 2
        assert line_complex.num_simplices() == 3  # [0], [1], [0,1]
        assert line_complex.get_max_dimension() == 1
        simplices = line_complex.get_simplices()
        assert len(simplices) == 3
        assert ([0], 0.0) in simplices
        assert ([1], 0.0) in simplices
        assert ([0, 1], 0.0) in simplices

    def test_add_simplex_triangle(self, triangle_complex):
        assert triangle_complex.num_vertices() == 3
        # [0], [1], [2], [0,1], [1,2], [0,2], [0,1,2]
        assert triangle_complex.num_simplices() == 7
        assert triangle_complex.get_max_dimension() == 2
        simplices_dim_0 = triangle_complex.get_simplices(0)
        assert len(simplices_dim_0) == 3
        assert ([0], 0.0) in simplices_dim_0
        assert ([1], 0.0) in simplices_dim_0
        assert ([2], 0.0) in simplices_dim_0
        simplices_dim_1 = triangle_complex.get_simplices(1)
        assert len(simplices_dim_1) == 3
        assert ([0, 1], 0.0) in simplices_dim_1
        assert ([1, 2], 0.0) in simplices_dim_1
        assert ([0, 2], 0.0) in simplices_dim_1
        simplices_dim_2 = triangle_complex.get_simplices(2)
        assert len(simplices_dim_2) == 1
        assert ([0, 1, 2], 0.0) in simplices_dim_2

    def test_add_simplex_with_filtration(self, empty_complex):
        empty_complex.add_simplex([0], filtration=1.0)
        empty_complex.add_simplex([0, 1], filtration=2.0)
        simplices = empty_complex.get_simplices()
        assert ([0], 1.0) in simplices
        assert ([0, 1], 2.0) in simplices

    def test_add_simplex_invalid_input(self, empty_complex):
        with pytest.raises(ValueError, match="Los vértices del simplex deben ser enteros."):
            empty_complex.add_simplex([0, "a"])

    def test_get_simplices_by_dimension(self, triangle_complex):
        # Test dimension 0
        dim_0_simplices = triangle_complex.get_simplices(0)
        assert len(dim_0_simplices) == 3
        assert all(len(s[0]) == 1 for s in dim_0_simplices)

        # Test dimension 1
        dim_1_simplices = triangle_complex.get_simplices(1)
        assert len(dim_1_simplices) == 3
        assert all(len(s[0]) == 2 for s in dim_1_simplices)

        # Test dimension 2
        dim_2_simplices = triangle_complex.get_simplices(2)
        assert len(dim_2_simplices) == 1
        assert all(len(s[0]) == 3 for s in dim_2_simplices)

        # Test dimension higher than max
        dim_3_simplices = triangle_complex.get_simplices(3)
        assert len(dim_3_simplices) == 0

    def test_string_representation(self, triangle_complex):
        s = str(triangle_complex)
        assert "SimplicialComplex" in s
        assert "num_vertices=3" in s
        assert "num_simplices=7" in s
        assert "max_dim=2" in s
        assert "([0, 1, 2], 0.0)" in s

