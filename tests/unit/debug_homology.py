import pytest
from lattice_weaver.topology.simplicial_complex import SimplicialComplex
from lattice_weaver.topology.homology_engine import HomologyEngine

def test_two_disconnected_vertices():
    sc = SimplicialComplex()
    sc.add_simplex([0])
    sc.add_simplex([1])
    engine = HomologyEngine()
    homology = engine.compute_homology(sc)
    assert homology["beta_0"] == 2

