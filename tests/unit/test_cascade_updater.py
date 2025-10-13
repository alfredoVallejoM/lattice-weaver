import pytest
from lattice_weaver.topology.simplicial_complex import SimplicialComplex
from lattice_weaver.topology.homology_engine import HomologyEngine
from lattice_weaver.topology.cascade_updater import CascadeUpdater

@pytest.fixture
def homology_engine():
    return HomologyEngine()

def test_initialization(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    assert updater.previous_homology == {"beta_0": 0, "beta_1": 0, "beta_2": 0}

def test_add_single_vertex(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    result = updater.add_element_and_update([0])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {"beta_0": "Aumento en 1 componentes/ciclos/cavidades de dimensión 0"}

def test_add_two_disconnected_vertices(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    updater.add_element_and_update([0])
    result = updater.add_element_and_update([1])
    assert result["current_homology"] == {"beta_0": 2, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {"beta_0": "Aumento en 1 componentes/ciclos/cavidades de dimensión 0"}

def test_connect_two_vertices(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    updater.add_element_and_update([0])
    updater.add_element_and_update([1])
    result = updater.add_element_and_update([0, 1])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {"beta_0": "Disminución en 1 componentes/ciclos/cavidades de dimensión 0"}

def test_create_a_cycle(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    updater.add_element_and_update([0])
    updater.add_element_and_update([1])
    updater.add_element_and_update([2])
    updater.add_element_and_update([0, 1])
    updater.add_element_and_update([1, 2])
    result = updater.add_element_and_update([0, 2])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {}

def test_fill_a_cycle(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    updater.add_element_and_update([0])
    updater.add_element_and_update([1])
    updater.add_element_and_update([2])
    updater.add_element_and_update([0, 1])
    updater.add_element_and_update([1, 2])
    updater.add_element_and_update([0, 2])
    result = updater.add_element_and_update([0, 1, 2])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {}

def test_no_change_in_homology(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)
    updater.add_element_and_update([0])
    result = updater.add_element_and_update([0])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {}

