import pytest
from lattice_weaver.topology.simplicial_complex import SimplicialComplex
from lattice_weaver.topology.homology_engine import HomologyEngine
from lattice_weaver.topology.cascade_updater import CascadeUpdater

@pytest.fixture
def homology_engine():
    return HomologyEngine()

def test_simplicial_homology_cascade_integration_single_vertex(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)

    # Añadir un solo vértice
    result = updater.add_element_and_update([0])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {"beta_0": "Aumento en 1 componentes/ciclos/cavidades de dimensión 0"}

def test_simplicial_homology_cascade_integration_connected_vertices(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)

    # Añadir dos vértices desconectados
    updater.add_element_and_update([0])
    result = updater.add_element_and_update([1])
    assert result["current_homology"] == {"beta_0": 2, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {"beta_0": "Aumento en 1 componentes/ciclos/cavidades de dimensión 0"}

    # Conectar los dos vértices con una arista
    result = updater.add_element_and_update([0, 1])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {"beta_0": "Disminución en 1 componentes/ciclos/cavidades de dimensión 0"}

def test_simplicial_homology_cascade_integration_create_and_fill_cycle(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)

    # Añadir vértices y aristas para formar un ciclo (triángulo)
    updater.add_element_and_update([0])
    updater.add_element_and_update([1])
    updater.add_element_and_update([2])
    updater.add_element_and_update([0, 1])
    updater.add_element_and_update([1, 2])
    result = updater.add_element_and_update([0, 2])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {}

    # Rellenar el ciclo con un 2-simplex
    result = updater.add_element_and_update([0, 1, 2])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {}

def test_simplicial_homology_cascade_integration_complex_structure(homology_engine):
    sc = SimplicialComplex()
    updater = CascadeUpdater(sc, homology_engine)

    # Añadir una estructura más compleja (ej. un tetraedro)
    updater.add_element_and_update([0])
    updater.add_element_and_update([1])
    updater.add_element_and_update([2])
    updater.add_element_and_update([3])
    updater.add_element_and_update([0, 1])
    updater.add_element_and_update([0, 2])
    updater.add_element_and_update([0, 3])
    updater.add_element_and_update([1, 2])
    updater.add_element_and_update([1, 3])
    updater.add_element_and_update([2, 3])
    updater.add_element_and_update([0, 1, 2])
    updater.add_element_and_update([0, 1, 3])
    updater.add_element_and_update([0, 2, 3])
    result = updater.add_element_and_update([1, 2, 3])
    assert result["current_homology"] == {"beta_0": 1, "beta_1": 0, "beta_2": 0}
    assert result["emergent_structures"] == {}

