import pytest
from typing import Dict, List, Any
from unittest.mock import Mock

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscapeOptimized,
    HacificationEngine,
    HacificationResult
)

# --- Fixtures para un problema simple --- #

@pytest.fixture
def simple_hierarchy():
    hierarchy = ConstraintHierarchy()
    # Restricción HARD local: Q0 != Q1
    hierarchy.add_local_constraint(
        "Q0", "Q1",
        lambda a, q0="Q0", q1="Q1": a[q0] != a[q1],
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "Q0_ne_Q1"}
    )
    # Restricción SOFT global: Preferir Q0=0
    hierarchy.add_global_constraint(
        ["Q0"],
        lambda a, q0="Q0": 1.0 if a[q0] != 0 else 0.0,
        objective="minimize",
        weight=1.0,
        hardness=Hardness.SOFT,
        metadata={"name": "Q0_is_0"}
    )
    return hierarchy

@pytest.fixture
def simple_landscape(simple_hierarchy):
    return EnergyLandscapeOptimized(simple_hierarchy)

@pytest.fixture
def hacification_engine(simple_hierarchy, simple_landscape):
    return HacificationEngine(simple_hierarchy, simple_landscape)

@pytest.fixture
def mock_arc_engine():
    mock = Mock(spec=['reset', 'enforce_arc_consistency', 'add_variable'])
    mock.reset.return_value = None
    mock.enforce_arc_consistency.return_value = True # Simular que es consistente
    mock.add_variable.return_value = None
    return mock

# --- Tests para HacificationEngine --- #

def test_hacification_engine_init(hacification_engine):
    assert isinstance(hacification_engine.hierarchy, ConstraintHierarchy)
    assert isinstance(hacification_engine.landscape, EnergyLandscapeOptimized)
    assert hacification_engine.energy_thresholds[ConstraintLevel.LOCAL] == 0.0
    assert hacification_engine.energy_thresholds[ConstraintLevel.GLOBAL] == 0.1

def test_hacify_coherent_assignment_strict(hacification_engine):
    assignment = {"Q0": 0, "Q1": 1}
    result = hacification_engine.hacify(assignment, strict=True)
    
    assert result.is_coherent is True
    assert result.level_results[ConstraintLevel.LOCAL] is True
    assert result.level_results[ConstraintLevel.GLOBAL] is True
    assert result.energy.total_energy == 0.0  # Q0=0, Q0!=Q1
    assert not result.violated_constraints

def test_hacify_incoherent_assignment_strict_hard_violation(hacification_engine):
    assignment = {"Q0": 0, "Q1": 0} # Violates Q0 != Q1 (HARD)
    result = hacification_engine.hacify(assignment, strict=True)
    
    assert result.is_coherent is False
    assert result.level_results[ConstraintLevel.LOCAL] is False
    assert result.level_results[ConstraintLevel.GLOBAL] is True 
    assert result.energy.total_energy > 0.0
    assert "LOCAL:Q0_ne_Q1" in result.violated_constraints

def test_hacify_incoherent_assignment_strict_soft_violation(hacification_engine):
    assignment = {"Q0": 1, "Q1": 0} # Violates Q0=0 (SOFT), but Q0!=Q1 (HARD) is met
    result = hacification_engine.hacify(assignment, strict=True)
    
    assert result.is_coherent is True 
    assert result.level_results[ConstraintLevel.LOCAL] is True
    assert result.level_results[ConstraintLevel.GLOBAL] is False
    assert result.energy.total_energy > 0.0
    assert "GLOBAL:Q0_is_0" in result.violated_constraints # SOFT violation should be listed

def test_hacify_incoherent_assignment_non_strict_soft_violation(hacification_engine):
    assignment = {"Q0": 1, "Q1": 0} # Violates Q0=0 (SOFT)
    result = hacification_engine.hacify(assignment, strict=False)
    
    assert result.is_coherent is False
    assert result.level_results[ConstraintLevel.LOCAL] is True
    assert result.level_results[ConstraintLevel.GLOBAL] is False
    assert result.energy.total_energy > 0.0
    assert "GLOBAL:Q0_is_0" in result.violated_constraints

def test_filter_coherent_extensions(hacification_engine):
    base_assignment = {"Q1": 0}
    variable = "Q0"
    domain = [0, 1, 2, 3]
    
    hierarchy_hard_only = ConstraintHierarchy()
    hierarchy_hard_only.add_local_constraint(
        "Q0", "Q1",
        lambda a, q0="Q0", q1="Q1": a[q0] != a[q1],
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "Q0_ne_Q1"}
    )
    landscape_hard_only = EnergyLandscapeOptimized(hierarchy_hard_only)
    hacification_engine_hard_only = HacificationEngine(hierarchy_hard_only, landscape_hard_only)
    
    coherent_values_hard_only = hacification_engine_hard_only.filter_coherent_extensions(base_assignment, variable, domain, strict=True)
    assert coherent_values_hard_only == [1, 2, 3] # Q0=0, Q1=0 violates HARD

def test_hacification_engine_get_statistics(hacification_engine):
    stats = hacification_engine.get_statistics()
    assert "energy_thresholds" in stats
    assert stats["energy_thresholds"]["LOCAL"] == 0.0


# --- Tests para Retrocompatibilidad y ArcEngine Opcional --- #

def test_retrocompatibility_init(simple_hierarchy, simple_landscape):
    """Verifica que el constructor sigue funcionando sin los nuevos parámetros."""
    engine = HacificationEngine(simple_hierarchy, simple_landscape)
    assert not engine._use_arc_engine
    assert engine._arc_engine is None

def test_arc_engine_init_disabled(simple_hierarchy, simple_landscape, mock_arc_engine):
    """Verifica que se puede pasar ArcEngine pero no usarlo si use_arc_engine=False."""
    engine = HacificationEngine(simple_hierarchy, simple_landscape, arc_engine=mock_arc_engine, use_arc_engine=False)
    assert not engine._use_arc_engine
    assert engine._arc_engine == mock_arc_engine

def test_arc_engine_init_enabled(simple_hierarchy, simple_landscape, mock_arc_engine):
    """Verifica que se puede habilitar el uso de ArcEngine."""
    engine = HacificationEngine(simple_hierarchy, simple_landscape, arc_engine=mock_arc_engine, use_arc_engine=True)
    assert engine._use_arc_engine
    assert engine._arc_engine == mock_arc_engine

def test_hacify_delegates_to_original_when_arc_engine_disabled(hacification_engine, mocker):
    """Verifica que se llama a _hacify_original cuando ArcEngine está deshabilitado."""
    spy_original = mocker.spy(hacification_engine, "_hacify_original")
    spy_arc_engine = mocker.spy(hacification_engine, "_hacify_with_arc_engine")
    
    assignment = {"Q0": 0, "Q1": 1}
    hacification_engine.hacify(assignment)
    
    spy_original.assert_called_once_with(assignment, strict=True)
    spy_arc_engine.assert_not_called()

def test_hacify_delegates_to_arc_engine_when_enabled(simple_hierarchy, simple_landscape, mock_arc_engine, mocker):
    """Verifica que se llama a _hacify_with_arc_engine cuando ArcEngine está habilitado."""
    engine = HacificationEngine(simple_hierarchy, simple_landscape, arc_engine=mock_arc_engine, use_arc_engine=True)
    
    spy_original = mocker.spy(engine, "_hacify_original")
    spy_arc_engine = mocker.spy(engine, "_hacify_with_arc_engine")
    
    assignment = {"Q0": 0, "Q1": 1}
    engine.hacify(assignment)
    
    spy_arc_engine.assert_called_once_with(assignment, strict=True)
    spy_original.assert_not_called()

def test_get_statistics_with_arc_engine(simple_hierarchy, simple_landscape, mock_arc_engine):
    """Verifica que las estadísticas reportan el uso de ArcEngine."""
    engine = HacificationEngine(simple_hierarchy, simple_landscape, arc_engine=mock_arc_engine, use_arc_engine=True)
    stats = engine.get_statistics()
    assert stats["use_arc_engine"] is True

    engine_disabled = HacificationEngine(simple_hierarchy, simple_landscape)
    stats_disabled = engine_disabled.get_statistics()
    assert stats_disabled["use_arc_engine"] is False

