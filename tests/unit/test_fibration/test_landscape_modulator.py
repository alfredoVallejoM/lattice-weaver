import pytest
from typing import Dict, List, Any

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscapeOptimized,
    LandscapeModulator,
    FocusOnLocalStrategy,
    FocusOnGlobalStrategy,
    AdaptiveStrategy
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
    # Restricción SOFT patrón: Q0 + Q1 = 2
    hierarchy.add_pattern_constraint(
        ["Q0", "Q1"],
        lambda a, q0="Q0", q1="Q1": abs(a[q0] + a[q1] - 2),
        pattern_type="sum_to_2",
        weight=1.0,
        hardness=Hardness.SOFT,
        metadata={"name": "Q0_plus_Q1_eq_2"}
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
def landscape_modulator(simple_landscape):
    return LandscapeModulator(simple_landscape)

# --- Tests para LandscapeModulator --- #

def test_landscape_modulator_init(landscape_modulator, simple_landscape):
    assert landscape_modulator.landscape == simple_landscape
    assert landscape_modulator.current_strategy is None
    assert landscape_modulator.base_weights == simple_landscape.level_weights

def test_set_strategy(landscape_modulator):
    strategy = FocusOnLocalStrategy()
    landscape_modulator.set_strategy(strategy)
    assert landscape_modulator.current_strategy == strategy

def test_apply_modulation_focus_on_local(landscape_modulator, simple_landscape):
    landscape_modulator.set_strategy(FocusOnLocalStrategy())
    initial_weights = simple_landscape.level_weights.copy()
    
    landscape_modulator.apply_modulation({})
    
    assert simple_landscape.level_weights[ConstraintLevel.LOCAL] == initial_weights[ConstraintLevel.LOCAL] * 2.0
    assert simple_landscape.level_weights[ConstraintLevel.PATTERN] == initial_weights[ConstraintLevel.PATTERN] * 1.0
    assert simple_landscape.level_weights[ConstraintLevel.GLOBAL] == initial_weights[ConstraintLevel.GLOBAL] * 0.5

def test_apply_modulation_focus_on_global(landscape_modulator, simple_landscape):
    landscape_modulator.set_strategy(FocusOnGlobalStrategy())
    initial_weights = simple_landscape.level_weights.copy()
    
    landscape_modulator.apply_modulation({})
    
    assert simple_landscape.level_weights[ConstraintLevel.LOCAL] == initial_weights[ConstraintLevel.LOCAL] * 0.5
    assert simple_landscape.level_weights[ConstraintLevel.PATTERN] == initial_weights[ConstraintLevel.PATTERN] * 1.0
    assert simple_landscape.level_weights[ConstraintLevel.GLOBAL] == initial_weights[ConstraintLevel.GLOBAL] * 2.0

def test_apply_modulation_adaptive(landscape_modulator, simple_landscape):
    landscape_modulator.set_strategy(AdaptiveStrategy())
    initial_weights = simple_landscape.level_weights.copy()
    
    # Context: early stage, more local violations
    context = {"progress": 0.2, "local_violations": 5, "global_violations": 1}
    landscape_modulator.apply_modulation(context)
    
    # Expected: local_weight = (2.0 - 0.2) * 1.5 = 1.8 * 1.5 = 2.7
    # global_weight = (1.0 + 0.2) = 1.2
    assert simple_landscape.level_weights[ConstraintLevel.LOCAL] == pytest.approx(initial_weights[ConstraintLevel.LOCAL] * 2.7)
    assert simple_landscape.level_weights[ConstraintLevel.PATTERN] == pytest.approx(initial_weights[ConstraintLevel.PATTERN] * 1.0)
    assert simple_landscape.level_weights[ConstraintLevel.GLOBAL] == pytest.approx(initial_weights[ConstraintLevel.GLOBAL] * 1.2)
    
    # Context: late stage, more global violations
    context = {"progress": 0.8, "local_violations": 1, "global_violations": 5}
    landscape_modulator.apply_modulation(context)
    
    # Expected: local_weight = (2.0 - 0.8) = 1.2
    # global_weight = (1.0 + 0.8) * 1.5 = 1.8 * 1.5 = 2.7
    assert simple_landscape.level_weights[ConstraintLevel.LOCAL] == pytest.approx(initial_weights[ConstraintLevel.LOCAL] * 1.2)
    assert simple_landscape.level_weights[ConstraintLevel.PATTERN] == pytest.approx(initial_weights[ConstraintLevel.PATTERN] * 1.0)
    assert simple_landscape.level_weights[ConstraintLevel.GLOBAL] == pytest.approx(initial_weights[ConstraintLevel.GLOBAL] * 2.7)

def test_reset_modulation(landscape_modulator, simple_landscape):
    landscape_modulator.set_strategy(FocusOnLocalStrategy())
    initial_weights = simple_landscape.level_weights.copy()
    landscape_modulator.apply_modulation({})
    
    assert simple_landscape.level_weights != initial_weights # Weights should have changed
    
    landscape_modulator.reset_modulation()
    assert simple_landscape.level_weights == initial_weights # Weights should be reset

def test_modulator_clears_landscape_cache(landscape_modulator, simple_landscape):
    landscape_modulator.set_strategy(FocusOnLocalStrategy())
    # Simulate some cache entries
    simple_landscape._energy_cache["test_assignment"] = 10.0
    # The _gradient_cache_by_var attribute is not present in the optimized version.
    # The cache is handled by the _energy_cache attribute.
    # We can check if the energy cache is cleared.
    pass
    
    landscape_modulator.apply_modulation({})
    
    assert not simple_landscape._energy_cache # Cache should be cleared
    # The _gradient_cache_by_var attribute is not present in the optimized version.
    # The cache is handled by the _energy_cache attribute.
    # We can check if the energy cache is cleared.
    pass

def test_get_statistics(landscape_modulator, simple_landscape):
    landscape_modulator.set_strategy(FocusOnLocalStrategy())
    stats = landscape_modulator.get_statistics()
    assert stats["current_strategy"] == "focus_on_local"
    assert stats["base_weights"] == {level.name: weight for level, weight in simple_landscape.level_weights.items()}
    assert stats["current_weights"] == {level.name: weight for level, weight in simple_landscape.level_weights.items()}

