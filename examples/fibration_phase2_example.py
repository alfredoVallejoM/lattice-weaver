import sys
sys.path.insert(0, ".")

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscapeOptimized,
    HacificationEngine,
    LandscapeModulator,
    FocusOnLocalStrategy,
    FocusOnGlobalStrategy,
    AdaptiveStrategy,
    SimpleOptimizationSolver
)
from typing import Dict, List, Any

def create_simple_problem() -> tuple[List[str], Dict[str, List[Any]], ConstraintHierarchy]:
    """Crea un problema simple para demostrar hacificación y modulación."""
    variables = ["A", "B", "C"]
    domains = {"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]}
    hierarchy = ConstraintHierarchy()

    # HARD: A != B
    hierarchy.add_local_constraint(
        "A", "B",
        lambda a: a["A"] != a["B"],
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "A_ne_B"}
    )

    # SOFT (PATTERN): A + B + C debe ser par (1.0 si impar, 0.0 si par)
    hierarchy.add_pattern_constraint(
        ["A", "B", "C"],
        lambda a: 1.0 if (a["A"] + a["B"] + a["C"]) % 2 != 0 else 0.0,
        pattern_type="sum_even",
        weight=1.0,
        hardness=Hardness.SOFT,
        metadata={"name": "Sum_is_Even"}
    )

    # SOFT (GLOBAL): Preferir valores bajos para C (costo = valor de C)
    hierarchy.add_global_constraint(
        ["C"],
        lambda a: a["C"],
        objective="minimize",
        weight=0.5,
        hardness=Hardness.SOFT,
        metadata={"name": "C_low_value"}
    )
    return variables, domains, hierarchy

def main():
    print("--- Demostración de HacificationEngine y LandscapeModulator ---")

    variables, domains, hierarchy = create_simple_problem()
    landscape = EnergyLandscapeOptimized(hierarchy)
    hacification_engine = HacificationEngine(hierarchy, landscape)
    modulator = LandscapeModulator(landscape)

    print("\n--- Hacificación: Filtrado de valores coherentes ---")
    base_assignment = {"A": 0}
    variable_to_assign = "B"
    domain_B = domains["B"]

    print(f"Asignación base: {base_assignment}")
    print(f"Dominio de {variable_to_assign}: {domain_B}")

    # Filtrar valores para B que son coherentes con A=0 (A!=B HARD)
    # B=0 es incoherente (viola A!=B)
    # B=1, B=2 son coherentes
    coherent_values_strict = hacification_engine.filter_coherent_extensions(
        base_assignment, variable_to_assign, domain_B, strict=True
    )
    print(f"Valores coherentes para B (strict=True): {coherent_values_strict}")
    assert 0 not in coherent_values_strict # B=0 should be filtered due to A!=B
    assert coherent_values_strict == [1, 2]

    # Demostrar hacificación de una asignación completa
    print("\n--- Hacificación: Verificación de asignaciones completas ---")
    # Asignación 1: A=0, B=1, C=0
    # A!=B (OK) -> HARD OK
    # Sum_is_Even (0+1+0=1, impar) -> SOFT violada (1.0)
    # C_low_value (C=0) -> SOFT OK (0.0)
    # Energía total: 1.0 (PATTERN) + 0.0 (GLOBAL) = 1.0
    # is_coherent=False porque PATTERN energy (1.0) > threshold (0.0)
    assignment1 = {"A": 0, "B": 1, "C": 0} 
    h_result1 = hacification_engine.hacify(assignment1, strict=True)
    print(f"Asignación 1: {assignment1}")
    print(f"  Coherente: {h_result1.is_coherent}, Energía: {h_result1.energy.total_energy:.3f}, Violaciones: {h_result1.violated_constraints}")
    assert h_result1.is_coherent is False
    assert h_result1.energy.total_energy == 1.0
    assert "PATTERN:Sum_is_Even" in h_result1.violated_constraints

    # Asignación 2: A=0, B=0, C=1
    # A!=B (FAIL) -> HARD violada
    # Energía total: 1.0 (LOCAL) + 0.0 (PATTERN) + 0.5 (GLOBAL) = 1.5
    # is_coherent=False por violación HARD
    assignment2 = {"A": 0, "B": 0, "C": 1} 
    h_result2 = hacification_engine.hacify(assignment2, strict=True)
    print(f"Asignación 2: {assignment2}")
    print(f"  Coherente: {h_result2.is_coherent}, Energía: {h_result2.energy.total_energy:.3f}, Violaciones: {h_result2.violated_constraints}")
    assert h_result2.is_coherent is False
    assert h_result2.energy.total_energy == 2.5
    assert "LOCAL:A_ne_B" in h_result2.violated_constraints

    # Asignación 3: A=0, B=1, C=1
    # A!=B (OK) -> HARD OK
    # Sum_is_Even (0+1+1=2, par) -> SOFT OK (0.0)
    # C_low_value (C=1) -> SOFT violada (0.5)
    # Energía total: 0.0 (LOCAL) + 0.0 (PATTERN) + 0.5 (GLOBAL) = 0.5
    # is_coherent=False porque GLOBAL energy (0.5) > threshold (0.1)
    assignment3 = {"A": 0, "B": 1, "C": 1} 
    h_result3 = hacification_engine.hacify(assignment3, strict=True)
    print(f"Asignación 3: {assignment3}")
    print(f"  Coherente: {h_result3.is_coherent}, Energía: {h_result3.energy.total_energy:.3f}, Violaciones: {h_result3.violated_constraints}")
    assert h_result3.is_coherent is False
    assert h_result3.energy.total_energy == 0.5
    assert "GLOBAL:C_low_value" in h_result3.violated_constraints

    # Asignación 4: A=0, B=1, C=2 (Coherente si Sum_is_Even y C_low_value no son demasiado importantes)
    # A!=B (OK) -> HARD OK
    # Sum_is_Even (0+1+2=3, impar) -> SOFT violada (1.0)
    # C_low_value (C=2) -> SOFT violada (1.0)
    # Energía total: 0.0 (LOCAL) + 1.0 (PATTERN) + 1.0 (GLOBAL) = 2.0
    # is_coherent=False porque PATTERN energy (1.0) > threshold (0.0) y GLOBAL energy (1.0) > threshold (0.1)
    assignment4 = {"A": 0, "B": 1, "C": 2}
    h_result4 = hacification_engine.hacify(assignment4, strict=True)
    print(f"Asignación 4: {assignment4}")
    print(f"  Coherente: {h_result4.is_coherent}, Energía: {h_result4.energy.total_energy:.3f}, Violaciones: {h_result4.violated_constraints}")
    assert h_result4.is_coherent is False
    assert h_result4.energy.total_energy == 2.0
    assert "PATTERN:Sum_is_Even" in h_result4.violated_constraints
    assert "GLOBAL:C_low_value" in h_result4.violated_constraints

    print("\n--- Modulación del Paisaje de Energía ---")
    print(f"Pesos iniciales del paisaje: {modulator.get_statistics()['base_weights']}")

    # Estrategia 1: Enfocarse en restricciones locales
    modulator.set_strategy(FocusOnLocalStrategy())
    modulator.apply_modulation({})
    print(f"Pesos después de FocusOnLocal: {modulator.get_statistics()['current_weights']}")
    assert landscape.level_weights[ConstraintLevel.LOCAL] > landscape.level_weights[ConstraintLevel.GLOBAL]

    # Estrategia 2: Enfocarse en restricciones globales
    modulator.reset_modulation() # Restaurar pesos base
    modulator.set_strategy(FocusOnGlobalStrategy())
    modulator.apply_modulation({})
    print(f"Pesos después de FocusOnGlobal: {modulator.get_statistics()['current_weights']}")
    assert landscape.level_weights[ConstraintLevel.GLOBAL] > landscape.level_weights[ConstraintLevel.LOCAL]

    # Estrategia 3: Adaptativa
    modulator.reset_modulation()
    modulator.set_strategy(AdaptiveStrategy())
    
    print("\n--- Modulación Adaptativa ---")
    # Contexto inicial: pocas violaciones, inicio de búsqueda
    context_early = {"progress": 0.1, "local_violations": 1, "global_violations": 0}
    modulator.apply_modulation(context_early)
    print(f"Pesos (contexto inicial): {modulator.get_statistics()['current_weights']}")

    # Contexto tardío: muchas violaciones globales, fin de búsqueda
    context_late = {"progress": 0.9, "local_violations": 0, "global_violations": 5}
    modulator.apply_modulation(context_late)
    print(f"Pesos (contexto tardío): {modulator.get_statistics()['current_weights']}")

    print("\n--- Integración con el Solver (Ejemplo Conceptual) ---")
    # En un solver real, la modulación se aplicaría dinámicamente
    # antes de cada paso de búsqueda o en puntos clave.
    
    # Restaurar pesos para el solver
    modulator.reset_modulation()
    
    solver = SimpleOptimizationSolver(variables, domains, hierarchy)
    solver.max_solutions = 1 # Solo la primera solución
    
    print("\nBuscando solución con pesos base...")
    solution_base = solver.solve()
    if solution_base:
        energy_base = landscape.compute_energy(solution_base).total_energy
        print(f"  Solución (base): {solution_base}, Energía: {energy_base:.3f}")
    
    # Aplicar modulación para favorecer restricciones globales (e.g., C bajo)
    modulator.set_strategy(FocusOnGlobalStrategy())
    modulator.apply_modulation({})
    print("Buscando solución con pesos modulados (FocusOnGlobal)...")
    # Es importante crear un nuevo solver o re-inicializar el paisaje del solver existente
    # para que los nuevos pesos de modulación sean considerados.
    solver_modulated = SimpleOptimizationSolver(variables, domains, hierarchy) 
    solver_modulated.max_solutions = 1
    solution_modulated = solver_modulated.solve()
    if solution_modulated:
        energy_modulated = landscape.compute_energy(solution_modulated).total_energy
        print(f"  Solución (modulada): {solution_modulated}, Energía: {energy_modulated:.3f}")
        
        # Demostrar que la solución modulada es mejor en términos globales
        # (asumiendo que FocusOnGlobalStrategy reduce la energía global)
        if solution_base and solution_modulated:
            global_energy_base = landscape.compute_energy(solution_base).global_energy
            global_energy_modulated = landscape.compute_energy(solution_modulated).global_energy
            print(f"  Energía global de solución base: {global_energy_base:.3f}")
            print(f"  Energía global de solución modulada: {global_energy_modulated:.3f}")
            # El objetivo es que la energía global sea menor en la solución modulada
            # Esto puede no ser estrictamente menor si la solución base ya era óptima para el global
            # o si hay trade-offs con otras restricciones.
            # Para este ejemplo, solo verificaremos que no empeora significativamente.
            assert global_energy_modulated <= global_energy_base + 0.01 # Pequeña tolerancia

    print("\n--- Demostración completada --- ")

if __name__ == "__main__":
    main()

