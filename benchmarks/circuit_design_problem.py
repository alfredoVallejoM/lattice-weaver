from typing import Dict, List, Any, Tuple
from lattice_weaver.fibration import ConstraintHierarchy, ConstraintLevel, Hardness

def create_circuit_design_problem() -> Tuple[List[str], Dict[str, List[Any]], ConstraintHierarchy]:
    """
    Crea un problema de diseño de circuitos con múltiples objetivos en conflicto.

    Objetivo: Diseñar un circuito con 8 compuertas lógicas (G0-G7) y 4 tipos de chips (Chip0-Chip3).
    Cada compuerta debe ser asignada a un chip. Los chips tienen capacidades y características diferentes.

    Variables:
    - `gate_assign_G{i}`: Asignación de la compuerta G{i} a un chip (Chip0, Chip1, Chip2, Chip3).
    - `chip_type_Chip{j}`: Tipo de chip (A, B, C, D) para Chip{j}.

    Dominios:
    - Asignación de compuerta: ["Chip0", "Chip1", "Chip2", "Chip3"]
    - Tipo de chip: ["TypeA", "TypeB", "TypeC", "TypeD"]

    Características de los chips:
    - TypeA: Costo = 120, Capacidad = 2 compuertas, Rendimiento = Muy Alto (penalización 0.05 por compuerta), Potencia = 6
    - TypeB: Costo = 80, Capacidad = 3 compuertas, Rendimiento = Alto (penalización 0.15 por compuerta), Potencia = 4
    - TypeC: Costo = 40, Capacidad = 4 compuertas, Rendimiento = Medio (penalización 0.3 por compuerta), Potencia = 2
    - TypeD: Costo = 10, Capacidad = 5 compuertas, Rendimiento = Bajo (penalización 0.7 por compuerta), Potencia = 1

    Restricciones:
    - HARD: La capacidad de cada chip no debe ser excedida.
    - HARD: Si G0 y G1 están en el mismo chip, deben ser de tipo 'TypeA' (alta velocidad).
    - HARD: G2 y G3 no pueden estar en el mismo chip si ese chip es 'TypeC' o 'TypeD'.
    - HARD: G4 y G5 deben estar en chips diferentes.
    - SOFT (GLOBAL): Minimizar el costo total del circuito (chips + compuertas).
    - SOFT (PATTERN): Maximizar el rendimiento total del circuito (minimizar penalización por rendimiento).
    - SOFT (LOCAL): Penalizar si G6 y G7 están en el mismo chip y es de tipo 'TypeA'.
    - SOFT (GLOBAL): Minimizar el consumo total de potencia del circuito.
    - SOFT (GLOBAL): Penalizar la dispersión de compuertas (preferir menos chips usados).
    """
    gates = [f"G{i}" for i in range(8)] # 8 compuertas
    chips = [f"Chip{i}" for i in range(4)] # 4 chips

    variables = []
    domains = {}

    # Variables para asignación de compuertas
    for gate in gates:
        var_name = f"gate_assign_{gate}"
        variables.append(var_name)
        domains[var_name] = chips

    # Variables para tipo de chip
    for chip in chips:
        var_name = f"chip_type_{chip}"
        variables.append(var_name)
        domains[var_name] = ["TypeA", "TypeB", "TypeC", "TypeD"]

    hierarchy = ConstraintHierarchy()

    # HARD: La capacidad de cada chip no debe ser excedida.
    # TypeA capacity = 2, TypeB capacity = 3, TypeC capacity = 4, TypeD capacity = 5
    for chip_idx, chip_name in enumerate(chips):
        chip_type_var = f"chip_type_{chip_name}"
        assigned_gates_vars = [f"gate_assign_{g}" for g in gates]

        def chip_capacity_hard_predicate(assignment: Dict[str, Any], ct_var: str, ag_vars: List[str], current_chip_name: str) -> bool:
            if ct_var not in assignment: return True # No se ha decidido el tipo de chip
            
            chip_type = assignment[ct_var]
            capacity = {"TypeA": 2, "TypeB": 3, "TypeC": 4, "TypeD": 5}.get(chip_type, 0)
            
            assigned_count = 0
            for gate_assign_var in ag_vars:
                if gate_assign_var in assignment and assignment[gate_assign_var] == current_chip_name:
                    assigned_count += 1
            
            return assigned_count <= capacity

        hierarchy.add_global_constraint(
            [chip_type_var] + assigned_gates_vars,
            lambda a, ct=chip_type_var, ag=assigned_gates_vars, c_name=chip_name: chip_capacity_hard_predicate(a, ct, ag, c_name),
            weight=1.0,
            hardness=Hardness.HARD,
            metadata={"name": f"ChipCapacity_{chip_name}"}
        )

    # HARD: Si G0 y G1 están en el mismo chip, deben ser de tipo 'TypeA' (alta velocidad).
    def g0_g1_same_chip_typeA_hard_predicate(assignment: Dict[str, Any]) -> bool:
        if "gate_assign_G0" not in assignment or "gate_assign_G1" not in assignment: return True
        
        chip0_assigned = assignment["gate_assign_G0"]
        chip1_assigned = assignment["gate_assign_G1"]

        if chip0_assigned == chip1_assigned: # Están en el mismo chip
            chip_type_var = f"chip_type_{chip0_assigned}"
            if chip_type_var not in assignment: return True # Tipo de chip no decidido
            return assignment[chip_type_var] == "TypeA"
        return True

    hierarchy.add_global_constraint(
        ["gate_assign_G0", "gate_assign_G1"] + [f"chip_type_{c}" for c in chips],
        g0_g1_same_chip_typeA_hard_predicate,
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "G0G1_SameChip_TypeA"}
    )

    # HARD: G2 y G3 no pueden estar en el mismo chip si ese chip es 'TypeC' o 'TypeD'.
    def g2_g3_not_same_chip_typeCD_hard_predicate(assignment: Dict[str, Any]) -> bool:
        if "gate_assign_G2" not in assignment or "gate_assign_G3" not in assignment: return True
        
        chip2_assigned = assignment["gate_assign_G2"]
        chip3_assigned = assignment["gate_assign_G3"]

        if chip2_assigned == chip3_assigned: # Están en el mismo chip
            chip_type_var = f"chip_type_{chip2_assigned}"
            if chip_type_var not in assignment: return True # Tipo de chip no decidido
            return assignment[chip_type_var] not in ["TypeC", "TypeD"]
        return True

    hierarchy.add_global_constraint(
        ["gate_assign_G2", "gate_assign_G3"] + [f"chip_type_{c}" for c in chips],
        g2_g3_not_same_chip_typeCD_hard_predicate,
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "G2G3_NotSameChip_TypeCD"}
    )

    # HARD: G4 y G5 deben estar en chips diferentes.
    def g4_g5_different_chips_hard_predicate(assignment: Dict[str, Any]) -> bool:
        if "gate_assign_G4" not in assignment or "gate_assign_G5" not in assignment: return True
        return assignment["gate_assign_G4"] != assignment["gate_assign_G5"]

    hierarchy.add_global_constraint(
        ["gate_assign_G4", "gate_assign_G5"],
        g4_g5_different_chips_hard_predicate,
        weight=1.0,
        hardness=Hardness.HARD,
        metadata={"name": "G4G5_DifferentChips"}
    )

    # SOFT (GLOBAL): Minimizar el costo total del circuito (chips + compuertas).
    # Costo chip: TypeA=120, TypeB=80, TypeC=40, TypeD=10
    # Costo compuerta: 10 (independiente del tipo, solo para añadir complejidad)
    def total_cost_soft_predicate(assignment: Dict[str, Any]) -> float:
        cost = 0.0
        for chip_name in chips:
            chip_type_var = f"chip_type_{chip_name}"
            if chip_type_var in assignment:
                chip_type = assignment[chip_type_var]
                cost += {"TypeA": 120, "TypeB": 80, "TypeC": 40, "TypeD": 10}.get(chip_type, 0)
        
        cost += len(gates) * 10 # Costo base por compuerta
        return cost

    hierarchy.add_global_constraint(
        variables,
        total_cost_soft_predicate,
        objective="minimize",
        weight=1.0, # Alta importancia
        hardness=Hardness.SOFT,
        metadata={"name": "TotalCircuitCost"}
    )

    # SOFT (PATTERN): Maximizar el rendimiento total del circuito (minimizar penalización por rendimiento).
    # Penalización: TypeA=0.05 por compuerta, TypeB=0.15 por compuerta, TypeC=0.3 por compuerta, TypeD=0.7 por compuerta
    def total_performance_soft_predicate(assignment: Dict[str, Any]) -> float:
        performance_penalty = 0.0
        chip_gate_counts = {chip: 0 for chip in chips}
        
        for gate in gates:
            gate_assign_var = f"gate_assign_{gate}"
            if gate_assign_var in assignment:
                chip_gate_counts[assignment[gate_assign_var]] += 1
        
        for chip_name in chips:
            chip_type_var = f"chip_type_{chip_name}"
            if chip_type_var in assignment:
                chip_type = assignment[chip_type_var]
                num_gates_on_chip = chip_gate_counts[chip_name]
                performance_penalty += num_gates_on_chip * {"TypeA": 0.05, "TypeB": 0.15, "TypeC": 0.3, "TypeD": 0.7}.get(chip_type, 0)
        return performance_penalty

    hierarchy.add_pattern_constraint(
        variables,
        total_performance_soft_predicate,
        pattern_type="performance_optimization",
        weight=0.8, # Importancia media
        hardness=Hardness.SOFT,
        metadata={"name": "TotalCircuitPerformance"}
    )

    # SOFT (LOCAL): Penalizar si G6 y G7 están en el mismo chip y es de tipo 'TypeA'.
    def g6_g7_same_chip_penalty_soft_predicate(assignment: Dict[str, Any]) -> float:
        if "gate_assign_G6" in assignment and "gate_assign_G7" in assignment:
            if assignment["gate_assign_G6"] == assignment["gate_assign_G7"]:
                chip_assigned = assignment["gate_assign_G6"]
                chip_type_var = f"chip_type_{chip_assigned}"
                if chip_type_var in assignment and assignment[chip_type_var] == "TypeA":
                    return 1.0 # Penalización alta
        return 0.0

    hierarchy.add_pattern_constraint(
        ["gate_assign_G6", "gate_assign_G7"] + [f"chip_type_{c}" for c in chips],
        g6_g7_same_chip_penalty_soft_predicate,
        pattern_type="g6g7_preference",
        weight=0.3, # Baja importancia
        hardness=Hardness.SOFT,
        metadata={"name": "G6G7_SameChipPenalty"}
    )

    # SOFT (GLOBAL): Minimizar el consumo total de potencia del circuito.
    # Potencia: TypeA=6, TypeB=4, TypeC=2, TypeD=1
    def total_power_soft_predicate(assignment: Dict[str, Any]) -> float:
        power_consumption = 0.0
        for chip_name in chips:
            chip_type_var = f"chip_type_{chip_name}"
            if chip_type_var in assignment:
                chip_type = assignment[chip_type_var]
                power_consumption += {"TypeA": 6, "TypeB": 4, "TypeC": 2, "TypeD": 1}.get(chip_type, 0)
        return power_consumption

    hierarchy.add_global_constraint(
        variables,
        total_power_soft_predicate,
        objective="minimize",
        weight=0.6, # Importancia media
        hardness=Hardness.SOFT,
        metadata={"name": "TotalPowerConsumption"}
    )

    # SOFT (GLOBAL): Penalizar la dispersión de compuertas (preferir menos chips usados).
    def chip_dispersion_soft_predicate(assignment: Dict[str, Any]) -> float:
        used_chips = set()
        for gate in gates:
            gate_assign_var = f"gate_assign_{gate}"
            if gate_assign_var in assignment:
                used_chips.add(assignment[gate_assign_var])
        
        # Penalización: 0 si todos los chips están usados, 1 por cada chip no usado
        return len(chips) - len(used_chips)

    hierarchy.add_global_constraint(
        variables,
        chip_dispersion_soft_predicate,
        objective="minimize",
        weight=0.5, # Importancia media
        hardness=Hardness.SOFT,
        metadata={"name": "ChipDispersionPenalty"}
    )

    return variables, domains, hierarchy

if __name__ == "__main__":
    vars, doms, hier = create_circuit_design_problem()
    print(f"Variables: {len(vars)}")
    print(f"Dominios: {doms}")
    print(f"Restricciones HARD: {len([c for c in hier.constraints if c.hardness == Hardness.HARD])}")
    print(f"Restricciones SOFT: {len([c for c in hier.constraints if c.hardness == Hardness.SOFT])}")

