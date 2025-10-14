from typing import Dict, List, Any, Tuple
from lattice_weaver.fibration import ConstraintHierarchy, ConstraintLevel, Hardness

def create_network_config_problem() -> Tuple[List[str], Dict[str, List[Any]], ConstraintHierarchy]:
    """
    Crea un problema de configuración de red con múltiples objetivos en conflicto.

    Variables:
    - `router_type_R{i}`: Tipo de router para cada nodo de la red (R0, R1, R2, R3)
    - `link_capacity_L{i}_{j}`: Capacidad del enlace entre nodos (L0_1, L1_2, etc.)

    Dominios:
    - Tipos de router: ["basic", "advanced"]
    - Capacidades de enlace: [100, 500, 1000] Mbps

    Restricciones:
    - HARD: Cada router avanzado debe tener al menos un enlace de 1000 Mbps.
    - SOFT (GLOBAL): Minimizar el costo total de la red (routers + enlaces).
    - SOFT (PATTERN): Maximizar el rendimiento promedio de la red (suma de capacidades de enlaces).
    - SOFT (LOCAL): Preferir routers básicos para nodos con bajo tráfico esperado.
    """
    nodes = [f"R{i}" for i in range(4)] # 4 routers
    links = [("R0", "R1"), ("R1", "R2"), ("R2", "R3"), ("R3", "R0"), ("R0", "R2")] # 5 enlaces

    variables = []
    domains = {}

    # Variables para tipos de router
    for node in nodes:
        var_name = f"router_type_{node}"
        variables.append(var_name)
        domains[var_name] = ["basic", "advanced"]

    # Variables para capacidades de enlace
    for i, (n1, n2) in enumerate(links):
        var_name = f"link_capacity_{n1}_{n2}"
        variables.append(var_name)
        domains[var_name] = [100, 500, 1000]

    hierarchy = ConstraintHierarchy()

    # HARD: Cada router avanzado debe tener al menos un enlace de 1000 Mbps.
    for node in nodes:
        router_type_var = f"router_type_{node}"
        related_links = [f"link_capacity_{n1}_{n2}" for n1, n2 in links if n1 == node or n2 == node]
        
        def advanced_router_link_capacity_hard_predicate(assignment: Dict[str, Any], rt_var: str, rl_vars: List[str]) -> bool:
            if rt_var not in assignment or assignment[rt_var] != "advanced":
                return True # No es un router avanzado, la restricción no aplica
            
            # Si es avanzado, debe tener al menos un enlace de 1000 Mbps
            has_high_capacity_link = False
            for link_var in rl_vars:
                if link_var in assignment and assignment[link_var] == 1000:
                    has_high_capacity_link = True
                    break
            return has_high_capacity_link

        hierarchy.add_global_constraint(
            [router_type_var] + related_links,
            lambda a, rt=router_type_var, rls=related_links: advanced_router_link_capacity_hard_predicate(a, rt, rls),
            weight=1.0,
            hardness=Hardness.HARD,
            metadata={"name": f"AdvancedRouter_{node}_HighCapacityLink"}
        )

    # SOFT (GLOBAL): Minimizar el costo total de la red (routers + enlaces).
    # Costo router: basic=100, advanced=500
    # Costo enlace: 100Mbps=10, 500Mbps=50, 1000Mbps=100
    def total_cost_soft_predicate(assignment: Dict[str, Any]) -> float:
        cost = 0.0
        for var, val in assignment.items():
            if var.startswith("router_type_"):
                cost += 100 if val == "basic" else 500
            elif var.startswith("link_capacity_"):
                if val == 100: cost += 10
                elif val == 500: cost += 50
                elif val == 1000: cost += 100
        return cost

    hierarchy.add_global_constraint(
        variables,
        total_cost_soft_predicate,
        objective="minimize",
        weight=1.0, # Alta importancia
        hardness=Hardness.SOFT,
        metadata={"name": "TotalNetworkCost"}
    )

    # SOFT (PATTERN): Maximizar el rendimiento promedio de la red (suma de capacidades de enlaces).
    # Se penaliza la baja capacidad total de enlaces.
    def average_throughput_soft_predicate(assignment: Dict[str, Any]) -> float:
        total_capacity = 0
        num_links = 0
        for var, val in assignment.items():
            if var.startswith("link_capacity_"):
                total_capacity += val
                num_links += 1
        
        if num_links == 0: return 0.0
        
        avg_capacity = total_capacity / num_links
        # Penalizar baja capacidad. Queremos maximizar, así que minimizamos el negativo.
        # O, penalizamos la diferencia con una capacidad ideal (ej. 1000 Mbps)
        ideal_capacity = 1000.0
        return (ideal_capacity - avg_capacity) / ideal_capacity # Normalizado entre 0 y 1

    hierarchy.add_pattern_constraint(
        [v for v in variables if v.startswith("link_capacity_")],
        average_throughput_soft_predicate,
        pattern_type="throughput_optimization",
        weight=0.8, # Importancia media
        hardness=Hardness.SOFT,
        metadata={"name": "AverageNetworkThroughput"}
    )

    # SOFT (LOCAL): Preferir routers básicos para nodos con bajo tráfico esperado.
    # Asumimos R0 y R3 tienen bajo tráfico esperado.
    def low_traffic_node_preference_soft_predicate(assignment: Dict[str, Any]) -> float:
        cost = 0.0
        if "router_type_R0" in assignment and assignment["router_type_R0"] == "advanced": cost += 0.5
        if "router_type_R3" in assignment and assignment["router_type_R3"] == "advanced": cost += 0.5
        return cost

    hierarchy.add_local_constraint(
        "router_type_R0", "router_type_R3", # No es realmente local, pero afecta a un subconjunto de variables
        low_traffic_node_preference_soft_predicate,
        weight=0.3, # Baja importancia
        hardness=Hardness.SOFT,
        metadata={"name": "LowTrafficNodePreference"}
    )

    return variables, domains, hierarchy

if __name__ == "__main__":
    vars, doms, hier = create_network_config_problem()
    print(f"Variables: {len(vars)}")
    print(f"Dominios: {doms}")
    print(f"Restricciones HARD: {len([c for c in hier.constraints if c.hardness == Hardness.HARD])}")
    print(f"Restricciones SOFT: {len([c for c in hier.constraints if c.hardness == Hardness.SOFT])}")

