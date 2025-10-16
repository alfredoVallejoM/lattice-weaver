from typing import Dict, List, Set, Any, Callable, Tuple
import time

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.arc_engine.core import ArcEngine
from lattice_weaver.formal.csp_cubical_bridge import CSPToCubicalBridge

def create_simple_csp_bridge(
    variables: List[str],
    domains: Dict[str, Set[Any]],
    constraints: List[Tuple[str, str, Callable]]
) -> CSPToCubicalBridge:
    """
    Crea un bridge desde una especificación simple de CSP.
    
    Args:
        variables: Lista de nombres de variables
        domains: Mapa variable → dominio
        constraints: Lista de (var1, var2, relation)
    
    Returns:
        Bridge CSP-Cubical
    """
    # Crear el motor ArcEngine
    engine = ArcEngine()

    # Añadir variables y dominios
    for var, domain_values in domains.items():
        engine.add_variable(var, domain_values)

    # Añadir restricciones
    for i, (var1, var2, relation_func) in enumerate(constraints):
        cid = f"C{i}_{var1}_{var2}_{int(time.time())}"
        relation_name = f"R{i}_{var1}_{var2}"
        engine.register_relation(relation_name, relation_func)
        engine.add_constraint(var1, var2, relation_name, cid=cid)

    # Crear el objeto CSP a partir del ArcEngine
    csp_problem = CSP(
        variables=frozenset(engine.variables.keys()),
        domains={var: frozenset(engine.variables[var].get_values()) for var in engine.variables.keys()},
        constraints=frozenset({
            Constraint(
                scope=frozenset({c.var1, c.var2}),
                relation=engine._relation_registry[c.relation_name],
                name=c.cid
            ) for c in engine.constraints.values()
        }),
        name="SimpleCSP"
    )

    # Crear bridge
    return CSPToCubicalBridge(csp_problem=csp_problem)

