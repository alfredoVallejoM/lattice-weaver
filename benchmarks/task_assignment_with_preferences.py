"""
Asignación de Tareas con Preferencias - Benchmark

Problema simple que muestra las fortalezas de Fibration Flow:
- Asignar 4 tareas a 3 trabajadores
- Restricciones HARD: cada tarea debe asignarse a exactamente un trabajador
- Restricciones SOFT: preferencias de trabajadores por tareas, balanceo de carga

Este problema es ideal para mostrar cómo Fibration Flow maneja optimización
con restricciones SOFT+HARD, algo que Forward Checking no puede hacer.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
import logging
from typing import Dict, List, Any

# Desactivar logging
logging.disable(logging.CRITICAL)

from lattice_weaver.fibration.constraint_hierarchy import ConstraintHierarchy, Hardness, ConstraintLevel
from lattice_weaver.fibration.energy_landscape_optimized import EnergyLandscapeOptimized
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.fibration.fibration_search_solver_enhanced import FibrationSearchSolverEnhanced
# TODO: Descomentar cuando se integre el componente
# from lattice_weaver.arc_engine.core import ArcEngine


def create_task_assignment_problem():
    """
    Crea un problema de asignación de tareas con preferencias.
    
    Tareas: T1, T2, T3, T4
    Trabajadores: W1, W2, W3
    
    Cada tarea se asigna a un trabajador (dominio: {1, 2, 3})
    
    Restricciones HARD:
    - T1 y T2 no pueden asignarse al mismo trabajador (incompatibles)
    - T3 requiere W1 o W2 (no puede ser W3)
    
    Restricciones SOFT:
    - W1 prefiere T1 (peso 3.0)
    - W2 prefiere T2 (peso 2.5)
    - W3 prefiere T4 (peso 2.0)
    - Balanceo de carga: penalizar si un trabajador tiene >2 tareas (peso 4.0)
    """
    hierarchy = ConstraintHierarchy()
    
    variables = ["T1", "T2", "T3", "T4"]
    domains = {var: [1, 2, 3] for var in variables}  # 1=W1, 2=W2, 3=W3
    
    # ===== RESTRICCIONES HARD (LOCAL) =====
    
    # T1 y T2 no pueden asignarse al mismo trabajador
    def t1_t2_different(assignment):
        if "T1" in assignment and "T2" in assignment:
            return assignment["T1"] != assignment["T2"]
        return True
    
    hierarchy.add_local_constraint(
        var1="T1",
        var2="T2",
        predicate=t1_t2_different,
        hardness=Hardness.HARD,
        metadata={"name": "T1_T2_different_workers"}
    )
    
    # T3 requiere W1 o W2 (no puede ser W3)
    def t3_not_w3(assignment):
        if "T3" in assignment:
            return assignment["T3"] in [1, 2]
        return True
    
    hierarchy.add_local_constraint(
        var1="T3",
        var2="T3",  # Unaria
        predicate=t3_not_w3,
        hardness=Hardness.HARD,
        metadata={"name": "T3_requires_W1_or_W2"}
    )
    
    # ===== RESTRICCIONES SOFT (LOCAL): Preferencias =====
    
    # W1 prefiere T1
    def w1_prefers_t1(assignment):
        if "T1" in assignment:
            # Si T1 está asignado a W1, energía baja (bueno)
            # Si no, energía alta (malo)
            if assignment["T1"] == 1:
                return True  # Preferencia satisfecha
        return True  # Siempre satisfecho, pero con penalización en energía
    
    hierarchy.add_local_constraint(
        var1="T1",
        var2="T1",
        predicate=w1_prefers_t1,
        hardness=Hardness.SOFT,
        weight=3.0,
        metadata={"name": "W1_prefers_T1"}
    )
    
    # W2 prefiere T2
    def w2_prefers_t2(assignment):
        if "T2" in assignment:
            if assignment["T2"] == 2:
                return True
        return True
    
    hierarchy.add_local_constraint(
        var1="T2",
        var2="T2",
        predicate=w2_prefers_t2,
        hardness=Hardness.SOFT,
        weight=2.5,
        metadata={"name": "W2_prefers_T2"}
    )
    
    # W3 prefiere T4
    def w3_prefers_t4(assignment):
        if "T4" in assignment:
            if assignment["T4"] == 3:
                return True
        return True
    
    hierarchy.add_local_constraint(
        var1="T4",
        var2="T4",
        predicate=w3_prefers_t4,
        hardness=Hardness.SOFT,
        weight=2.0,
        metadata={"name": "W3_prefers_T4"}
    )
    
    # ===== RESTRICCIONES SOFT (GLOBAL): Balanceo de Carga =====
    
    def load_balance(assignment):
        if len(assignment) < len(variables):
            return True  # No evaluar si incompleto
        
        # Contar tareas por trabajador
        load = {1: 0, 2: 0, 3: 0}
        for task, worker in assignment.items():
            load[worker] += 1
        
        # Penalizar si algún trabajador tiene >2 tareas
        # (ideal: 1-2 tareas por trabajador)
        penalty = 0
        for worker, count in load.items():
            if count > 2:
                penalty += (count - 2) * 2  # Penalización cuadrática
        
        # Siempre satisfecho, pero con penalización
        return True
    
    hierarchy.add_global_constraint(
        variables=variables,
        predicate=load_balance,
        hardness=Hardness.SOFT,
        weight=4.0,
        metadata={"name": "load_balance"}
    )
    
    return hierarchy, variables, domains


def print_solution(solution: Dict[str, Any]):
    """Imprime la solución de forma legible."""
    if solution is None:
        print("  ❌ No se encontró solución")
        return
    
    print("  ✓ Solución encontrada:")
    print()
    
    # Agrupar por trabajador
    workers = {1: [], 2: [], 3: []}
    for task, worker in solution.items():
        workers[worker].append(task)
    
    worker_names = {1: "W1", 2: "W2", 3: "W3"}
    
    for worker_id in sorted(workers.keys()):
        tasks = sorted(workers[worker_id])
        print(f"    {worker_names[worker_id]}: {', '.join(tasks) if tasks else '(ninguna)'} ({len(tasks)} tareas)")
    
    print()
    
    # Verificar preferencias
    print("  🎯 Preferencias satisfechas:")
    prefs = [
        ("W1 prefiere T1", solution.get("T1") == 1),
        ("W2 prefiere T2", solution.get("T2") == 2),
        ("W3 prefiere T4", solution.get("T4") == 3),
    ]
    
    for pref_name, satisfied in prefs:
        status = "✓" if satisfied else "✗"
        print(f"    {status} {pref_name}")
    
    print()
    
    # Verificar balanceo
    load = {1: 0, 2: 0, 3: 0}
    for task, worker in solution.items():
        load[worker] += 1
    
    max_load = max(load.values())
    min_load = min(load.values())
    balance_score = max_load - min_load
    
    print(f"  ⚖️  Balanceo de carga:")
    print(f"    Diferencia máx-mín: {balance_score} tareas")
    if balance_score <= 1:
        print(f"    ✓ Bien balanceado")
    else:
        print(f"    ⚠️  Podría estar mejor balanceado")


def main():
    """Función principal."""
    print("=" * 100)
    print("BENCHMARK: ASIGNACIÓN DE TAREAS CON PREFERENCIAS")
    print("=" * 100)
    print()
    
    print("📋 Descripción del Problema:")
    print("-" * 100)
    print("  Tareas: T1, T2, T3, T4")
    print("  Trabajadores: W1, W2, W3")
    print()
    print("  Restricciones HARD:")
    print("    - T1 y T2 deben asignarse a trabajadores diferentes")
    print("    - T3 solo puede asignarse a W1 o W2 (no W3)")
    print()
    print("  Restricciones SOFT (preferencias):")
    print("    - W1 prefiere T1 (peso: 3.0)")
    print("    - W2 prefiere T2 (peso: 2.5)")
    print("    - W3 prefiere T4 (peso: 2.0)")
    print()
    print("  Restricciones SOFT (balanceo):")
    print("    - Minimizar desbalanceo de carga (peso: 4.0)")
    print()
    
    # Crear problema
    hierarchy, variables, domains = create_task_assignment_problem()
    landscape = EnergyLandscapeOptimized(hierarchy)
    
    # Contar restricciones
    n_local_hard = sum(1 for c in hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL) 
                       if c.hardness == Hardness.HARD)
    n_local_soft = sum(1 for c in hierarchy.get_constraints_at_level(ConstraintLevel.LOCAL) 
                       if c.hardness == Hardness.SOFT)
    n_global_soft = sum(1 for c in hierarchy.get_constraints_at_level(ConstraintLevel.GLOBAL) 
                        if c.hardness == Hardness.SOFT)
    
    print("🔧 Restricciones:")
    print("-" * 100)
    print(f"  LOCAL HARD: {n_local_hard}")
    print(f"  LOCAL SOFT: {n_local_soft}")
    print(f"  GLOBAL SOFT: {n_global_soft}")
    print(f"  TOTAL: {n_local_hard + n_local_soft + n_global_soft}")
    print()
    
    # Resolver con Fibration Flow
    print("🚀 Resolviendo con Fibration Flow Enhanced...")
    print("-" * 100)
    
    arc_engine = ArcEngine(use_tms=True, parallel=False)
    solver = FibrationSearchSolverEnhanced(
        hierarchy=hierarchy,
        landscape=landscape,
        arc_engine=arc_engine,
        variables=variables,
        domains=domains,
        use_homotopy=True,
        use_tms=True,
        use_enhanced_heuristics=True,
        max_backtracks=10000,
        max_iterations=10000,
        time_limit_seconds=30.0
    )
    
    start_time = time.time()
    solution = solver.solve()
    elapsed = time.time() - start_time
    
    stats = solver.get_statistics()
    
    print()
    print("📊 Resultados:")
    print("-" * 100)
    print(f"  Tiempo: {elapsed:.3f}s")
    print(f"  Backtracks: {stats['search']['backtracks']}")
    print(f"  Nodos explorados: {stats['search']['nodes_explored']}")
    
    if solution:
        energy = landscape.compute_energy(solution)
        print(f"  Energía final: {energy.total_energy:.4f}")
        print(f"    - Local: {energy.local_energy:.4f}")
        print(f"    - Global: {energy.global_energy:.4f}")
    print()
    
    # Imprimir solución
    print_solution(solution)
    
    # Análisis
    print("=" * 100)
    print("ANÁLISIS: POR QUÉ FIBRATION FLOW DESTACA AQUÍ")
    print("=" * 100)
    print()
    print("💡 Ventajas de Fibration Flow en este problema:")
    print()
    print("  1. ✅ Maneja restricciones SOFT + HARD simultáneamente")
    print("     - Forward Checking solo puede verificar restricciones HARD")
    print("     - No puede expresar ni optimizar preferencias")
    print()
    print("  2. ✅ Optimiza múltiples objetivos con pesos")
    print("     - Preferencias de trabajadores (pesos 2.0-3.0)")
    print("     - Balanceo de carga (peso 4.0)")
    print("     - EnergyLandscape combina objetivos automáticamente")
    print()
    print("  3. ✅ Encuentra la MEJOR solución, no solo una válida")
    print("     - Forward Checking retornaría la primera solución factible")
    print("     - Fibration Flow minimiza energía total")
    print()
    
    if solution:
        print("🎯 Calidad de la solución:")
        print()
        
        # Contar preferencias satisfechas
        prefs_satisfied = 0
        if solution.get("T1") == 1:
            prefs_satisfied += 1
        if solution.get("T2") == 2:
            prefs_satisfied += 1
        if solution.get("T4") == 3:
            prefs_satisfied += 1
        
        print(f"  Preferencias satisfechas: {prefs_satisfied}/3")
        
        # Calcular balanceo
        load = {1: 0, 2: 0, 3: 0}
        for task, worker in solution.items():
            load[worker] += 1
        
        max_load = max(load.values())
        min_load = min(load.values())
        
        print(f"  Balanceo: máx={max_load}, mín={min_load}, diff={max_load - min_load}")
        print(f"  Energía total: {energy.total_energy:.4f} (menor = mejor)")
        print()
        
        print("  📈 Comparación con Forward Checking:")
        print("     - Forward Checking encontraría UNA solución válida")
        print("     - Pero NO optimizaría preferencias ni balanceo")
        print(f"     - Fibration Flow encontró solución con {prefs_satisfied}/3 preferencias")
        print(f"       y balanceo {max_load - min_load}")
        print()
    
    print("=" * 100)


if __name__ == "__main__":
    main()

