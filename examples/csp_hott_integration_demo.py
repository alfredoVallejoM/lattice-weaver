#!/usr/bin/env python3
# csp_hott_integration_demo.py

"""
Demostración de Integración Completa CSP-HoTT

Ejemplos de traducción de problemas CSP a tipos HoTT y
conversión de soluciones a pruebas formales.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lattice_weaver.formal import (
    CSPProblem, CSPSolution,
    create_extended_bridge
)


def demo_n_queens():
    """Demostración: Problema de N-Reinas (N=4)."""
    print("=" * 70)
    print("DEMO: Problema de 4-Reinas")
    print("=" * 70)
    
    # 4 reinas en tablero 4x4
    # Variables: fila de cada reina (columnas fijas)
    problem = CSPProblem(
        variables=['Q1', 'Q2', 'Q3', 'Q4'],
        domains={
            'Q1': {1, 2, 3, 4},
            'Q2': {1, 2, 3, 4},
            'Q3': {1, 2, 3, 4},
            'Q4': {1, 2, 3, 4}
        },
        constraints=[
            # No en misma fila
            ('Q1', 'Q2', lambda a, b: a != b),
            ('Q1', 'Q3', lambda a, b: a != b),
            ('Q1', 'Q4', lambda a, b: a != b),
            ('Q2', 'Q3', lambda a, b: a != b),
            ('Q2', 'Q4', lambda a, b: a != b),
            ('Q3', 'Q4', lambda a, b: a != b),
            
            # No en misma diagonal
            ('Q1', 'Q2', lambda a, b: abs(a - b) != abs(1 - 2)),
            ('Q1', 'Q3', lambda a, b: abs(a - b) != abs(1 - 3)),
            ('Q1', 'Q4', lambda a, b: abs(a - b) != abs(1 - 4)),
            ('Q2', 'Q3', lambda a, b: abs(a - b) != abs(2 - 3)),
            ('Q2', 'Q4', lambda a, b: abs(a - b) != abs(2 - 4)),
            ('Q3', 'Q4', lambda a, b: abs(a - b) != abs(3 - 4)),
        ]
    )
    
    print(f"\nProblema:")
    print(f"  Variables: {len(problem.variables)} reinas")
    print(f"  Dominio: Filas 1-4 para cada reina")
    print(f"  Restricciones: {len(problem.constraints)}")
    
    # Traducir a tipo HoTT
    bridge = create_extended_bridge()
    problem_type = bridge.translate_csp_to_type(problem)
    
    print(f"\nTipo HoTT del problema:")
    print(f"  {problem_type}")
    
    # Solución conocida: [2, 4, 1, 3]
    solution = CSPSolution(
        assignment={'Q1': 2, 'Q2': 4, 'Q3': 1, 'Q4': 3},
        is_consistent=True
    )
    
    print(f"\nSolución propuesta:")
    print(f"  Q1 en fila {solution.assignment['Q1']}")
    print(f"  Q2 en fila {solution.assignment['Q2']}")
    print(f"  Q3 en fila {solution.assignment['Q3']}")
    print(f"  Q4 en fila {solution.assignment['Q4']}")
    
    # Convertir a prueba
    proof = bridge.solution_to_proof_complete(solution, problem)
    
    if proof:
        print(f"\n✅ Prueba formal generada exitosamente")
        print(f"  Término de prueba: {proof.term}")
    else:
        print(f"\n❌ Solución inválida")
    
    print("\n" + "=" * 70)


def demo_sudoku_subset():
    """Demostración: Subconjunto de Sudoku (3x3)."""
    print("\n" + "=" * 70)
    print("DEMO: Sudoku 3x3 Simplificado")
    print("=" * 70)
    
    # 9 celdas, valores 1-3
    cells = [f'c{i}' for i in range(1, 10)]
    
    problem = CSPProblem(
        variables=cells,
        domains={cell: {1, 2, 3} for cell in cells},
        constraints=[]
    )
    
    # Restricciones de fila (3 filas)
    for row in range(3):
        for i in range(3):
            for j in range(i+1, 3):
                c1 = f'c{row*3 + i + 1}'
                c2 = f'c{row*3 + j + 1}'
                problem.constraints.append((c1, c2, lambda a, b: a != b))
    
    # Restricciones de columna (3 columnas)
    for col in range(3):
        for i in range(3):
            for j in range(i+1, 3):
                c1 = f'c{i*3 + col + 1}'
                c2 = f'c{j*3 + col + 1}'
                problem.constraints.append((c1, c2, lambda a, b: a != b))
    
    print(f"\nProblema:")
    print(f"  Variables: {len(problem.variables)} celdas")
    print(f"  Dominio: {1, 2, 3}")
    print(f"  Restricciones: {len(problem.constraints)}")
    
    bridge = create_extended_bridge()
    
    # Solución válida (cuadrado latino 3x3)
    solution = CSPSolution(
        assignment={
            'c1': 1, 'c2': 2, 'c3': 3,
            'c4': 2, 'c5': 3, 'c6': 1,
            'c7': 3, 'c8': 1, 'c9': 2
        },
        is_consistent=True
    )
    
    print(f"\nSolución propuesta:")
    print(f"  {solution.assignment['c1']} {solution.assignment['c2']} {solution.assignment['c3']}")
    print(f"  {solution.assignment['c4']} {solution.assignment['c5']} {solution.assignment['c6']}")
    print(f"  {solution.assignment['c7']} {solution.assignment['c8']} {solution.assignment['c9']}")
    
    # Convertir a prueba
    proof = bridge.solution_to_proof_complete(solution, problem)
    
    if proof:
        print(f"\n✅ Prueba formal generada")
    else:
        print(f"\n❌ Solución inválida")
    
    print("\n" + "=" * 70)


def demo_scheduling():
    """Demostración: Problema de Scheduling."""
    print("\n" + "=" * 70)
    print("DEMO: Scheduling de Tareas")
    print("=" * 70)
    
    # 3 tareas, 3 slots de tiempo
    problem = CSPProblem(
        variables=['T1', 'T2', 'T3'],
        domains={
            'T1': {'morning', 'afternoon', 'evening'},
            'T2': {'morning', 'afternoon', 'evening'},
            'T3': {'morning', 'afternoon', 'evening'}
        },
        constraints=[
            # T1 y T2 no pueden ser al mismo tiempo
            ('T1', 'T2', lambda a, b: a != b),
            # T2 debe ser antes que T3
            ('T2', 'T3', lambda a, b: 
                ('morning', 'afternoon', 'evening').index(a) < 
                ('morning', 'afternoon', 'evening').index(b)),
        ]
    )
    
    print(f"\nProblema:")
    print(f"  Tareas: {problem.variables}")
    print(f"  Slots: morning, afternoon, evening")
    print(f"  Restricciones:")
    print(f"    - T1 y T2 en diferentes slots")
    print(f"    - T2 antes que T3")
    
    bridge = create_extended_bridge()
    
    # Traducir a tipo
    problem_type = bridge.translate_csp_to_type(problem)
    print(f"\nTipo HoTT: {problem_type}")
    
    # Solución válida
    solution = CSPSolution(
        assignment={'T1': 'morning', 'T2': 'afternoon', 'T3': 'evening'},
        is_consistent=True
    )
    
    print(f"\nSolución:")
    print(f"  T1: {solution.assignment['T1']}")
    print(f"  T2: {solution.assignment['T2']}")
    print(f"  T3: {solution.assignment['T3']}")
    
    # Convertir a prueba
    proof = bridge.solution_to_proof_complete(solution, problem)
    
    if proof:
        print(f"\n✅ Scheduling válido (prueba formal generada)")
    else:
        print(f"\n❌ Scheduling inválido")
    
    # Solución inválida (T2 después de T3)
    invalid_solution = CSPSolution(
        assignment={'T1': 'morning', 'T2': 'evening', 'T3': 'afternoon'},
        is_consistent=True
    )
    
    print(f"\nSolución inválida (T2 después de T3):")
    proof_invalid = bridge.solution_to_proof_complete(invalid_solution, problem)
    
    if proof_invalid is None:
        print(f"  ✅ Correctamente rechazada")
    else:
        print(f"  ❌ Error: debería ser rechazada")
    
    print("\n" + "=" * 70)


def demo_propositions_extraction():
    """Demostración: Extracción de proposiciones."""
    print("\n" + "=" * 70)
    print("DEMO: Extracción de Proposiciones Lógicas")
    print("=" * 70)
    
    problem = CSPProblem(
        variables=['x', 'y', 'z'],
        domains={'x': {1, 2, 3}, 'y': {1, 2, 3}, 'z': {1, 2, 3}},
        constraints=[
            ('x', 'y', lambda a, b: a < b),
            ('y', 'z', lambda a, b: a < b),
            ('x', 'z', lambda a, b: a < b)
        ]
    )
    
    print(f"\nProblema: x < y < z")
    print(f"  Variables: {problem.variables}")
    print(f"  Restricciones: {len(problem.constraints)}")
    
    bridge = create_extended_bridge()
    
    # Extraer proposiciones
    propositions = bridge.extract_constraints_as_propositions(problem)
    
    print(f"\nProposiciones extraídas:")
    for i, prop in enumerate(propositions, 1):
        print(f"  P{i}: {prop}")
    
    # Estadísticas
    stats = bridge.get_translation_statistics()
    print(f"\nEstadísticas:")
    print(f"  Tipos cacheados: {stats['cached_types']}")
    print(f"  Restricciones cacheadas: {stats['cached_constraints']}")
    
    print("\n" + "=" * 70)


def main():
    """Ejecuta las demostraciones."""
    print("\nDemostraciones de Integración Completa CSP-HoTT")
    print("LatticeWeaver v4\n")
    
    try:
        demo_n_queens()
        demo_sudoku_subset()
        demo_scheduling()
        demo_propositions_extraction()
        
        print("\n✅ Todas las demostraciones completadas exitosamente\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

