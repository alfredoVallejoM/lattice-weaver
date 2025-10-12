"""
Ejemplo de uso del SearchSpaceTracer con N-Reinas.

Este ejemplo demuestra cómo usar el tracer para capturar
la evolución del espacio de búsqueda durante la resolución.
"""

import sys
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer


def create_nqueens_problem(n):
    """Crea un problema de N-Reinas."""
    cg = ConstraintGraph()
    
    # Añadir variables (una por fila)
    for i in range(n):
        cg.add_variable(f'Q{i}', set(range(n)))
    
    # Añadir restricciones (no atacarse)
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(row_diff):
                def constraint(vi, vj):
                    # No misma columna
                    if vi == vj:
                        return False
                    # No misma diagonal
                    if abs(vi - vj) == row_diff:
                        return False
                    return True
                return constraint
            
            cg.add_constraint(f'Q{i}', f'Q{j}', make_constraint(j - i))
    
    return cg


def main():
    """Función principal."""
    n = 4
    
    print(f"Resolviendo {n}-Reinas con tracing habilitado...")
    print()
    
    # Crear problema
    problem = create_nqueens_problem(n)
    
    # Crear tracer
    tracer = SearchSpaceTracer(
        enabled=True,
        output_path=f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_trace.csv',
        output_format='csv'
    )
    
    # Crear solver con tracer
    engine = AdaptiveConsistencyEngine(tracer=tracer)
    
    # Resolver
    stats = engine.solve(problem, max_solutions=2)
    
    print(f"Soluciones encontradas: {len(stats.solutions)}")
    print(f"Nodos explorados: {stats.nodes_explored}")
    print(f"Backtracks: {stats.backtracks}")
    print(f"Tiempo: {stats.time_elapsed:.4f}s")
    print()
    
    # Estadísticas del tracer
    tracer_stats = tracer.get_statistics()
    print("Estadísticas del tracer:")
    print(f"  Total de eventos: {tracer_stats['total_events']}")
    print(f"  Nodos explorados: {tracer_stats['nodes_explored']}")
    print(f"  Backtracks: {tracer_stats['backtracks']}")
    print(f"  Tasa de backtrack: {tracer_stats['backtrack_rate']:.2%}")
    print(f"  Profundidad máxima: {tracer_stats['max_depth']}")
    print(f"  Eventos/segundo: {tracer_stats['events_per_second']:.0f}")
    print()
    
    print(f"Trace guardado en: nqueens_{n}_trace.csv")
    
    # Mostrar primeras soluciones
    for i, sol in enumerate(stats.solutions[:2], 1):
        print(f"\nSolución {i}:")
        board = ['.' * n for _ in range(n)]
        for var, col in sol.items():
            row = int(var[1:])
            board[row] = '.' * col + 'Q' + '.' * (n - col - 1)
        for row in board:
            print(f"  {row}")


if __name__ == '__main__':
    main()
