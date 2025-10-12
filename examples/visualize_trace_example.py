"""
Ejemplo de uso del SearchSpaceVisualizer.

Este ejemplo demuestra cómo generar visualizaciones y reportes
a partir de un trace de búsqueda.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import sys
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.arc_weaver.graph_structures import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer
from lattice_weaver.visualization import (
    load_trace,
    plot_search_tree,
    plot_domain_evolution,
    plot_backtrack_heatmap,
    generate_report
)


def create_nqueens_problem(n):
    """Crea un problema de N-Reinas."""
    cg = ConstraintGraph()
    
    for i in range(n):
        cg.add_variable(f'Q{i}', set(range(n)))
    
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(row_diff):
                def constraint(vi, vj):
                    if vi == vj:
                        return False
                    if abs(vi - vj) == row_diff:
                        return False
                    return True
                return constraint
            
            cg.add_constraint(f'Q{i}', f'Q{j}', make_constraint(j - i))
    
    return cg


def main():
    """Función principal."""
    n = 6
    
    print(f"Resolviendo {n}-Reinas y generando visualizaciones...")
    print()
    
    # Crear problema
    problem = create_nqueens_problem(n)
    
    # Resolver con tracing
    trace_path = f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_trace.csv'
    
    with SearchSpaceTracer(enabled=True, output_path=trace_path, async_mode=True) as tracer:
        engine = AdaptiveConsistencyEngine(tracer=tracer)
        stats = engine.solve(problem, max_solutions=2)
        
        print(f"Soluciones encontradas: {len(stats.solutions)}")
        print(f"Nodos explorados: {stats.nodes_explored}")
        print(f"Backtracks: {stats.backtracks}")
        print(f"Tiempo: {stats.time_elapsed:.4f}s")
        print()
    
    # Cargar trace
    print("Cargando trace...")
    df = load_trace(trace_path)
    print(f"Eventos cargados: {len(df)}")
    print()
    
    # Generar visualizaciones individuales
    print("Generando visualizaciones...")
    
    # Árbol de búsqueda
    fig_tree = plot_search_tree(df)
    tree_path = f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_tree.html'
    fig_tree.write_html(tree_path)
    print(f"  ✓ Árbol de búsqueda: {tree_path}")
    
    # Evolución de dominios
    fig_domain = plot_domain_evolution(df)
    domain_path = f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_domain.html'
    fig_domain.write_html(domain_path)
    print(f"  ✓ Evolución de dominios: {domain_path}")
    
    # Heatmap de backtracks
    fig_heatmap = plot_backtrack_heatmap(df)
    heatmap_path = f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_heatmap.html'
    fig_heatmap.write_html(heatmap_path)
    print(f"  ✓ Heatmap de backtracks: {heatmap_path}")
    
    print()
    
    # Generar reporte completo
    print("Generando reporte completo...")
    report_path = f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_report.html'
    generate_report(df, report_path, title=f"Reporte de {n}-Reinas")
    print(f"  ✓ Reporte completo: {report_path}")
    print()
    
    print("¡Visualizaciones generadas exitosamente!")
    print()
    print("Archivos generados:")
    print(f"  - {trace_path}")
    print(f"  - {tree_path}")
    print(f"  - {domain_path}")
    print(f"  - {heatmap_path}")
    print(f"  - {report_path}")


if __name__ == '__main__':
    main()

