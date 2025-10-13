"""
Ejemplo de uso avanzado del SearchSpaceVisualizer.

Este ejemplo demuestra las funcionalidades avanzadas:
- Visualizaciones adicionales (timeline, estadísticas por variable)
- Comparación de múltiples traces
- Exportación a múltiples formatos
- Reporte avanzado

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import sys
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.core.csp_engine.graph import ConstraintGraph
from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer
from lattice_weaver.visualization import (
    load_trace,
    plot_timeline,
    plot_variable_statistics,
    compare_traces,
    export_visualizations,
    generate_advanced_report
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
                    if vi == vj or abs(vi - vj) == row_diff:
                        return False
                    return True
                return constraint
            
            cg.add_constraint(f'Q{i}', f'Q{j}', make_constraint(j - i))
    
    return cg


def main():
    """Función principal."""
    print("=== Ejemplo Avanzado de Visualización ===")
    print()
    
    # 1. Generar múltiples traces con diferentes tamaños
    print("1. Generando traces de diferentes tamaños...")
    traces = {}
    
    for n in [4, 6]:
        problem = create_nqueens_problem(n)
        trace_path = f'/home/ubuntu/lattice-weaver/examples/nqueens_{n}_trace.csv'
        
        with SearchSpaceTracer(enabled=True, output_path=trace_path, async_mode=True) as tracer:
            engine = AdaptiveConsistencyEngine(tracer=tracer)
            stats = engine.solve(problem, max_solutions=1)
            print(f"  ✓ {n}-Reinas: {stats.nodes_explored} nodos, {stats.backtracks} backtracks")
        
        traces[f'{n}-Reinas'] = load_trace(trace_path)
    
    print()
    
    # 2. Visualizaciones adicionales
    print("2. Generando visualizaciones adicionales...")
    
    df = traces['6-Reinas']
    
    # Timeline
    fig_timeline = plot_timeline(df)
    timeline_path = '/home/ubuntu/lattice-weaver/examples/timeline.html'
    fig_timeline.write_html(timeline_path)
    print(f"  ✓ Timeline: {timeline_path}")
    
    # Estadísticas por variable
    fig_var_stats = plot_variable_statistics(df)
    var_stats_path = '/home/ubuntu/lattice-weaver/examples/variable_stats.html'
    fig_var_stats.write_html(var_stats_path)
    print(f"  ✓ Estadísticas por variable: {var_stats_path}")
    
    print()
    
    # 3. Comparación de traces
    print("3. Generando comparación de traces...")
    
    fig_comparison = compare_traces(traces, metric='nodes_explored')
    comparison_path = '/home/ubuntu/lattice-weaver/examples/comparison.html'
    fig_comparison.write_html(comparison_path)
    print(f"  ✓ Comparación: {comparison_path}")
    
    print()
    
    # 4. Exportación a múltiples formatos
    print("4. Exportando visualizaciones a múltiples formatos...")
    
    # Nota: PNG/PDF requieren kaleido, que puede no estar disponible
    # En ese caso, solo exportamos HTML
    try:
        files = export_visualizations(
            df,
            '/home/ubuntu/lattice-weaver/examples/exports/',
            formats=['html']  # Cambiar a ['html', 'png'] si kaleido está disponible
        )
        print(f"  ✓ Exportadas {len(files)} visualizaciones")
    except Exception as e:
        print(f"  ⚠ Error en exportación: {e}")
        print("  (Esto es normal si kaleido no está instalado)")
    
    print()
    
    # 5. Reporte avanzado
    print("5. Generando reporte avanzado...")
    
    advanced_report_path = '/home/ubuntu/lattice-weaver/examples/advanced_report.html'
    generate_advanced_report(
        df,
        advanced_report_path,
        title="Reporte Avanzado - 6-Reinas",
        include_timeline=True,
        include_variable_stats=True
    )
    
    print()
    print("=== Ejemplo completado exitosamente ===")
    print()
    print("Archivos generados:")
    print(f"  - {timeline_path}")
    print(f"  - {var_stats_path}")
    print(f"  - {comparison_path}")
    print(f"  - {advanced_report_path}")
    print(f"  - /home/ubuntu/lattice-weaver/examples/exports/ (múltiples archivos)")


if __name__ == '__main__':
    main()

