"""
Ejemplo de análisis avanzado de resultados de experimentos.

Este ejemplo demuestra cómo usar las funciones de análisis avanzado
para procesar y visualizar resultados de experimentos.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import sys
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.core.csp_engine.graph import ConstraintGraph
from lattice_weaver.benchmarks import (
    ExperimentRunner,
    ExperimentConfig,
    compute_statistics_with_confidence,
    detect_outliers,
    generate_detailed_report,
    export_results_to_csv,
    export_summary_to_markdown
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
    print("=== Ejemplo de Análisis Avanzado ===")
    print()
    
    # 1. Ejecutar experimentos
    print("1. Ejecutando experimentos...")
    
    runner = ExperimentRunner(output_dir="/home/ubuntu/lattice-weaver/examples/experiments")
    
    # Añadir varias configuraciones
    for n in [4, 6]:
        for run_idx in range(3):
            config = ExperimentConfig(
                name=f"nqueens_{n}",
                problem_generator=create_nqueens_problem,
                problem_params={"n": n},
                solver_params={"max_solutions": 1},
                num_runs=5,
                enable_tracing=False
            )
            runner.add_config(config)
    
    results = runner.run_all(parallel=True, max_workers=4)
    print(f"  ✓ {len(results)} ejecuciones completadas")
    print()
    
    # 2. Análisis estadístico con intervalos de confianza
    print("2. Análisis estadístico con IC 95%...")
    
    df = runner.to_dataframe()
    stats = compute_statistics_with_confidence(df, confidence_level=0.95)
    
    print("  Tiempo de ejecución:")
    print(f"    Media: {stats['time']['mean']:.4f}s")
    print(f"    IC 95%: [{stats['time']['confidence_interval'][0]:.4f}, {stats['time']['confidence_interval'][1]:.4f}]")
    print()
    
    print("  Nodos explorados:")
    print(f"    Media: {stats['nodes']['mean']:.1f}")
    print(f"    IC 95%: [{stats['nodes']['confidence_interval'][0]:.1f}, {stats['nodes']['confidence_interval'][1]:.1f}]")
    print()
    
    # 3. Detección de outliers
    print("3. Detección de outliers...")
    
    time_outliers = detect_outliers(df, 'time_elapsed')
    nodes_outliers = detect_outliers(df, 'nodes_explored')
    
    print(f"  Outliers en tiempo: {len(time_outliers)}")
    print(f"  Outliers en nodos: {len(nodes_outliers)}")
    print()
    
    # 4. Generar reporte detallado
    print("4. Generando reporte detallado...")
    
    generate_detailed_report(
        df,
        "/home/ubuntu/lattice-weaver/examples/detailed_report.html",
        title="Análisis Detallado de N-Reinas"
    )
    print()
    
    # 5. Exportar a CSV
    print("5. Exportando resultados a CSV...")
    
    export_results_to_csv(
        df,
        "/home/ubuntu/lattice-weaver/examples/results.csv"
    )
    print()
    
    # 6. Exportar resumen a Markdown
    print("6. Exportando resumen a Markdown...")
    
    export_summary_to_markdown(
        df,
        "/home/ubuntu/lattice-weaver/examples/summary.md",
        title="Resumen de Experimentos N-Reinas"
    )
    print()
    
    print("=== Análisis completado ===")
    print()
    print("Archivos generados:")
    print("  - /home/ubuntu/lattice-weaver/examples/detailed_report.html")
    print("  - /home/ubuntu/lattice-weaver/examples/results.csv")
    print("  - /home/ubuntu/lattice-weaver/examples/summary.md")


if __name__ == '__main__':
    main()

