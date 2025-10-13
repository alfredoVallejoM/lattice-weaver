"""
Ejemplo de uso del ExperimentRunner.

Este ejemplo demuestra cómo ejecutar experimentos masivos con diferentes
configuraciones y analizar los resultados.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
"""

import sys
sys.path.insert(0, '/home/ubuntu/lattice-weaver')

from lattice_weaver.core.csp_engine.graph import ConstraintGraph
from lattice_weaver.benchmarks import ExperimentRunner, ExperimentConfig


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
    print("=== Ejemplo de ExperimentRunner ===")
    print()
    
    # Crear runner
    runner = ExperimentRunner(output_dir="/home/ubuntu/lattice-weaver/examples/experiments")
    
    # Método 1: Añadir configuraciones manualmente
    print("1. Añadiendo configuraciones de experimentos...")
    
    config1 = ExperimentConfig(
        name="nqueens_4_manual",
        problem_generator=create_nqueens_problem,
        problem_params={"n": 4},
        solver_params={"max_solutions": 1},
        num_runs=5,
        enable_tracing=True,
        trace_output_dir="/home/ubuntu/lattice-weaver/examples/experiments/traces/nqueens_4"
    )
    runner.add_config(config1)
    
    config2 = ExperimentConfig(
        name="nqueens_6_manual",
        problem_generator=create_nqueens_problem,
        problem_params={"n": 6},
        solver_params={"max_solutions": 1},
        num_runs=3,
        enable_tracing=True,
        trace_output_dir="/home/ubuntu/lattice-weaver/examples/experiments/traces/nqueens_6"
    )
    runner.add_config(config2)
    
    print(f"  ✓ {len(runner.configs)} configuraciones añadidas")
    print()
    
    # Ejecutar experimentos
    print("2. Ejecutando experimentos en paralelo...")
    print()
    
    results = runner.run_all(parallel=True, max_workers=2)
    
    print()
    print(f"Total de ejecuciones: {len(results)}")
    print()
    
    # Obtener estadísticas
    print("3. Estadísticas resumidas:")
    print()
    
    stats = runner.get_summary_statistics()
    
    print(f"  Total de ejecuciones: {stats['total_runs']}")
    print(f"  Exitosas: {stats['successful_runs']}")
    print(f"  Fallidas: {stats['failed_runs']}")
    print(f"  Tiempo promedio: {stats['avg_time']:.4f}s ± {stats['std_time']:.4f}s")
    print(f"  Nodos promedio: {stats['avg_nodes_explored']:.1f} ± {stats['std_nodes_explored']:.1f}")
    print(f"  Backtracks promedio: {stats['avg_backtracks']:.1f} ± {stats['std_backtracks']:.1f}")
    print(f"  Soluciones totales: {stats['total_solutions']}")
    print()
    
    # Guardar resultados
    print("4. Guardando resultados...")
    runner.save_results()
    
    # Convertir a DataFrame
    df = runner.to_dataframe()
    print(f"  ✓ DataFrame con {len(df)} filas")
    print()
    
    # Mostrar resumen por configuración
    print("5. Resumen por configuración:")
    print()
    
    for config_name in df['config_name'].unique():
        config_df = df[df['config_name'] == config_name]
        successful = config_df[config_df['success'] == True]
        
        if len(successful) > 0:
            print(f"  {config_name}:")
            print(f"    Ejecuciones: {len(config_df)}")
            print(f"    Tiempo: {successful['time_elapsed'].mean():.4f}s ± {successful['time_elapsed'].std():.4f}s")
            print(f"    Nodos: {successful['nodes_explored'].mean():.1f} ± {successful['nodes_explored'].std():.1f}")
            print()
    
    print("=== Ejemplo completado ===")


if __name__ == '__main__':
    main()

