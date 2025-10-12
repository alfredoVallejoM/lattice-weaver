import argparse
import json
import os
import pandas as pd
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def load_results(input_dir: str) -> List[Dict[str, Any]]:
    """
    Carga todos los archivos JSON de resultados desde un directorio.
    """
    all_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                all_data.extend(data)
    return all_data

def generate_markdown_report(df: pd.DataFrame, output_path: str):
    """
    Genera un reporte en formato Markdown a partir de los resultados.
    """
    with open(output_path, "w") as f:
        f.write("# Reporte de Benchmarking CSP\n\n")
        f.write("Este reporte presenta los resultados de los experimentos de benchmarking realizados con LatticeWeaver.\n\n")
        
        # Resumen general
        f.write("## Resumen General\n\n")
        f.write(f"- **Número total de experimentos:** {len(df)}\n")
        f.write(f"- **Familias de problemas testeadas:** {df['problem_family'].nunique()}\n")
        f.write(f"- **Solvers utilizados:** {df['solver_name'].nunique()}\n\n")

        # Agregación por familia de problema y solver
        f.write("## Resultados Agregados por Problema y Solver\n\n")
        aggregated_df = df.groupby(["problem_family", "solver_name"]).agg({
            "time_taken": ["mean", "std"],
            "solutions_found": "mean",
            "nodes_explored": "mean",
            "backtracks": "mean",
            "constraints_checked": "mean",
            "solution_valid": lambda x: (x == True).sum() / len(x) * 100 # Porcentaje de soluciones válidas
        }).reset_index()
        aggregated_df.columns = ["_" if col == "" else "_" if col[1] == "" else "_" if col[0] == "solution_valid" else col[0] + "_" + col[1] for col in aggregated_df.columns.values]
        aggregated_df.rename(columns={
            "problem_family_": "Problem Family",
            "solver_name_": "Solver",
            "time_taken_mean": "Tiempo Promedio (s)",
            "time_taken_std": "Desviación Estándar Tiempo (s)",
            "solutions_found_mean": "Soluciones Encontradas",
            "nodes_explored_mean": "Nodos Explorados",

            "backtracks_mean": "Backtracks",
            "constraints_checked_mean": "Restricciones Chequeadas",
            "solution_valid_lambda": "% Soluciones Válidas"
        }, inplace=True)
        
        f.write(aggregated_df.to_markdown(index=False))
        f.write("\n\n")

        # Detalles de cada experimento
        f.write("## Detalles de Experimentos Individuales\n\n")
        f.write(df[[
            "problem_family", "solver_name", "time_taken", "solutions_found",
            "nodes_explored", "backtracks", "constraints_checked", "solution_valid"
        ]].to_markdown(index=False))
        f.write("\n\n")

        f.write("\n---\n\n*Generado automáticamente por LatticeWeaver Benchmark Report Generator*\n")
    logger.info(f"Reporte Markdown generado en {output_path}")

def generate_csv_report(df: pd.DataFrame, output_path: str):
    """
    Genera un reporte en formato CSV a partir de los resultados.
    """
    df.to_csv(output_path, index=False)
    logger.info(f"Reporte CSV generado en {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Genera reportes de benchmarking a partir de resultados JSON.")
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default="./experiments_results", 
        help="Directorio donde se encuentran los archivos JSON de resultados."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./reports", 
        help="Directorio donde se guardarán los reportes generados."
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de logging."
    )

    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    if not os.path.exists(args.input_dir):
        logger.error(f"Directorio de entrada no encontrado: {args.input_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)

    results_data = load_results(args.input_dir)
    if not results_data:
        logger.warning("No se encontraron resultados en el directorio de entrada.")
        return

    df = pd.DataFrame(results_data)

    markdown_output_path = os.path.join(args.output_dir, "benchmark_report.md")
    generate_markdown_report(df, markdown_output_path)

    csv_output_path = os.path.join(args.output_dir, "benchmark_report.csv")
    generate_csv_report(df, csv_output_path)

    logger.info("Generación de reportes completada.")

if __name__ == "__main__":
    main()

