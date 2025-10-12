import argparse
import json
import os
import logging

from lattice_weaver.experimentation.config import ExperimentConfig
from lattice_weaver.experimentation.runner import BenchmarkRunner
from lattice_weaver.problems.catalog import ProblemCatalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de benchmarking para problemas CSP.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Ruta al archivo de configuración JSON del experimento."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./experiments_results", 
        help="Directorio donde se guardarán los resultados."
    )
    parser.add_argument(
        "--repetitions", 
        type=int, 
        default=1, 
        help="Número de veces que se repetirá cada experimento."
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

    if not os.path.exists(args.config):
        logger.error(f"Archivo de configuración no encontrado: {args.config}")
        return

    with open(args.config, "r") as f:
        config_data = json.load(f)

    # Actualizar la configuración con los argumentos de la línea de comandos
    config_data["output_dir"] = args.output_dir
    config_data["repetitions"] = args.repetitions

    try:
        config = ExperimentConfig(**config_data)
    except TypeError as e:
        logger.error(f"Error al cargar la configuración del experimento: {e}")
        logger.error("Asegúrese de que el archivo JSON coincide con la estructura de ExperimentConfig.")
        return

    # Asegurarse de que todas las familias de problemas estén registradas
    # Esto es importante si el script se ejecuta de forma independiente
    # y los módulos de generadores no han sido importados previamente.
    # Una forma de hacerlo es importar todos los generadores conocidos.
    # Por simplicidad, aquí asumimos que ya están registrados o que el usuario
    # se asegura de que lo estén.
    from lattice_weaver.problems.catalog import get_catalog
    catalog = get_catalog()
    if not catalog.has(config.problem_family):
        logger.warning(f"La familia de problemas \'{config.problem_family}\' no está registrada. "
                       "Asegúrese de que el módulo del generador ha sido importado.")
        # Intentar importar el módulo del generador dinámicamente
        try:
            __import__(f"lattice_weaver.problems.generators.{config.problem_family}")
            if not catalog.has(config.problem_family):
                logger.error(f"La familia \'{config.problem_family}\' no se registró después de la importación. "
                             "Verifique el método _register() en el generador.")
                return
        except ImportError:
            logger.error(f"No se pudo importar el módulo del generador para \'{config.problem_family}\'.")
            return

    runner = BenchmarkRunner(config)
    logger.info(f"Iniciando experimento para {config.problem_family} con {config.repetitions} repeticiones...")
    results = runner.run()
    logger.info("Experimento completado. Resultados guardados.")

    # Opcional: imprimir un resumen de los resultados
    print("\n--- Resumen de Resultados ---")
    for r in results:
        print(f"Solver: {r['solver_name']}, Tiempo: {r['time_taken']:.4f}s, Soluciones: {r['solutions_found']}")

if __name__ == "__main__":
    main()

