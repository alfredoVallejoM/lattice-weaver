"""
ExperimentRunner - Framework de Experimentación Masiva.

Este módulo proporciona un framework para ejecutar experimentos masivos
con diferentes configuraciones, problemas y parámetros del solver.

Autor: LatticeWeaver Team
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import yaml
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from lattice_weaver.arc_weaver.adaptive_consistency import AdaptiveConsistencyEngine
from lattice_weaver.arc_weaver.tracing import SearchSpaceTracer


@dataclass
class ExperimentConfig:
    """
    Configuración de un experimento.
    
    Attributes:
        name: Nombre del experimento
        problem_generator: Función que genera el problema
        problem_params: Parámetros para el generador de problemas
        solver_params: Parámetros del solver
        num_runs: Número de ejecuciones por configuración
        enable_tracing: Si se debe habilitar el tracing
        trace_output_dir: Directorio para guardar traces
        timeout: Timeout en segundos por ejecución
    """
    name: str
    problem_generator: Optional[Callable] = None
    problem_params: Dict[str, Any] = field(default_factory=dict)
    solver_params: Dict[str, Any] = field(default_factory=dict)
    num_runs: int = 1
    enable_tracing: bool = False
    trace_output_dir: Optional[str] = None
    timeout: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario."""
        d = asdict(self)
        # Excluir el callable que no es serializable
        d.pop('problem_generator', None)
        return d


@dataclass
class ExperimentResult:
    """
    Resultado de una ejecución de experimento.
    
    Attributes:
        config_name: Nombre de la configuración
        run_id: ID de la ejecución
        success: Si la ejecución fue exitosa
        nodes_explored: Nodos explorados
        backtracks: Número de backtracks
        solutions_found: Número de soluciones encontradas
        time_elapsed: Tiempo transcurrido en segundos
        timeout_reached: Si se alcanzó el timeout
        error_message: Mensaje de error si falló
        trace_path: Ruta del archivo de trace
        metadata: Metadata adicional
    """
    config_name: str
    run_id: int
    success: bool
    nodes_explored: int = 0
    backtracks: int = 0
    solutions_found: int = 0
    time_elapsed: float = 0.0
    timeout_reached: bool = False
    error_message: Optional[str] = None
    trace_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario."""
        return asdict(self)


class ExperimentRunner:
    """
    Framework para ejecutar experimentos masivos con el solver.
    
    Permite ejecutar múltiples configuraciones en paralelo, con diferentes
    problemas y parámetros, recopilando estadísticas y traces para análisis.
    
    Examples:
        >>> runner = ExperimentRunner()
        >>> config = ExperimentConfig(
        ...     name="nqueens_8",
        ...     problem_generator=create_nqueens_problem,
        ...     problem_params={"n": 8},
        ...     num_runs=10
        ... )
        >>> results = runner.run_experiment(config)
        >>> runner.save_results("results.json")
    """
    
    def __init__(self, output_dir: str = "experiments"):
        """
        Inicializa el runner.
        
        Args:
            output_dir: Directorio base para guardar resultados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ExperimentResult] = []
        self.configs: List[ExperimentConfig] = []
    
    def add_config(self, config: ExperimentConfig):
        """
        Añade una configuración de experimento.
        
        Args:
            config: Configuración del experimento
        """
        self.configs.append(config)
    
    def load_config_from_yaml(self, yaml_path: str):
        """
        Carga configuraciones desde un archivo YAML.
        
        El archivo YAML debe tener el formato:
        
        experiments:
          - name: "exp1"
            problem_params:
              n: 8
            solver_params:
              max_solutions: 1
            num_runs: 10
        
        Args:
            yaml_path: Ruta del archivo YAML
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for exp_data in data.get('experiments', []):
            config = ExperimentConfig(
                name=exp_data['name'],
                problem_params=exp_data.get('problem_params', {}),
                solver_params=exp_data.get('solver_params', {}),
                num_runs=exp_data.get('num_runs', 1),
                enable_tracing=exp_data.get('enable_tracing', False),
                trace_output_dir=exp_data.get('trace_output_dir'),
                timeout=exp_data.get('timeout')
            )
            self.add_config(config)
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[ExperimentResult]:
        """
        Ejecuta un experimento con la configuración dada.
        
        Args:
            config: Configuración del experimento
            parallel: Si se debe ejecutar en paralelo
            max_workers: Número máximo de workers (None = CPU count)
            
        Returns:
            Lista de resultados de las ejecuciones
        """
        print(f"Ejecutando experimento: {config.name}")
        print(f"  Número de ejecuciones: {config.num_runs}")
        print(f"  Modo paralelo: {parallel}")
        
        results = []
        
        if parallel and config.num_runs > 1:
            # Ejecución paralela
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for run_id in range(config.num_runs):
                    future = executor.submit(
                        self._run_single_experiment,
                        config,
                        run_id
                    )
                    futures.append(future)
                
                # Recopilar resultados
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        self.results.append(result)
                        
                        if result.success:
                            print(f"  ✓ Run {result.run_id}: {result.time_elapsed:.4f}s, "
                                  f"{result.nodes_explored} nodos")
                        else:
                            print(f"  ✗ Run {result.run_id}: {result.error_message}")
                    
                    except Exception as e:
                        print(f"  ✗ Error en ejecución: {e}")
        
        else:
            # Ejecución secuencial
            for run_id in range(config.num_runs):
                result = self._run_single_experiment(config, run_id)
                results.append(result)
                self.results.append(result)
                
                if result.success:
                    print(f"  ✓ Run {run_id}: {result.time_elapsed:.4f}s, "
                          f"{result.nodes_explored} nodos")
                else:
                    print(f"  ✗ Run {run_id}: {result.error_message}")
        
        print(f"Experimento completado: {len(results)} ejecuciones")
        print()
        
        return results
    
    def _run_single_experiment(
        self,
        config: ExperimentConfig,
        run_id: int
    ) -> ExperimentResult:
        """
        Ejecuta una sola ejecución de experimento.
        
        Args:
            config: Configuración del experimento
            run_id: ID de la ejecución
            
        Returns:
            Resultado de la ejecución
        """
        try:
            # Generar problema
            if config.problem_generator is None:
                raise ValueError("problem_generator no está definido")
            
            problem = config.problem_generator(**config.problem_params)
            
            # Configurar tracing
            tracer = None
            trace_path = None
            
            if config.enable_tracing and config.trace_output_dir:
                trace_dir = Path(config.trace_output_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = str(trace_dir / f"{config.name}_run_{run_id}.csv")
                
                tracer = SearchSpaceTracer(
                    enabled=True,
                    output_path=trace_path,
                    async_mode=True
                )
            
            # Crear solver
            engine = AdaptiveConsistencyEngine(tracer=tracer)
            
            # Ejecutar con timeout
            start_time = time.time()
            
            try:
                stats = engine.solve(problem, **config.solver_params)
                time_elapsed = time.time() - start_time
                timeout_reached = False
            
            except TimeoutError:
                time_elapsed = time.time() - start_time
                timeout_reached = True
                stats = None
            
            # Crear resultado
            if stats is not None:
                result = ExperimentResult(
                    config_name=config.name,
                    run_id=run_id,
                    success=True,
                    nodes_explored=stats.nodes_explored,
                    backtracks=stats.backtracks,
                    solutions_found=len(stats.solutions),
                    time_elapsed=time_elapsed,
                    timeout_reached=timeout_reached,
                    trace_path=trace_path,
                    metadata=config.to_dict()
                )
            else:
                result = ExperimentResult(
                    config_name=config.name,
                    run_id=run_id,
                    success=False,
                    time_elapsed=time_elapsed,
                    timeout_reached=timeout_reached,
                    error_message="Timeout alcanzado",
                    trace_path=trace_path,
                    metadata=config.to_dict()
                )
            
            return result
        
        except Exception as e:
            return ExperimentResult(
                config_name=config.name,
                run_id=run_id,
                success=False,
                error_message=str(e),
                metadata=config.to_dict()
            )
    
    def run_all(
        self,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[ExperimentResult]:
        """
        Ejecuta todos los experimentos configurados.
        
        Args:
            parallel: Si se debe ejecutar en paralelo
            max_workers: Número máximo de workers
            
        Returns:
            Lista de todos los resultados
        """
        all_results = []
        
        for config in self.configs:
            results = self.run_experiment(config, parallel, max_workers)
            all_results.extend(results)
        
        return all_results
    
    def save_results(self, output_path: Optional[str] = None):
        """
        Guarda los resultados en un archivo JSON.
        
        Args:
            output_path: Ruta del archivo de salida (None = auto-generar)
        """
        if output_path is None:
            output_path = self.output_dir / "results.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'results': [r.to_dict() for r in self.results],
            'configs': [c.to_dict() for c in self.configs]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Resultados guardados en: {output_path}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte los resultados a un DataFrame de pandas.
        
        Returns:
            DataFrame con los resultados
        """
        data = [r.to_dict() for r in self.results]
        return pd.DataFrame(data)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calcula estadísticas resumidas de todos los experimentos.
        
        Returns:
            Diccionario con estadísticas agregadas
        """
        df = self.to_dataframe()
        
        if len(df) == 0:
            return {}
        
        # Filtrar solo ejecuciones exitosas
        successful = df[df['success'] == True]
        
        if len(successful) == 0:
            return {
                'total_runs': len(df),
                'successful_runs': 0,
                'failed_runs': len(df)
            }
        
        stats = {
            'total_runs': len(df),
            'successful_runs': len(successful),
            'failed_runs': len(df) - len(successful),
            'avg_nodes_explored': successful['nodes_explored'].mean(),
            'std_nodes_explored': successful['nodes_explored'].std(),
            'avg_backtracks': successful['backtracks'].mean(),
            'std_backtracks': successful['backtracks'].std(),
            'avg_time': successful['time_elapsed'].mean(),
            'std_time': successful['time_elapsed'].std(),
            'min_time': successful['time_elapsed'].min(),
            'max_time': successful['time_elapsed'].max(),
            'total_solutions': successful['solutions_found'].sum()
        }
        
        return stats

