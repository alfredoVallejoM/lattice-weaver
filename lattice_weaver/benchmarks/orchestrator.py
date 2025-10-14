"""
Módulo de orquestación para pruebas integradas del compilador multiescala.

Este módulo coordina la compilación y resolución de problemas CSP, midiendo
métricas de rendimiento para validar la mejora del sistema con el compilador.
"""

import time
import tracemalloc
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from lattice_weaver.core.csp_problem import CSP
from lattice_weaver.compiler_multiescala.level_0 import Level0
from lattice_weaver.compiler_multiescala.level_1 import Level1
from lattice_weaver.compiler_multiescala.level_2 import Level2
from lattice_weaver.compiler_multiescala.level_3 import Level3
from lattice_weaver.compiler_multiescala.level_4 import Level4
from lattice_weaver.compiler_multiescala.level_5 import Level5
from lattice_weaver.compiler_multiescala.level_6 import Level6


@dataclass
class BenchmarkMetrics:
    """Métricas de rendimiento de una ejecución de benchmark."""
    
    # Tiempos
    total_time: float = 0.0  # Tiempo total de ejecución (segundos)
    compilation_time: float = 0.0  # Tiempo de compilación (segundos)
    solving_time: float = 0.0  # Tiempo de resolución (segundos)
    
    # Memoria
    peak_memory: float = 0.0  # Pico de memoria (MB)
    
    # Métricas del solucionador
    nodes_explored: int = 0  # Nodos explorados en el árbol de búsqueda
    backtracks: int = 0  # Número de backtracks
    solution_found: bool = False  # Si se encontró una solución
    
    # Métricas del compilador
    compilation_level: Optional[str] = None  # Nivel de compilación usado
    compiled_variables: int = 0  # Número de variables después de compilar
    compiled_constraints: int = 0  # Número de restricciones después de compilar
    compression_ratio: float = 1.0  # Ratio de compresión (original/compilado)
    
    # Información adicional
    error: Optional[str] = None  # Mensaje de error si hubo fallo
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadatos adicionales


class CompilationStrategy:
    """Estrategia base para la compilación de problemas CSP."""
    
    def compile(self, csp: CSP) -> tuple[Any, BenchmarkMetrics]:
        """
        Compila un problema CSP según la estrategia.
        
        Args:
            csp: Problema CSP a compilar.
            
        Returns:
            Tupla con el problema compilado y las métricas de compilación.
        """
        raise NotImplementedError("Subclasses must implement compile()")
    
    def get_name(self) -> str:
        """Retorna el nombre de la estrategia."""
        raise NotImplementedError("Subclasses must implement get_name()")


class NoCompilationStrategy(CompilationStrategy):
    """Estrategia que no compila el problema (sistema base)."""
    
    def compile(self, csp: CSP) -> tuple[CSP, BenchmarkMetrics]:
        metrics = BenchmarkMetrics(
            compilation_level="L0",
            compiled_variables=len(csp.variables),
            compiled_constraints=len(csp.constraints)
        )
        return csp, metrics
    
    def get_name(self) -> str:
        return "NoCompilation"


class FixedLevelStrategy(CompilationStrategy):
    """Estrategia que compila a un nivel de abstracción fijo."""
    
    def __init__(self, level: int):
        """
        Inicializa la estrategia.
        
        Args:
            level: Nivel de abstracción (0-6).
        """
        if level < 0 or level > 6:
            raise ValueError(f"Level must be between 0 and 6, got {level}")
        self.level = level
    
    def compile(self, csp: CSP) -> tuple[Any, BenchmarkMetrics]:
        start_time = time.time()
        
        # Compilar al nivel especificado
        l0 = Level0(csp)
        current_level = l0
        
        if self.level >= 1:
            l1 = Level1([], [], config={'original_domains': csp.domains})
            l1.build_from_lower(l0)
            current_level = l1
        
        if self.level >= 2:
            l2 = Level2([], [], [], config={'original_domains': csp.domains})
            l2.build_from_lower(l1)
            current_level = l2
        
        if self.level >= 3:
            l3 = Level3([], [], [], [], config={'original_domains': csp.domains})
            l3.build_from_lower(l2)
            current_level = l3
        
        if self.level >= 4:
            l4 = Level4([], [], [], config={
                'original_domains': csp.domains,
                'original_structures': l3.structures,
                'original_isolated_patterns': l3.isolated_patterns,
                'original_isolated_blocks': l3.isolated_blocks
            })
            l4.build_from_lower(l3)
            current_level = l4
        
        if self.level >= 5:
            l5 = Level5([], [], [], config={'original_domains': csp.domains})
            l5.build_from_lower(l4)
            current_level = l5
        
        if self.level >= 6:
            from lattice_weaver.compiler_multiescala.level_6 import ProblemDescription
            l6 = Level6(
                problem=ProblemDescription(
                    name="Benchmark Problem",
                    description="A CSP for benchmarking",
                    domain="Benchmarking"
                )
            )
            l6.build_from_lower(l5, "Benchmark Problem", "A CSP for benchmarking", "Benchmarking")
            current_level = l6
        
        compilation_time = time.time() - start_time
        
        # Calcular métricas de compilación
        stats = current_level.get_statistics()
        
        metrics = BenchmarkMetrics(
            compilation_time=compilation_time,
            compilation_level=f"L{self.level}",
            compiled_variables=stats.get('total_variables', len(csp.variables)),
            compiled_constraints=stats.get('total_constraints', len(csp.constraints)),
            compression_ratio=len(csp.variables) / max(1, stats.get('total_variables', len(csp.variables)))
        )
        
        return current_level, metrics
    
    def get_name(self) -> str:
        return f"FixedLevel_L{self.level}"


class Orchestrator:
    """
    Orquestador de pruebas integradas.
    
    Coordina la compilación y resolución de problemas CSP, midiendo métricas
    de rendimiento para validar la mejora del sistema con el compilador.
    """
    
    def __init__(self, solver: Callable[[CSP], tuple[Optional[Dict], int, int]]):
        """
        Inicializa el orquestador.
        
        Args:
            solver: Función que resuelve un CSP y retorna (solución, nodos, backtracks).
        """
        self.solver = solver
    
    def run_benchmark(
        self,
        csp: CSP,
        strategy: CompilationStrategy,
        timeout: Optional[float] = None
    ) -> BenchmarkMetrics:
        """
        Ejecuta un benchmark con una estrategia de compilación.
        
        Args:
            csp: Problema CSP a resolver.
            strategy: Estrategia de compilación a usar.
            timeout: Tiempo máximo de ejecución en segundos (None = sin límite).
            
        Returns:
            Métricas de rendimiento del benchmark.
        """
        try:
            # Iniciar medición de memoria
            tracemalloc.start()
            start_time = time.time()
            
            # Compilar el problema
            compiled_problem, compilation_metrics = strategy.compile(csp)
            
            # Si el problema compilado no es un CSP, refinarlo a L0
            if not isinstance(compiled_problem, CSP):
                # Refinar recursivamente hasta L0
                current = compiled_problem
                while hasattr(current, 'refine_to_lower'):
                    current = current.refine_to_lower()
                    if isinstance(current, Level0):
                        compiled_problem = current.csp
                        break
                    elif isinstance(current, CSP):
                        compiled_problem = current
                        break
            
            # Resolver el problema
            solving_start = time.time()
            solution, nodes_explored, backtracks = self.solver(compiled_problem)
            solving_time = time.time() - solving_start
            
            # Finalizar medición
            total_time = time.time() - start_time
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Combinar métricas
            metrics = compilation_metrics
            metrics.total_time = total_time
            metrics.solving_time = solving_time
            metrics.peak_memory = peak_memory / (1024 * 1024)  # Convertir a MB
            metrics.nodes_explored = nodes_explored
            metrics.backtracks = backtracks
            metrics.solution_found = solution is not None
            
            return metrics
            
        except Exception as e:
            tracemalloc.stop()
            return BenchmarkMetrics(
                error=str(e)
            )
    
    def run_comparison(
        self,
        csp: CSP,
        strategies: list[CompilationStrategy],
        timeout: Optional[float] = None
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Ejecuta un benchmark comparativo con múltiples estrategias.
        
        Args:
            csp: Problema CSP a resolver.
            strategies: Lista de estrategias de compilación a comparar.
            timeout: Tiempo máximo de ejecución por estrategia en segundos.
            
        Returns:
            Diccionario con las métricas de cada estrategia.
        """
        results = {}
        
        for strategy in strategies:
            strategy_name = strategy.get_name()
            print(f"Running benchmark with strategy: {strategy_name}")
            
            metrics = self.run_benchmark(csp, strategy, timeout)
            results[strategy_name] = metrics
            
            if metrics.error:
                print(f"  ERROR: {metrics.error}")
            else:
                print(f"  Total time: {metrics.total_time:.4f}s")
                print(f"  Compilation time: {metrics.compilation_time:.4f}s")
                print(f"  Solving time: {metrics.solving_time:.4f}s")
                print(f"  Peak memory: {metrics.peak_memory:.2f} MB")
                print(f"  Nodes explored: {metrics.nodes_explored}")
                print(f"  Solution found: {metrics.solution_found}")
        
        return results

