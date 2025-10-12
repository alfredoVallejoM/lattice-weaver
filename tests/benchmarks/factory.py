"""
Factory para generación automatizada de problemas de benchmark.

Este módulo proporciona una interfaz unificada para generar problemas
de diferentes tipos y tamaños de forma automatizada.
"""
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import random
from enum import Enum

from .problems import (
    BenchmarkProblem,
    create_nqueens,
    create_sudoku_4x4,
    create_graph_coloring,
    create_map_coloring,
    create_scheduling_problem,
)


class ProblemType(Enum):
    """Tipos de problemas disponibles."""
    NQUEENS = "nqueens"
    SUDOKU = "sudoku"
    GRAPH_COLORING = "graph_coloring"
    MAP_COLORING = "map_coloring"
    SCHEDULING = "scheduling"


@dataclass
class ProblemConfig:
    """
    Configuración para generación de problemas.
    
    Attributes:
        problem_type: Tipo de problema
        size: Tamaño del problema
        difficulty: Nivel de dificultad deseado
        seed: Semilla para generación aleatoria
        custom_params: Parámetros adicionales específicos del problema
    """
    problem_type: ProblemType
    size: int
    difficulty: Optional[str] = None
    seed: Optional[int] = None
    custom_params: Optional[Dict[str, Any]] = None


class ProblemFactory:
    """
    Factory para generación automatizada de problemas.
    
    Permite generar problemas de diferentes tipos y tamaños
    de forma programática y reproducible.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Inicializa la factory.
        
        Args:
            seed: Semilla global para generación aleatoria
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    def create(self, config: ProblemConfig) -> BenchmarkProblem:
        """
        Crea un problema según la configuración.
        
        Args:
            config: Configuración del problema
        
        Returns:
            Problema de benchmark
        
        Raises:
            ValueError: Si el tipo de problema no es soportado
        """
        if config.seed is not None:
            random.seed(config.seed)
        
        if config.problem_type == ProblemType.NQUEENS:
            return self._create_nqueens(config)
        elif config.problem_type == ProblemType.SUDOKU:
            return self._create_sudoku(config)
        elif config.problem_type == ProblemType.GRAPH_COLORING:
            return self._create_graph_coloring(config)
        elif config.problem_type == ProblemType.MAP_COLORING:
            return self._create_map_coloring(config)
        elif config.problem_type == ProblemType.SCHEDULING:
            return self._create_scheduling(config)
        else:
            raise ValueError(f"Tipo de problema no soportado: {config.problem_type}")
    
    def _create_nqueens(self, config: ProblemConfig) -> BenchmarkProblem:
        """Crea problema de N-Reinas."""
        return create_nqueens(config.size)
    
    def _create_sudoku(self, config: ProblemConfig) -> BenchmarkProblem:
        """
        Crea problema de Sudoku.
        
        El tamaño determina el número de givens:
        - size = 4: 4 givens (muy difícil)
        - size = 6: 6 givens (difícil)
        - size = 8: 8 givens (medio)
        - size = 10: 10 givens (fácil)
        """
        n = 4  # Sudoku 4x4
        num_givens = min(config.size, 12)  # Máximo 12 givens
        
        # Generar givens aleatorios
        givens = {}
        cells = [(i, j) for i in range(n) for j in range(n)]
        random.shuffle(cells)
        
        for i in range(num_givens):
            row, col = cells[i]
            value = random.randint(1, n)
            givens[f"C{row}{col}"] = value
        
        return create_sudoku_4x4(givens)
    
    def _create_graph_coloring(self, config: ProblemConfig) -> BenchmarkProblem:
        """
        Crea problema de coloreo de grafos.
        
        El tamaño determina el número de nodos.
        La dificultad se controla con la densidad de aristas.
        """
        num_nodes = config.size
        
        # Determinar número de colores y densidad según dificultad
        if config.difficulty == "easy":
            num_colors = max(3, num_nodes // 2)
            density = 0.2
        elif config.difficulty == "hard":
            num_colors = max(2, num_nodes // 4)
            density = 0.7
        else:  # medium o None
            num_colors = max(3, num_nodes // 3)
            density = 0.4
        
        # Obtener de custom_params si está disponible
        if config.custom_params:
            num_colors = config.custom_params.get('num_colors', num_colors)
            density = config.custom_params.get('density', density)
        
        # Generar aristas aleatorias
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < density:
                    edges.append((i, j))
        
        return create_graph_coloring(num_nodes, edges, num_colors)
    
    def _create_map_coloring(self, config: ProblemConfig) -> BenchmarkProblem:
        """Crea problema de coloreo de mapa (Australia)."""
        return create_map_coloring()
    
    def _create_scheduling(self, config: ProblemConfig) -> BenchmarkProblem:
        """
        Crea problema de scheduling.
        
        El tamaño determina el número de trabajos.
        El número de slots se calcula automáticamente.
        """
        num_jobs = config.size
        
        # Calcular slots según dificultad
        if config.difficulty == "easy":
            num_slots = num_jobs  # Muchos slots
        elif config.difficulty == "hard":
            num_slots = max(2, num_jobs // 2)  # Pocos slots
        else:  # medium o None
            num_slots = max(3, (num_jobs * 2) // 3)
        
        # Obtener de custom_params si está disponible
        if config.custom_params:
            num_slots = config.custom_params.get('num_slots', num_slots)
        
        return create_scheduling_problem(num_jobs, num_slots)
    
    def create_batch(self, configs: List[ProblemConfig]) -> List[BenchmarkProblem]:
        """
        Crea múltiples problemas en batch.
        
        Args:
            configs: Lista de configuraciones
        
        Returns:
            Lista de problemas
        """
        return [self.create(config) for config in configs]
    
    def create_suite(self, problem_type: ProblemType, 
                    sizes: List[int],
                    difficulty: Optional[str] = None) -> List[BenchmarkProblem]:
        """
        Crea una suite de problemas del mismo tipo con diferentes tamaños.
        
        Args:
            problem_type: Tipo de problema
            sizes: Lista de tamaños
            difficulty: Nivel de dificultad
        
        Returns:
            Lista de problemas
        """
        configs = [
            ProblemConfig(
                problem_type=problem_type,
                size=size,
                difficulty=difficulty
            )
            for size in sizes
        ]
        return self.create_batch(configs)
    
    def create_scalability_suite(self, problem_type: ProblemType,
                                 min_size: int,
                                 max_size: int,
                                 step: int = 1) -> List[BenchmarkProblem]:
        """
        Crea una suite para análisis de escalabilidad.
        
        Args:
            problem_type: Tipo de problema
            min_size: Tamaño mínimo
            max_size: Tamaño máximo
            step: Incremento entre tamaños
        
        Returns:
            Lista de problemas
        """
        sizes = list(range(min_size, max_size + 1, step))
        return self.create_suite(problem_type, sizes)
    
    def create_difficulty_suite(self, problem_type: ProblemType,
                               size: int) -> Dict[str, BenchmarkProblem]:
        """
        Crea problemas del mismo tamaño con diferentes dificultades.
        
        Args:
            problem_type: Tipo de problema
            size: Tamaño del problema
        
        Returns:
            Diccionario {dificultad: problema}
        """
        difficulties = ["easy", "medium", "hard"]
        problems = {}
        
        for difficulty in difficulties:
            config = ProblemConfig(
                problem_type=problem_type,
                size=size,
                difficulty=difficulty
            )
            problems[difficulty] = self.create(config)
        
        return problems


class ProblemGenerator:
    """
    Generador de alto nivel para casos de uso comunes.
    """
    
    def __init__(self, factory: Optional[ProblemFactory] = None):
        """
        Inicializa el generador.
        
        Args:
            factory: Factory a usar (crea una nueva si no se proporciona)
        """
        self.factory = factory or ProblemFactory()
    
    def generate_nqueens_suite(self, max_n: int = 12) -> List[BenchmarkProblem]:
        """
        Genera suite estándar de N-Reinas.
        
        Args:
            max_n: Tamaño máximo (default: 12)
        
        Returns:
            Lista de problemas N-Reinas [4, 5, 6, ..., max_n]
        """
        return self.factory.create_scalability_suite(
            ProblemType.NQUEENS,
            min_size=4,
            max_size=max_n,
            step=1
        )
    
    def generate_quick_suite(self) -> List[BenchmarkProblem]:
        """
        Genera suite rápida para validación.
        
        Returns:
            Lista de problemas pequeños y rápidos
        """
        configs = [
            ProblemConfig(ProblemType.NQUEENS, size=4),
            ProblemConfig(ProblemType.NQUEENS, size=6),
            ProblemConfig(ProblemType.SUDOKU, size=8),
            ProblemConfig(ProblemType.GRAPH_COLORING, size=5, difficulty="easy"),
            ProblemConfig(ProblemType.SCHEDULING, size=5, difficulty="easy"),
        ]
        return self.factory.create_batch(configs)
    
    def generate_stress_suite(self) -> List[BenchmarkProblem]:
        """
        Genera suite de estrés con problemas grandes.
        
        Returns:
            Lista de problemas grandes y complejos
        """
        configs = [
            ProblemConfig(ProblemType.NQUEENS, size=12),
            ProblemConfig(ProblemType.NQUEENS, size=14),
            ProblemConfig(ProblemType.GRAPH_COLORING, size=20, difficulty="hard"),
            ProblemConfig(ProblemType.SCHEDULING, size=15, difficulty="hard"),
        ]
        return self.factory.create_batch(configs)
    
    def generate_comparison_suite(self) -> Dict[str, List[BenchmarkProblem]]:
        """
        Genera suite para comparación de algoritmos.
        
        Returns:
            Diccionario {categoría: lista de problemas}
        """
        return {
            "small": [
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=4)),
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=5)),
            ],
            "medium": [
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=6)),
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=7)),
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=8)),
            ],
            "large": [
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=10)),
                self.factory.create(ProblemConfig(ProblemType.NQUEENS, size=12)),
            ],
        }
    
    def generate_custom_suite(self, 
                            problem_type: ProblemType,
                            sizes: List[int],
                            difficulties: Optional[List[str]] = None) -> List[BenchmarkProblem]:
        """
        Genera suite personalizada.
        
        Args:
            problem_type: Tipo de problema
            sizes: Lista de tamaños
            difficulties: Lista de dificultades (opcional)
        
        Returns:
            Lista de problemas
        """
        if difficulties is None:
            return self.factory.create_suite(problem_type, sizes)
        
        configs = []
        for size in sizes:
            for difficulty in difficulties:
                configs.append(ProblemConfig(
                    problem_type=problem_type,
                    size=size,
                    difficulty=difficulty
                ))
        
        return self.factory.create_batch(configs)


# Funciones de conveniencia
def quick_problem(problem_type: str, size: int, **kwargs) -> BenchmarkProblem:
    """
    Crea un problema rápidamente.
    
    Args:
        problem_type: Tipo de problema ("nqueens", "sudoku", etc.)
        size: Tamaño del problema
        **kwargs: Parámetros adicionales
    
    Returns:
        Problema de benchmark
    
    Examples:
        >>> problem = quick_problem("nqueens", 8)
        >>> problem = quick_problem("graph_coloring", 10, difficulty="hard")
    """
    factory = ProblemFactory()
    config = ProblemConfig(
        problem_type=ProblemType(problem_type),
        size=size,
        difficulty=kwargs.get('difficulty'),
        seed=kwargs.get('seed'),
        custom_params=kwargs.get('custom_params')
    )
    return factory.create(config)


def batch_problems(problem_type: str, sizes: List[int], **kwargs) -> List[BenchmarkProblem]:
    """
    Crea múltiples problemas del mismo tipo.
    
    Args:
        problem_type: Tipo de problema
        sizes: Lista de tamaños
        **kwargs: Parámetros adicionales
    
    Returns:
        Lista de problemas
    
    Examples:
        >>> problems = batch_problems("nqueens", [4, 6, 8, 10])
        >>> problems = batch_problems("graph_coloring", [5, 10, 15], difficulty="medium")
    """
    factory = ProblemFactory(seed=kwargs.get('seed'))
    return factory.create_suite(
        ProblemType(problem_type),
        sizes,
        difficulty=kwargs.get('difficulty')
    )


def scalability_suite(problem_type: str, min_size: int, max_size: int, step: int = 1) -> List[BenchmarkProblem]:
    """
    Crea suite para análisis de escalabilidad.
    
    Args:
        problem_type: Tipo de problema
        min_size: Tamaño mínimo
        max_size: Tamaño máximo
        step: Incremento
    
    Returns:
        Lista de problemas
    
    Examples:
        >>> problems = scalability_suite("nqueens", 4, 12, step=2)  # [4, 6, 8, 10, 12]
    """
    factory = ProblemFactory()
    return factory.create_scalability_suite(
        ProblemType(problem_type),
        min_size,
        max_size,
        step
    )


# Suites predefinidas
def get_quick_suite() -> List[BenchmarkProblem]:
    """Obtiene suite rápida para validación."""
    return ProblemGenerator().generate_quick_suite()


def get_stress_suite() -> List[BenchmarkProblem]:
    """Obtiene suite de estrés."""
    return ProblemGenerator().generate_stress_suite()


def get_nqueens_suite(max_n: int = 12) -> List[BenchmarkProblem]:
    """Obtiene suite de N-Reinas."""
    return ProblemGenerator().generate_nqueens_suite(max_n)


def get_comparison_suite() -> Dict[str, List[BenchmarkProblem]]:
    """Obtiene suite para comparación."""
    return ProblemGenerator().generate_comparison_suite()

