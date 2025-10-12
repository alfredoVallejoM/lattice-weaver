"""
MetaAnalyzer - Capa 4: Análisis de Alto Nivel

Clasifica problemas en arquetipos y recomienda estrategias de resolución.
"""

import sys
sys.path.insert(0, '/home/ubuntu/latticeweaver_v4')

from typing import Dict, List, Tuple
from enum import Enum


class ProblemArchetype(Enum):
    """Arquetipos de problemas CSP."""
    TREE = "tree"
    CYCLE = "cycle"
    SPARSE = "sparse"
    DENSE = "dense"
    DECOMPOSABLE = "decomposable"
    CHAOTIC = "chaotic"
    UNKNOWN = "unknown"


class ResolutionStrategy(Enum):
    """Estrategias de resolución."""
    BACKTRACKING_SIMPLE = "backtracking_simple"
    AC3_BACKTRACKING = "ac3_backtracking"
    PARALLEL_DECOMPOSITION = "parallel_decomposition"
    SYMMETRY_BREAKING = "symmetry_breaking"
    SAT_SOLVER = "sat_solver"
    HEURISTIC_SEARCH = "heuristic_search"


class MetaAnalyzer:
    """
    Capa 4: Análisis de alto nivel y clasificación en arquetipos.
    
    Funcionalidades:
    - Clasificación automática de problemas
    - Recomendación de estrategias
    - Estimación de complejidad
    - Predicción de tiempo de resolución
    """
    
    # Definición de arquetipos
    ARCHETYPES = {
        ProblemArchetype.TREE: {
            'b0': lambda x: x == 1,
            'b1': lambda x: x == 0,
            'density': lambda x: x < 0.1,
            'description': 'Problema con estructura de árbol (sin ciclos)',
        },
        ProblemArchetype.CYCLE: {
            'b0': lambda x: x == 1,
            'b1': lambda x: 0 < x < 5,
            'density': lambda x: x < 0.3,
            'description': 'Problema con pocos ciclos',
        },
        ProblemArchetype.SPARSE: {
            'b0': lambda x: x == 1,
            'b1': lambda x: x < 5,
            'density': lambda x: x < 0.2,
            'description': 'Problema sparse (pocas restricciones)',
        },
        ProblemArchetype.DENSE: {
            'b0': lambda x: x == 1,
            'b1': lambda x: x > 10,
            'density': lambda x: x > 0.5,
            'description': 'Problema denso (muchas restricciones)',
        },
        ProblemArchetype.DECOMPOSABLE: {
            'b0': lambda x: x > 1,
            'b1': lambda x: True,  # Cualquier valor
            'density': lambda x: True,  # Cualquier valor
            'description': 'Problema con componentes independientes',
        },
        ProblemArchetype.CHAOTIC: {
            'b0': lambda x: x == 1,
            'b1': lambda x: x > 100,
            'density': lambda x: x > 0.7,
            'description': 'Problema caótico (muy complejo)',
        },
    }
    
    # Estrategias recomendadas por arquetipo
    STRATEGIES = {
        ProblemArchetype.TREE: {
            'primary': ResolutionStrategy.BACKTRACKING_SIMPLE,
            'alternative': ResolutionStrategy.AC3_BACKTRACKING,
            'use_latticeweaver': False,
            'expected_speedup': '1x',
            'reason': 'Estructura simple, backtracking es suficiente',
        },
        ProblemArchetype.CYCLE: {
            'primary': ResolutionStrategy.SYMMETRY_BREAKING,
            'alternative': ResolutionStrategy.AC3_BACKTRACKING,
            'use_latticeweaver': True,
            'expected_speedup': '2-10x',
            'reason': 'Simetrías detectables con análisis topológico',
        },
        ProblemArchetype.SPARSE: {
            'primary': ResolutionStrategy.AC3_BACKTRACKING,
            'alternative': ResolutionStrategy.BACKTRACKING_SIMPLE,
            'use_latticeweaver': False,
            'expected_speedup': '1-5x',
            'reason': 'AC-3 eficiente para problemas sparse',
        },
        ProblemArchetype.DENSE: {
            'primary': ResolutionStrategy.SAT_SOLVER,
            'alternative': ResolutionStrategy.AC3_BACKTRACKING,
            'use_latticeweaver': False,
            'expected_speedup': '5-50x',
            'reason': 'SAT solvers optimizados para problemas densos',
        },
        ProblemArchetype.DECOMPOSABLE: {
            'primary': ResolutionStrategy.PARALLEL_DECOMPOSITION,
            'alternative': ResolutionStrategy.AC3_BACKTRACKING,
            'use_latticeweaver': True,
            'expected_speedup': '10-100x',
            'reason': 'Descomposición paralela explota componentes independientes',
        },
        ProblemArchetype.CHAOTIC: {
            'primary': ResolutionStrategy.HEURISTIC_SEARCH,
            'alternative': ResolutionStrategy.SAT_SOLVER,
            'use_latticeweaver': False,
            'expected_speedup': '1-2x',
            'reason': 'Problema muy complejo, requiere heurísticas avanzadas',
        },
    }
    
    def __init__(self, topology_summary: Dict = None):
        """
        Args:
            topology_summary: Resumen del análisis topológico
        """
        self.topology_summary = topology_summary
        self.archetype = None
        self.strategy = None
        self.complexity_estimate = None
    
    def classify_problem(self, topology_summary: Dict = None) -> ProblemArchetype:
        """
        Clasifica el problema en un arquetipo.
        
        Args:
            topology_summary: Resumen del análisis topológico
            
        Returns:
            Arquetipo del problema
        """
        if topology_summary:
            self.topology_summary = topology_summary
        
        if not self.topology_summary:
            return ProblemArchetype.UNKNOWN
        
        # Extraer métricas
        b0 = self.topology_summary.get('betti_numbers', {}).get('b0', 1)
        b1 = self.topology_summary.get('betti_numbers', {}).get('b1', 0)
        density = self.topology_summary.get('graph_statistics', {}).get('density', 0)
        
        # Clasificar según arquetipos (orden de prioridad)
        if self._matches_archetype(ProblemArchetype.DECOMPOSABLE, b0, b1, density):
            self.archetype = ProblemArchetype.DECOMPOSABLE
        elif self._matches_archetype(ProblemArchetype.CHAOTIC, b0, b1, density):
            self.archetype = ProblemArchetype.CHAOTIC
        elif self._matches_archetype(ProblemArchetype.TREE, b0, b1, density):
            self.archetype = ProblemArchetype.TREE
        elif self._matches_archetype(ProblemArchetype.CYCLE, b0, b1, density):
            self.archetype = ProblemArchetype.CYCLE
        elif self._matches_archetype(ProblemArchetype.DENSE, b0, b1, density):
            self.archetype = ProblemArchetype.DENSE
        elif self._matches_archetype(ProblemArchetype.SPARSE, b0, b1, density):
            self.archetype = ProblemArchetype.SPARSE
        else:
            self.archetype = ProblemArchetype.UNKNOWN
        
        return self.archetype
    
    def _matches_archetype(self, archetype: ProblemArchetype, b0: int, b1: int, density: float) -> bool:
        """Verifica si las métricas coinciden con un arquetipo."""
        rules = self.ARCHETYPES.get(archetype, {})
        
        b0_match = rules.get('b0', lambda x: True)(b0)
        b1_match = rules.get('b1', lambda x: True)(b1)
        density_match = rules.get('density', lambda x: True)(density)
        
        return b0_match and b1_match and density_match
    
    def recommend_strategy(self, archetype: ProblemArchetype = None) -> Dict:
        """
        Recomienda estrategia de resolución según arquetipo.
        
        Args:
            archetype: Arquetipo del problema (si None, usa self.archetype)
            
        Returns:
            Diccionario con estrategia recomendada
        """
        if archetype:
            self.archetype = archetype
        
        if not self.archetype or self.archetype == ProblemArchetype.UNKNOWN:
            # Estrategia por defecto
            return {
                'primary': ResolutionStrategy.AC3_BACKTRACKING,
                'alternative': ResolutionStrategy.BACKTRACKING_SIMPLE,
                'use_latticeweaver': False,
                'expected_speedup': '1-5x',
                'reason': 'Arquetipo desconocido, usar estrategia estándar',
            }
        
        self.strategy = self.STRATEGIES.get(self.archetype, {})
        return self.strategy
    
    def estimate_complexity(self, problem_size: int = None) -> Dict:
        """
        Estima la complejidad del problema.
        
        Args:
            problem_size: Número de variables (si None, usa topology_summary)
            
        Returns:
            Diccionario con estimación de complejidad
        """
        if not problem_size and self.topology_summary:
            problem_size = self.topology_summary.get('graph_statistics', {}).get('num_nodes', 0)
        
        if not problem_size:
            return {'complexity': 'unknown'}
        
        # Extraer métricas
        b0 = self.topology_summary.get('betti_numbers', {}).get('b0', 1) if self.topology_summary else 1
        b1 = self.topology_summary.get('betti_numbers', {}).get('b1', 0) if self.topology_summary else 0
        density = self.topology_summary.get('graph_statistics', {}).get('density', 0) if self.topology_summary else 0
        
        # Calcular complejidad
        if self.archetype == ProblemArchetype.DECOMPOSABLE:
            # Complejidad reducida por descomposición
            effective_size = problem_size / b0
            complexity_class = self._classify_size(effective_size)
            complexity_factor = b0  # Factor de reducción
        else:
            complexity_class = self._classify_size(problem_size)
            complexity_factor = 1
        
        # Ajustar por densidad y ciclos
        if density > 0.7:
            complexity_multiplier = 2.0
        elif b1 > 100:
            complexity_multiplier = 1.5
        else:
            complexity_multiplier = 1.0
        
        self.complexity_estimate = {
            'complexity_class': complexity_class,
            'effective_size': problem_size / complexity_factor,
            'complexity_multiplier': complexity_multiplier,
            'estimated_time': self._estimate_time(problem_size, complexity_factor, complexity_multiplier),
        }
        
        return self.complexity_estimate
    
    def _classify_size(self, size: float) -> str:
        """Clasifica el tamaño del problema."""
        if size < 20:
            return 'trivial'
        elif size < 50:
            return 'small'
        elif size < 100:
            return 'medium'
        elif size < 500:
            return 'large'
        else:
            return 'massive'
    
    def _estimate_time(self, size: int, factor: float, multiplier: float) -> str:
        """
        Estima el tiempo de resolución.
        
        Fórmula simplificada: T ≈ 2^(n/factor) × multiplier
        """
        effective_size = size / factor
        
        # Estimación logarítmica
        import math
        log_time = effective_size * math.log2(effective_size) * multiplier
        
        if log_time < 10:
            return '<1 second'
        elif log_time < 100:
            return '1-10 seconds'
        elif log_time < 1000:
            return '10-100 seconds'
        elif log_time < 10000:
            return '1-10 minutes'
        elif log_time < 100000:
            return '10-100 minutes'
        else:
            return '>1 hour'
    
    def generate_report(self) -> Dict:
        """
        Genera reporte completo del análisis.
        
        Returns:
            Diccionario con análisis completo
        """
        if not self.archetype:
            self.classify_problem()
        
        if not self.strategy:
            self.recommend_strategy()
        
        if not self.complexity_estimate:
            self.estimate_complexity()
        
        return {
            'archetype': {
                'type': self.archetype.value if self.archetype else 'unknown',
                'description': self.ARCHETYPES.get(self.archetype, {}).get('description', 'Unknown'),
            },
            'strategy': {
                'primary': self.strategy.get('primary').value if self.strategy.get('primary') else 'unknown',
                'alternative': self.strategy.get('alternative').value if self.strategy.get('alternative') else 'unknown',
                'use_latticeweaver': self.strategy.get('use_latticeweaver', False),
                'expected_speedup': self.strategy.get('expected_speedup', 'unknown'),
                'reason': self.strategy.get('reason', 'No reason provided'),
            },
            'complexity': self.complexity_estimate or {},
            'topology': self.topology_summary or {},
        }
    
    def should_use_latticeweaver(self) -> bool:
        """
        Determina si se debe usar LatticeWeaver para este problema.
        
        Returns:
            True si LatticeWeaver es recomendado
        """
        if not self.strategy:
            self.recommend_strategy()
        
        return self.strategy.get('use_latticeweaver', False)
    
    def get_expected_speedup(self) -> str:
        """
        Retorna el speedup esperado si se usa LatticeWeaver.
        
        Returns:
            String con rango de speedup (ej: '10-100x')
        """
        if not self.strategy:
            self.recommend_strategy()
        
        if not self.should_use_latticeweaver():
            return 'N/A (not recommended)'
        
        return self.strategy.get('expected_speedup', 'unknown')

