"""
Auto Profiler - Profiling Automático para Decisiones Adaptativas

Sistema de profiling que mide características del problema y toma decisiones
sobre qué optimizaciones activar.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class OptimizationLevel(Enum):
    """Niveles de optimización."""
    NONE = 0
    LITE = 1
    MEDIUM = 2
    FULL = 3


@dataclass
class ProfilingMetrics:
    """Métricas recolectadas durante profiling."""
    
    # Características del problema
    num_variables: int = 0
    num_constraints: int = 0
    avg_domain_size: float = 0.0
    max_domain_size: int = 0
    has_soft_constraints: bool = False
    has_hierarchy: bool = False
    constraint_density: float = 0.0  # constraints / (vars * vars)
    
    # Métricas de ejecución
    time_in_propagation: float = 0.0
    time_in_search: float = 0.0
    time_in_heuristics: float = 0.0
    time_in_energy: float = 0.0
    
    num_backtracks: int = 0
    num_propagations: int = 0
    num_constraint_evaluations: int = 0
    num_energy_computations: int = 0
    
    # Efectividad de optimizaciones
    propagation_effectiveness: float = 0.0  # valores eliminados / propagaciones
    heuristic_quality: float = 0.0  # 1 - (backtracks / nodos)
    
    # Uso de recursos
    peak_memory_mb: float = 0.0
    avg_assignment_size: float = 0.0


@dataclass
class ProfilingRecommendations:
    """Recomendaciones basadas en profiling."""
    
    optimization_level: OptimizationLevel = OptimizationLevel.MEDIUM
    
    use_ac3: bool = True
    use_tms: bool = True
    use_homotopy_rules: bool = True
    use_energy_caching: bool = True
    use_sparse_sets: bool = True
    
    propagation_level: str = "AC3"  # "NONE", "FC", "AC3", "PC"
    heuristic_mode: str = "ENHANCED"  # "SIMPLE", "ENHANCED", "ADAPTIVE"
    
    reasons: List[str] = field(default_factory=list)


class AutoProfiler:
    """
    Profiler automático que analiza el problema y recomienda optimizaciones.
    """
    
    def __init__(self, profiling_backtracks: int = 100):
        """
        Inicializa AutoProfiler.
        
        Args:
            profiling_backtracks: Número de backtracks para profiling inicial
        """
        self.profiling_backtracks = profiling_backtracks
        self.metrics = ProfilingMetrics()
        self.recommendations: Optional[ProfilingRecommendations] = None
        
        # Timers
        self._timers: Dict[str, float] = {}
        self._timer_stack: List[str] = []
    
    def start_timer(self, name: str):
        """Inicia un timer."""
        self._timer_stack.append(name)
        self._timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str):
        """Detiene un timer y acumula tiempo."""
        if name not in self._timers:
            return
        
        elapsed = time.perf_counter() - self._timers[name]
        
        if name == "propagation":
            self.metrics.time_in_propagation += elapsed
        elif name == "search":
            self.metrics.time_in_search += elapsed
        elif name == "heuristics":
            self.metrics.time_in_heuristics += elapsed
        elif name == "energy":
            self.metrics.time_in_energy += elapsed
        
        if self._timer_stack and self._timer_stack[-1] == name:
            self._timer_stack.pop()
    
    def record_problem_characteristics(
        self,
        num_variables: int,
        num_constraints: int,
        domain_sizes: List[int],
        has_soft: bool,
        has_hierarchy: bool
    ):
        """
        Registra características del problema.
        
        Args:
            num_variables: Número de variables
            num_constraints: Número de restricciones
            domain_sizes: Tamaños de dominios
            has_soft: Si tiene restricciones SOFT
            has_hierarchy: Si tiene jerarquía de restricciones
        """
        self.metrics.num_variables = num_variables
        self.metrics.num_constraints = num_constraints
        self.metrics.avg_domain_size = sum(domain_sizes) / len(domain_sizes) if domain_sizes else 0
        self.metrics.max_domain_size = max(domain_sizes) if domain_sizes else 0
        self.metrics.has_soft_constraints = has_soft
        self.metrics.has_hierarchy = has_hierarchy
        
        # Densidad del grafo de restricciones
        max_constraints = num_variables * (num_variables - 1) / 2
        self.metrics.constraint_density = num_constraints / max_constraints if max_constraints > 0 else 0
    
    def record_backtrack(self):
        """Registra un backtrack."""
        self.metrics.num_backtracks += 1
    
    def record_propagation(self, values_eliminated: int):
        """
        Registra una propagación.
        
        Args:
            values_eliminated: Número de valores eliminados
        """
        self.metrics.num_propagations += 1
        
        # Actualizar efectividad de propagación
        total_eliminated = (self.metrics.propagation_effectiveness * 
                           (self.metrics.num_propagations - 1) + values_eliminated)
        self.metrics.propagation_effectiveness = total_eliminated / self.metrics.num_propagations
    
    def record_constraint_evaluation(self):
        """Registra una evaluación de restricción."""
        self.metrics.num_constraint_evaluations += 1
    
    def record_energy_computation(self):
        """Registra un cálculo de energía."""
        self.metrics.num_energy_computations += 1
    
    def should_continue_profiling(self) -> bool:
        """
        Determina si debe continuar profiling.
        
        Returns:
            True si debe continuar profiling
        """
        return self.metrics.num_backtracks < self.profiling_backtracks
    
    def analyze_and_recommend(self) -> ProfilingRecommendations:
        """
        Analiza métricas y genera recomendaciones.
        
        Returns:
            Recomendaciones de optimización
        """
        rec = ProfilingRecommendations()
        
        # Calcular calidad de heurísticas
        if self.metrics.num_backtracks > 0:
            total_nodes = self.metrics.num_backtracks * 2  # Aproximación
            self.metrics.heuristic_quality = 1 - (self.metrics.num_backtracks / total_nodes)
        
        # Análisis de tamaño del problema
        is_small = self.metrics.num_variables < 20
        is_medium = 20 <= self.metrics.num_variables < 100
        is_large = self.metrics.num_variables >= 100
        
        has_large_domains = self.metrics.max_domain_size > 50
        is_dense = self.metrics.constraint_density > 0.3
        
        # Decisión de nivel de optimización
        if is_small and not self.metrics.has_soft_constraints:
            rec.optimization_level = OptimizationLevel.LITE
            rec.reasons.append("Problema pequeño sin restricciones SOFT")
        elif is_medium or self.metrics.has_soft_constraints:
            rec.optimization_level = OptimizationLevel.MEDIUM
            rec.reasons.append("Problema mediano o con restricciones SOFT")
        else:
            rec.optimization_level = OptimizationLevel.FULL
            rec.reasons.append("Problema grande o complejo")
        
        # Decisión sobre AC-3
        if self.metrics.propagation_effectiveness > 1.0:
            rec.use_ac3 = True
            rec.propagation_level = "AC3"
            rec.reasons.append(f"Propagación efectiva ({self.metrics.propagation_effectiveness:.1f} valores/prop)")
        elif is_small:
            rec.use_ac3 = True
            rec.propagation_level = "FC"
            rec.reasons.append("Forward Checking suficiente para problema pequeño")
        else:
            rec.use_ac3 = True
            rec.propagation_level = "AC3"
        
        # Decisión sobre TMS
        if self.metrics.num_backtracks > 50 and not is_small:
            rec.use_tms = True
            rec.reasons.append(f"Muchos backtracks ({self.metrics.num_backtracks}), TMS puede ayudar")
        elif is_small:
            rec.use_tms = False
            rec.reasons.append("Problema pequeño, overhead de TMS no vale la pena")
        else:
            rec.use_tms = True
        
        # Decisión sobre HomotopyRules
        if self.metrics.has_hierarchy or is_large:
            rec.use_homotopy_rules = True
            rec.reasons.append("Jerarquía de restricciones o problema grande")
        elif is_small:
            rec.use_homotopy_rules = False
            rec.reasons.append("Problema pequeño, overhead de HomotopyRules no vale la pena")
        else:
            rec.use_homotopy_rules = False
        
        # Decisión sobre Energy Caching
        if self.metrics.num_energy_computations > 100:
            rec.use_energy_caching = True
            rec.reasons.append(f"Muchos cálculos de energía ({self.metrics.num_energy_computations})")
        else:
            rec.use_energy_caching = False
        
        # Decisión sobre Sparse Sets
        if has_large_domains or is_large:
            rec.use_sparse_sets = True
            rec.reasons.append("Dominios grandes o problema grande")
        else:
            rec.use_sparse_sets = False
        
        # Decisión sobre modo de heurísticas
        if rec.optimization_level == OptimizationLevel.LITE:
            rec.heuristic_mode = "SIMPLE"
        elif rec.optimization_level == OptimizationLevel.MEDIUM:
            rec.heuristic_mode = "ENHANCED"
        else:
            rec.heuristic_mode = "ADAPTIVE"
        
        self.recommendations = rec
        return rec
    
    def get_metrics(self) -> ProfilingMetrics:
        """Retorna métricas recolectadas."""
        return self.metrics
    
    def get_recommendations(self) -> Optional[ProfilingRecommendations]:
        """Retorna recomendaciones generadas."""
        return self.recommendations
    
    def print_report(self):
        """Imprime reporte de profiling."""
        print("=" * 80)
        print("AUTO PROFILER REPORT")
        print("=" * 80)
        print()
        
        print("Problem Characteristics:")
        print(f"  Variables: {self.metrics.num_variables}")
        print(f"  Constraints: {self.metrics.num_constraints}")
        print(f"  Avg Domain Size: {self.metrics.avg_domain_size:.1f}")
        print(f"  Max Domain Size: {self.metrics.max_domain_size}")
        print(f"  Has SOFT: {self.metrics.has_soft_constraints}")
        print(f"  Has Hierarchy: {self.metrics.has_hierarchy}")
        print(f"  Constraint Density: {self.metrics.constraint_density:.2%}")
        print()
        
        print("Execution Metrics:")
        print(f"  Backtracks: {self.metrics.num_backtracks}")
        print(f"  Propagations: {self.metrics.num_propagations}")
        print(f"  Constraint Evaluations: {self.metrics.num_constraint_evaluations}")
        print(f"  Energy Computations: {self.metrics.num_energy_computations}")
        print()
        
        print("Time Distribution:")
        total_time = (self.metrics.time_in_propagation + self.metrics.time_in_search +
                     self.metrics.time_in_heuristics + self.metrics.time_in_energy)
        if total_time > 0:
            print(f"  Propagation: {self.metrics.time_in_propagation:.3f}s ({self.metrics.time_in_propagation/total_time:.1%})")
            print(f"  Search: {self.metrics.time_in_search:.3f}s ({self.metrics.time_in_search/total_time:.1%})")
            print(f"  Heuristics: {self.metrics.time_in_heuristics:.3f}s ({self.metrics.time_in_heuristics/total_time:.1%})")
            print(f"  Energy: {self.metrics.time_in_energy:.3f}s ({self.metrics.time_in_energy/total_time:.1%})")
        print()
        
        if self.recommendations:
            print("Recommendations:")
            print(f"  Optimization Level: {self.recommendations.optimization_level.name}")
            print(f"  Use AC-3: {self.recommendations.use_ac3}")
            print(f"  Use TMS: {self.recommendations.use_tms}")
            print(f"  Use HomotopyRules: {self.recommendations.use_homotopy_rules}")
            print(f"  Use Energy Caching: {self.recommendations.use_energy_caching}")
            print(f"  Use Sparse Sets: {self.recommendations.use_sparse_sets}")
            print(f"  Propagation Level: {self.recommendations.propagation_level}")
            print(f"  Heuristic Mode: {self.recommendations.heuristic_mode}")
            print()
            print("  Reasons:")
            for reason in self.recommendations.reasons:
                print(f"    - {reason}")
        
        print("=" * 80)

