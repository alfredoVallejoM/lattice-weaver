"""
Interfaces base para el sistema de estrategias de LatticeWeaver v8.0.

Este módulo define las interfaces abstractas que todas las estrategias deben implementar.
El sistema de estrategias permite inyectar comportamiento personalizado en diferentes
fases del proceso de resolución de problemas.
"""

from abc import ABC, abstractmethod
from typing import Dict, Set, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class StrategyType(Enum):
    """Tipos de estrategias disponibles."""
    ANALYSIS = "analysis"
    HEURISTIC = "heuristic"
    PROPAGATION = "propagation"
    VERIFICATION = "verification"
    OPTIMIZATION = "optimization"


@dataclass
class AnalysisResult:
    """Resultado de una estrategia de análisis."""
    strategy_name: str
    data: Dict[str, Any]
    recommendations: List[str]
    confidence: float = 1.0
    accelerated: bool = False
    speedup: float = 1.0


@dataclass
class PropagationResult:
    """Resultado de una estrategia de propagación."""
    success: bool
    pruned_values: Dict[str, Set[Any]]
    inconsistency_detected: bool = False
    message: str = ""


@dataclass
class VerificationResult:
    """Resultado de una estrategia de verificación."""
    is_valid: bool
    properties_verified: List[str]
    properties_failed: List[str]
    formal_proof: Optional[Any] = None
    message: str = ""


@dataclass
class OptimizationResult:
    """Resultado de una estrategia de optimización."""
    should_apply: bool
    optimized_csp: Optional[Any] = None
    transformations_applied: List[str] = None
    estimated_improvement: float = 0.0


class SolverContext:
    """
    Contexto compartido durante la resolución.
    
    Almacena información recopilada por diferentes estrategias
    y permite la comunicación entre componentes.
    """
    
    def __init__(self):
        self.original_csp = None
        self.current_csp = None
        self.analysis_results = {}
        self.optimization_results = {}
        self.verification_results = {}
        self.abstraction_hierarchy = None
        self.ml_predictions = {}
        self.statistics = {}
    
    def add_analysis(self, name: str, result: AnalysisResult):
        """Añade resultado de análisis al contexto."""
        self.analysis_results[name] = result
    
    def get_analysis(self, name: str) -> Optional[AnalysisResult]:
        """Obtiene resultado de análisis por nombre."""
        return self.analysis_results.get(name)
    
    def add_optimization(self, name: str, result: OptimizationResult):
        """Añade resultado de optimización al contexto."""
        self.optimization_results[name] = result
    
    def add_verification(self, name: str, result: VerificationResult):
        """Añade resultado de verificación al contexto."""
        self.verification_results[name] = result


class AnalysisStrategy(ABC):
    """
    Interfaz base para estrategias de análisis.
    
    Las estrategias de análisis examinan el CSP antes de la resolución
    para extraer información estructural, topológica, algebraica, etc.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, csp: Any) -> AnalysisResult:
        """
        Analiza el CSP y retorna información estructural.
        
        Args:
            csp: El problema CSP a analizar
            
        Returns:
            AnalysisResult con la información extraída
        """
        pass
    
    @abstractmethod
    def is_applicable(self, csp: Any) -> bool:
        """
        Determina si esta estrategia es aplicable al CSP dado.
        
        Args:
            csp: El problema CSP
            
        Returns:
            True si la estrategia es aplicable
        """
        pass


class HeuristicStrategy(ABC):
    """
    Interfaz base para estrategias heurísticas.
    
    Las estrategias heurísticas guían la búsqueda mediante la selección
    de variables y el ordenamiento de valores.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select_variable(
        self,
        unassigned_vars: Set[str],
        domains: Dict[str, Set],
        context: SolverContext
    ) -> str:
        """
        Selecciona la próxima variable a asignar.
        
        Args:
            unassigned_vars: Variables sin asignar
            domains: Dominios actuales
            context: Contexto del solver
            
        Returns:
            Nombre de la variable seleccionada
        """
        pass
    
    @abstractmethod
    def order_values(
        self,
        variable: str,
        domain: Set,
        context: SolverContext
    ) -> List:
        """
        Ordena los valores del dominio de una variable.
        
        Args:
            variable: Variable a asignar
            domain: Dominio de la variable
            context: Contexto del solver
            
        Returns:
            Lista ordenada de valores
        """
        pass


class PropagationStrategy(ABC):
    """
    Interfaz base para estrategias de propagación.
    
    Las estrategias de propagación reducen los dominios de las variables
    basándose en las restricciones y asignaciones actuales.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def propagate(
        self,
        variable: str,
        value: Any,
        domains: Dict[str, Set],
        context: SolverContext
    ) -> PropagationResult:
        """
        Propaga la asignación de una variable.
        
        Args:
            variable: Variable asignada
            value: Valor asignado
            domains: Dominios actuales
            context: Contexto del solver
            
        Returns:
            PropagationResult con los valores podados
        """
        pass


class VerificationStrategy(ABC):
    """
    Interfaz base para estrategias de verificación.
    
    Las estrategias de verificación validan formalmente problemas
    y soluciones usando métodos formales (tipos cúbicos, lógica, etc.).
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def verify_problem(self, csp: Any) -> VerificationResult:
        """
        Verifica propiedades del problema.
        
        Args:
            csp: El problema CSP
            
        Returns:
            VerificationResult con las propiedades verificadas
        """
        pass
    
    @abstractmethod
    def verify_solution(
        self,
        csp: Any,
        solution: Dict[str, Any]
    ) -> VerificationResult:
        """
        Verifica formalmente la solución.
        
        Args:
            csp: El problema CSP
            solution: La solución propuesta
            
        Returns:
            VerificationResult indicando si la solución es válida
        """
        pass
    
    @abstractmethod
    def extract_properties(self, csp: Any) -> Dict[str, Any]:
        """
        Extrae propiedades formales del CSP.
        
        Args:
            csp: El problema CSP
            
        Returns:
            Diccionario con propiedades extraídas
        """
        pass


class OptimizationStrategy(ABC):
    """
    Interfaz base para estrategias de optimización.
    
    Las estrategias de optimización transforman el CSP antes de la
    resolución para mejorar el rendimiento (romper simetrías, etc.).
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def optimize(
        self,
        csp: Any,
        context: SolverContext
    ) -> OptimizationResult:
        """
        Optimiza el CSP.
        
        Args:
            csp: El problema CSP
            context: Contexto del solver
            
        Returns:
            OptimizationResult con el CSP optimizado
        """
        pass
    
    @abstractmethod
    def estimate_benefit(self, csp: Any) -> float:
        """
        Estima el beneficio de aplicar esta optimización.
        
        Args:
            csp: El problema CSP
            
        Returns:
            Estimación del beneficio (0-1)
        """
        pass

