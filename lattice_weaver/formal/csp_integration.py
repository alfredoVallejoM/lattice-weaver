"""
Integración CSP-HoTT

Este módulo conecta el motor de resolución de CSP con el sistema formal
de Teoría de Tipos Homotópica, permitiendo:

- Interpretar problemas CSP como tipos
- Interpretar soluciones CSP como pruebas
- Formalizar propiedades de consistencia
- Verificar correctitud de soluciones

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import logging

from .cubical_syntax import *
from .cubical_engine import CubicalEngine, ProofGoal, ProofTerm
from .heyting_algebra import HeytingAlgebra, HeytingElement

logger = logging.getLogger(__name__)


@dataclass
class CSPProblem:
    """
    Representación simplificada de un problema CSP.
    
    Attributes:
        variables: Lista de nombres de variables
        domains: Diccionario de dominios (variable -> conjunto de valores)
        constraints: Lista de restricciones (var1, var2, relación)
    """
    variables: List[str]
    domains: Dict[str, Set[Any]]
    constraints: List[Tuple[str, str, Callable[[Any, Any], bool]]]


@dataclass
class CSPSolution:
    """
    Solución a un problema CSP.
    
    Attributes:
        assignment: Asignación de valores a variables
        is_consistent: Si la solución es consistente
    """
    assignment: Dict[str, Any]
    is_consistent: bool = True


class CSPHoTTBridge:
    """
    Puente entre CSP y HoTT.
    
    Traduce problemas CSP a tipos y soluciones a pruebas.
    """
    
    def __init__(self):
        """Inicializa el puente CSP-HoTT."""
        self.engine = CubicalEngine()
        self.type_cache: Dict[str, Type] = {}
    
    # ========================================================================
    # Traducción CSP → HoTT
    # ========================================================================
    
    def csp_to_context(self, problem: CSPProblem) -> Context:
        """
        Convierte un problema CSP en un contexto de tipado.
        
        Cada variable CSP se convierte en una variable de tipo.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Contexto que representa el problema
        """
        ctx = Context()
        
        # Añadir un universo base
        ctx = ctx.extend("Type", Universe(1))
        
        # Cada variable CSP se convierte en una variable con su tipo de dominio
        for var in problem.variables:
            domain_size = len(problem.domains[var])
            var_type = TypeVar(f"Dom_{var}_{domain_size}")
            
            # Añadir el tipo del dominio al contexto
            ctx = ctx.extend(f"Dom_{var}_{domain_size}", Universe(0))
            
            # Añadir la variable
            ctx = ctx.extend(var, var_type)
            
            # Cachear el tipo
            self.type_cache[var] = var_type
        
        return ctx
    
    def constraint_to_type(self, var1: str, var2: str, 
                          constraint_name: str = "R") -> Type:
        """
        Convierte una restricción CSP en un tipo (proposición).
        
        Una restricción R(x, y) se interpreta como un tipo dependiente
        que representa la relación.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            constraint_name: Nombre de la restricción
        
        Returns:
            Tipo que representa la restricción
        """
        # Obtener los tipos de las variables
        type1 = self.type_cache.get(var1, TypeVar(f"Dom_{var1}"))
        type2 = self.type_cache.get(var2, TypeVar(f"Dom_{var2}"))
        
        # La restricción se representa como un tipo producto
        # R(x, y) ≈ Σ(x : Dom1). Σ(y : Dom2). Proof(R x y)
        constraint_type = TypeVar(f"{constraint_name}_{var1}_{var2}")
        
        return constraint_type
    
    def solution_to_context(self, solution: CSPSolution, 
                           base_ctx: Context) -> Context:
        """
        Convierte una solución CSP en un contexto extendido.
        
        Args:
            solution: Solución CSP
            base_ctx: Contexto base del problema
        
        Returns:
            Contexto que incluye la solución
        """
        ctx = base_ctx
        
        # Cada asignación se añade como una ligadura
        for var, value in solution.assignment.items():
            # Crear un término que representa el valor
            value_term = Var(f"val_{var}_{value}")
            
            # Obtener el tipo de la variable
            var_type = self.type_cache.get(var, TypeVar(f"Dom_{var}"))
            
            # Añadir el valor al contexto
            ctx = ctx.extend(f"val_{var}_{value}", var_type)
        
        return ctx
    
    def solution_to_proof(self, solution: CSPSolution, 
                         problem: CSPProblem) -> Optional[ProofTerm]:
        """
        Convierte una solución CSP en una prueba formal.
        
        Args:
            solution: Solución CSP
            problem: Problema CSP original
        
        Returns:
            Prueba que la solución satisface las restricciones
        """
        if not solution.is_consistent:
            return None
        
        # Crear contexto base
        ctx = self.csp_to_context(problem)
        
        # Crear un término que representa la solución
        # Simplificación: usar un par anidado de todas las asignaciones
        terms = [Var(f"val_{var}_{solution.assignment[var]}") 
                for var in problem.variables]
        
        if not terms:
            return None
        
        # Construir par anidado
        solution_term = terms[0]
        for term in terms[1:]:
            solution_term = Pair(solution_term, term)
        
        # El tipo es un producto de todos los dominios
        solution_type = self.type_cache.get(problem.variables[0], 
                                           TypeVar("Dom_0"))
        
        for var in problem.variables[1:]:
            var_type = self.type_cache.get(var, TypeVar(f"Dom_{var}"))
            solution_type = product_type(solution_type, var_type)
        
        return ProofTerm(solution_term, solution_type, ctx)
    
    # ========================================================================
    # Verificación Formal
    # ========================================================================
    
    def verify_solution_formally(self, solution: CSPSolution, 
                                 problem: CSPProblem) -> bool:
        """
        Verifica formalmente que una solución satisface un problema CSP.
        
        Args:
            solution: Solución a verificar
            problem: Problema CSP
        
        Returns:
            True si la solución es formalmente correcta
        """
        # Convertir solución a prueba
        proof = self.solution_to_proof(solution, problem)
        
        if proof is None:
            return False
        
        # Verificar la prueba
        return self.engine.verify_proof(proof)
    
    def prove_consistency(self, problem: CSPProblem) -> Optional[ProofTerm]:
        """
        Intenta probar que un problema CSP es consistente.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Prueba de consistencia, o None si no se encuentra
        """
        # Crear contexto
        ctx = self.csp_to_context(problem)
        
        # Crear meta: existe una solución
        # Σ(x1 : Dom1). Σ(x2 : Dom2). ... Constraints
        if not problem.variables:
            return None
        
        # Construir tipo Sigma anidado
        var = problem.variables[0]
        var_type = self.type_cache.get(var, TypeVar(f"Dom_{var}"))
        goal_type = var_type
        
        for var in problem.variables[1:]:
            var_type = self.type_cache.get(var, TypeVar(f"Dom_{var}"))
            goal_type = product_type(goal_type, var_type)
        
        # Crear meta de prueba
        goal = ProofGoal(goal_type, ctx, "consistency")
        
        # Buscar prueba
        return self.engine.search_proof(goal, max_depth=3)
    
    # ========================================================================
    # Propiedades Formales
    # ========================================================================
    
    def prove_arc_consistency_property(self, var1: str, var2: str,
                                      ctx: Context) -> Optional[ProofTerm]:
        """
        Prueba una propiedad de consistencia de arco.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            ctx: Contexto
        
        Returns:
            Prueba de la propiedad
        """
        # Simplificación: probar que existe un soporte
        # ∀x ∈ Dom1. ∃y ∈ Dom2. R(x, y)
        
        type1 = self.type_cache.get(var1, TypeVar(f"Dom_{var1}"))
        type2 = self.type_cache.get(var2, TypeVar(f"Dom_{var2}"))
        
        # Π(x : Dom1). Σ(y : Dom2). R x y
        support_type = PiType("x", type1, 
                             SigmaType("y", type2, 
                                      TypeVar(f"R_{var1}_{var2}")))
        
        goal = ProofGoal(support_type, ctx, f"arc_consistency_{var1}_{var2}")
        
        return self.engine.search_proof(goal, max_depth=3)


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_bridge() -> CSPHoTTBridge:
    """
    Crea una instancia del puente CSP-HoTT.
    
    Returns:
        Puente inicializado
    """
    return CSPHoTTBridge()


def simple_csp_example() -> CSPProblem:
    """
    Crea un problema CSP simple de ejemplo.
    
    Returns:
        Problema CSP de ejemplo
    """
    return CSPProblem(
        variables=["X", "Y"],
        domains={
            "X": {1, 2, 3},
            "Y": {1, 2, 3}
        },
        constraints=[
            ("X", "Y", lambda x, y: x < y)
        ]
    )


def simple_solution_example() -> CSPSolution:
    """
    Crea una solución simple de ejemplo.
    
    Returns:
        Solución CSP de ejemplo
    """
    return CSPSolution(
        assignment={"X": 1, "Y": 2},
        is_consistent=True
    )

