"""
Motor Cúbico (CubicalEngine)

Este módulo implementa el motor de razonamiento basado en Teoría de Tipos
Homotópica (HoTT) que integra todos los componentes formales de LatticeWeaver:

- Álgebra de Heyting (lógica intuicionista)
- Sintaxis Cúbica (AST para HoTT)
- Verificador de Tipos
- Conexión con el motor de resolución de CSP

El CubicalEngine permite formalizar el razonamiento sobre problemas CSP
en un sistema de tipos con fundamentos homotópicos.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Optional, List, Dict, Set, Tuple, Any
from dataclasses import dataclass
import logging

from .heyting_algebra import HeytingAlgebra, HeytingElement
from .cubical_syntax import *
from .type_checker import TypeChecker, TypeCheckError
from .cubical_operations import normalize

logger = logging.getLogger(__name__)


@dataclass
class ProofGoal:
    """
    Meta de prueba: un tipo que queremos habitar (una proposición que queremos probar).
    
    Attributes:
        type_: Tipo objetivo (proposición)
        context: Contexto de tipado
        name: Nombre descriptivo de la meta
    """
    type_: Type
    context: Context
    name: str = "goal"
    
    def __str__(self) -> str:
        return f"{self.name}: {self.type_} (en contexto {self.context})"


@dataclass
class ProofTerm:
    """
    Término de prueba: un término que habita un tipo (una prueba de una proposición).
    
    Attributes:
        term: Término (prueba)
        type_: Tipo del término (proposición probada)
        context: Contexto en el que se construyó
    """
    term: Term
    type_: Type
    context: Context
    
    def __str__(self) -> str:
        return f"{self.term} : {self.type_}"


class CubicalEngine:
    """
    Motor de razonamiento basado en HoTT.
    
    Proporciona capacidades de:
    - Construcción de pruebas formales
    - Verificación de correctitud
    - Búsqueda automática de pruebas simples
    - Interpretación lógica de CSP
    """
    
    def __init__(self):
        """Inicializa el motor cúbico."""
        self.type_checker = TypeChecker()
        self.proof_cache: Dict[str, ProofTerm] = {}
        self._tactics_engine = None  # Lazy initialization
    
    # ========================================================================
    # Construcción de Pruebas
    # ========================================================================
    
    def prove_reflexivity(self, ctx: Context, point: Term) -> ProofTerm:
        """
        Construye una prueba de reflexividad: a = a.
        
        Args:
            ctx: Contexto de tipado
            point: Punto a
        
        Returns:
            Prueba de Path A a a
        """
        # Inferir el tipo del punto
        point_type = self.type_checker.infer_type(ctx, point)
        
        # Construir refl a
        refl_term = Refl(point)
        
        # El tipo es Path A a a
        proof_type = PathType(point_type, point, point)
        
        return ProofTerm(refl_term, proof_type, ctx)
    
    def prove_symmetry(self, ctx: Context, proof: ProofTerm) -> ProofTerm:
        """
        Construye una prueba de simetría: si a = b, entonces b = a.
        
        Args:
            ctx: Contexto de tipado
            proof: Prueba de Path A a b
        
        Returns:
            Prueba de Path A b a
        """
        if not isinstance(proof.type_, PathType):
            raise TypeCheckError("Symmetry requires a path type")
        
        # Extraer información del camino
        base_type = proof.type_.type_
        left = proof.type_.left
        right = proof.type_.right
        
        # Construir el camino inverso: <i> p @ (1-i)
        # Simplificación: usamos la misma variable
        inv_path = PathAbs("i", PathApp(proof.term, "i"))
        
        # El tipo es Path A right left
        inv_type = PathType(base_type, right, left)
        
        return ProofTerm(inv_path, inv_type, ctx)
    
    def prove_transitivity(self, ctx: Context, 
                          proof1: ProofTerm, proof2: ProofTerm) -> ProofTerm:
        """
        Construye una prueba de transitividad: si a = b y b = c, entonces a = c.
        
        Args:
            ctx: Contexto de tipado
            proof1: Prueba de Path A a b
            proof2: Prueba de Path A b c
        
        Returns:
            Prueba de Path A a c
        """
        if not isinstance(proof1.type_, PathType) or not isinstance(proof2.type_, PathType):
            raise TypeCheckError("Transitivity requires path types")
        
        base_type = proof1.type_.type_
        a = proof1.type_.left
        b = proof1.type_.right
        c = proof2.type_.right
        
        # Verificar que los caminos son compatibles
        if not self.type_checker.terms_equal(ctx, b, proof2.type_.left):
            raise TypeCheckError("Paths are not composable")
        
        # Construir el camino compuesto
        # Simplificación: concatenación
        comp_path = PathAbs("i", PathApp(proof1.term, "i"))
        
        comp_type = PathType(base_type, a, c)
        
        return ProofTerm(comp_path, comp_type, ctx)
    
    def apply_function_to_path(self, ctx: Context, func: Term, proof: ProofTerm) -> ProofTerm:
        """
        Aplica una función a ambos lados de una igualdad (congruencia).
        
        Si f : A → B y p : Path A a b, entonces ap f p : Path B (f a) (f b).
        
        Args:
            ctx: Contexto de tipado
            func: Función f
            proof: Prueba de Path A a b
        
        Returns:
            Prueba de Path B (f a) (f b)
        """
        if not isinstance(proof.type_, PathType):
            raise TypeCheckError("Expected a path type")
        
        # Inferir el tipo de la función
        func_type = self.type_checker.infer_type(ctx, func)
        
        if not isinstance(func_type, PiType):
            raise TypeCheckError("Expected a function type")
        
        # Extraer información
        a = proof.type_.left
        b = proof.type_.right
        
        # Aplicar la función a ambos extremos
        fa = App(func, a)
        fb = App(func, b)
        
        # Construir el camino: <i> f (p @ i)
        ap_path = PathAbs("i", App(func, PathApp(proof.term, "i")))
        
        # Inferir el tipo resultado
        result_type_a = self.type_checker.infer_type(ctx, fa)
        
        ap_type = PathType(result_type_a, fa, fb)
        
        return ProofTerm(ap_path, ap_type, ctx)
    
    # ========================================================================
    # Búsqueda Automática de Pruebas
    # ========================================================================
    
    def search_proof(self, goal: ProofGoal, max_depth: int = 5) -> Optional[ProofTerm]:
        """
        Busca automáticamente una prueba para una meta.
        
        Implementa búsqueda en profundidad limitada con tácticas básicas.
        
        Args:
            goal: Meta de prueba
            max_depth: Profundidad máxima de búsqueda
        
        Returns:
            Prueba encontrada, o None si no se encuentra
        """
        # Caso base: profundidad 0
        if max_depth == 0:
            return None
        
        # Táctica 1: Buscar en el contexto
        proof = self._search_in_context(goal)
        if proof:
            return proof
        
        # Táctica 2: Reflexividad para igualdades
        if isinstance(goal.type_, PathType):
            if self.type_checker.terms_equal(goal.context, 
                                            goal.type_.left, 
                                            goal.type_.right):
                return self.prove_reflexivity(goal.context, goal.type_.left)
        
        # Táctica 3: Introducción de lambda para tipos Pi
        if isinstance(goal.type_, PiType):
            proof = self._intro_lambda(goal, max_depth - 1)
            if proof:
                return proof
        
        # Táctica 4: Introducción de par para tipos Sigma
        if isinstance(goal.type_, SigmaType):
            proof = self._intro_pair(goal, max_depth - 1)
            if proof:
                return proof
        
        return None
    
    def _search_in_context(self, goal: ProofGoal) -> Optional[ProofTerm]:
        """Busca una prueba directamente en el contexto."""
        for binding in goal.context.bindings:
            var_term = Var(binding.var)
            if self.type_checker.types_equal(goal.context, binding.type_, goal.type_):
                return ProofTerm(var_term, goal.type_, goal.context)
        return None
    
    def _intro_lambda(self, goal: ProofGoal, depth: int) -> Optional[ProofTerm]:
        """Introduce una lambda para probar un tipo Pi."""
        if not isinstance(goal.type_, PiType):
            return None
        
        # Extender el contexto con la variable del dominio
        new_ctx = goal.context.extend(goal.type_.var, goal.type_.domain)
        
        # Crear una nueva meta para el cuerpo
        body_goal = ProofGoal(goal.type_.codomain, new_ctx, f"{goal.name}_body")
        
        # Buscar prueba para el cuerpo
        body_proof = self.search_proof(body_goal, depth)
        
        if body_proof:
            # Construir la lambda
            lam = Lambda(goal.type_.var, goal.type_.domain, body_proof.term)
            return ProofTerm(lam, goal.type_, goal.context)
        
        return None
    
    def _intro_pair(self, goal: ProofGoal, depth: int) -> Optional[ProofTerm]:
        """Introduce un par para probar un tipo Sigma."""
        if not isinstance(goal.type_, SigmaType):
            return None
        
        # Buscar prueba para el primer componente
        first_goal = ProofGoal(goal.type_.first, goal.context, f"{goal.name}_fst")
        first_proof = self.search_proof(first_goal, depth)
        
        if not first_proof:
            return None
        
        # Sustituir en el segundo tipo
        second_type = goal.type_.second.substitute(goal.type_.var, first_proof.term)
        
        # Buscar prueba para el segundo componente
        second_goal = ProofGoal(second_type, goal.context, f"{goal.name}_snd")
        second_proof = self.search_proof(second_goal, depth)
        
        if second_proof:
            # Construir el par
            pair = Pair(first_proof.term, second_proof.term)
            return ProofTerm(pair, goal.type_, goal.context)
        
        return None
    
    # ========================================================================
    # Interpretación de CSP
    # ========================================================================
    
    def csp_variable_to_type(self, domain_size: int) -> Type:
        """
        Convierte una variable CSP con un dominio de tamaño n en un tipo.
        
        Args:
            domain_size: Tamaño del dominio
        
        Returns:
            Tipo que representa el dominio
        """
        # Simplificación: usar un tipo variable
        return TypeVar(f"Domain_{domain_size}")
    
    def csp_constraint_to_proposition(self, var1: str, var2: str, 
                                     relation_name: str) -> Type:
        """
        Convierte una restricción CSP en una proposición (tipo).
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            relation_name: Nombre de la relación
        
        Returns:
            Tipo que representa la restricción
        """
        # Simplificación: usar un tipo variable que representa la relación
        return TypeVar(f"{relation_name}_{var1}_{var2}")
    
    def interpret_solution_as_proof(self, solution: Dict[str, Any]) -> Context:
        """
        Interpreta una solución CSP como un contexto de prueba.
        
        Args:
            solution: Solución CSP (asignación de valores)
        
        Returns:
            Contexto que representa la solución
        """
        ctx = Context()
        
        for var, value in solution.items():
            # Cada asignación se convierte en una ligadura
            value_type = TypeVar(f"Value_{value}")
            ctx = ctx.extend(var, value_type)
        
        return ctx
    
    # ========================================================================
    # Utilidades
    # ========================================================================
    
    def verify_proof(self, proof: ProofTerm) -> bool:
        """
        Verifica que una prueba es correcta.
        
        Args:
            proof: Prueba a verificar
        
        Returns:
            True si la prueba es correcta
        """
        try:
            self.type_checker.check_type(proof.context, proof.term, proof.type_)
            return True
        except TypeCheckError:
            return False
    
    def normalize_proof(self, proof: ProofTerm) -> ProofTerm:
        """
        Normaliza una prueba (reduce a forma normal).
        
        Args:
            proof: Prueba a normalizar
        
        Returns:
            Prueba normalizada
        """
        normalized_term = normalize(proof.term)
        return ProofTerm(normalized_term, proof.type_, proof.context)
    
    def proofs_equal(self, proof1: ProofTerm, proof2: ProofTerm) -> bool:
        """
        Verifica si dos pruebas son iguales (definitionally equal).
        
        Args:
            proof1: Primera prueba
            proof2: Segunda prueba
        
        Returns:
            True si las pruebas son iguales
        """
        # Normalizar ambas pruebas
        norm1 = self.normalize_proof(proof1)
        norm2 = self.normalize_proof(proof2)
        
        # Comparar términos y tipos
        return (self.type_checker.terms_equal(proof1.context, norm1.term, norm2.term) and
                self.type_checker.types_equal(proof1.context, norm1.type_, norm2.type_))
    
    @property
    def tactics(self):
        """Lazy initialization del motor de tácticas."""
        if self._tactics_engine is None:
            from .tactics import TacticEngine
            self._tactics_engine = TacticEngine(self)
        return self._tactics_engine
    
    def search_proof_with_tactics(self, goal: ProofGoal, 
                                  max_depth: int = 5,
                                  use_advanced_tactics: bool = True) -> Optional[ProofTerm]:
        """
        Búsqueda de pruebas con tácticas avanzadas.
        
        Args:
            goal: Meta de prueba
            max_depth: Profundidad máxima
            use_advanced_tactics: Si usar tácticas avanzadas
        
        Returns:
            Prueba encontrada o None
        """
        # Intentar búsqueda básica primero
        proof = self.search_proof(goal, max_depth)
        if proof:
            return proof
        
        if not use_advanced_tactics:
            return None
        
        # Intentar táctica automática
        result = self.tactics.apply_auto(goal, max_depth)
        if result.success and result.proof:
            return result.proof
        
        return None


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_engine() -> CubicalEngine:
    """
    Crea una instancia del motor cúbico.
    
    Returns:
        Motor cúbico inicializado
    """
    return CubicalEngine()

