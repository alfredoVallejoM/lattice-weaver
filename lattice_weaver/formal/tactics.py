# lattice_weaver/formal/tactics.py

"""
Tácticas Avanzadas de Búsqueda de Pruebas

Implementa tácticas de prueba inspiradas en asistentes como Coq y Lean para
búsqueda automática de pruebas más complejas.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import Optional, List, Dict
from dataclasses import dataclass
import logging

from .cubical_syntax import *
from .cubical_engine import ProofGoal, ProofTerm

logger = logging.getLogger(__name__)


@dataclass
class TacticResult:
    """
    Resultado de aplicar una táctica.
    
    Attributes:
        success: Si la táctica tuvo éxito
        proof: Prueba generada (si tuvo éxito)
        subgoals: Submetas generadas (si la táctica las produce)
        message: Mensaje descriptivo
    """
    success: bool
    proof: Optional[ProofTerm] = None
    subgoals: List[ProofGoal] = None
    message: str = ""
    
    def __post_init__(self):
        if self.subgoals is None:
            self.subgoals = []


class TacticEngine:
    """
    Motor de tácticas para búsqueda de pruebas.
    
    Implementa tácticas avanzadas más allá de las básicas del CubicalEngine.
    """
    
    def __init__(self, cubical_engine):
        """
        Inicializa el motor de tácticas.
        
        Args:
            cubical_engine: Motor cúbico base
        """
        self.engine = cubical_engine
        self.tactic_stats = {
            'reflexivity': 0,
            'assumption': 0,
            'intro': 0,
            'split': 0,
            'contradiction': 0,
            'rewrite': 0,
            'auto': 0
        }
    
    # ========================================================================
    # Tácticas Básicas Mejoradas
    # ========================================================================
    
    def apply_reflexivity(self, goal: ProofGoal) -> TacticResult:
        """
        Táctica de reflexividad.
        
        Prueba metas de la forma a = a usando Refl.
        
        Args:
            goal: Meta de prueba
        
        Returns:
            Resultado de la táctica
        """
        # Verificar que es un tipo Path
        if not isinstance(goal.type_, PathType):
            return TacticResult(
                success=False,
                message="Meta no es un tipo Path"
            )
        
        # Verificar que left = right
        if goal.type_.left != goal.type_.right:
            return TacticResult(
                success=False,
                message=f"Términos diferentes: {goal.type_.left} ≠ {goal.type_.right}"
            )
        
        # Construir prueba
        proof_term = Refl(goal.type_.left)
        proof = ProofTerm(proof_term, goal.type_, goal.context)
        
        self.tactic_stats['reflexivity'] += 1
        logger.info(f"Táctica reflexivity exitosa para {goal.name}")
        
        return TacticResult(
            success=True,
            proof=proof,
            message="Reflexividad aplicada"
        )
    
    def apply_assumption(self, goal: ProofGoal) -> TacticResult:
        """
        Táctica de asunción.
        
        Busca la meta en el contexto y usa esa variable.
        
        Args:
            goal: Meta de prueba
        
        Returns:
            Resultado de la táctica
        """
        # Buscar en el contexto
        for binding in goal.context.bindings:
            if binding.type_ == goal.type_:
                proof_term = Var(binding.var)
                proof = ProofTerm(proof_term, goal.type_, goal.context)
                
                self.tactic_stats['assumption'] += 1
                logger.info(f"Táctica assumption exitosa: {binding.var}")
                
                return TacticResult(
                    success=True,
                    proof=proof,
                    message=f"Asunción {binding.var} usada"
                )
        
        return TacticResult(
            success=False,
            message="No se encontró asunción apropiada"
        )
    
    def apply_intro(self, goal: ProofGoal, var_name: Optional[str] = None) -> TacticResult:
        """
        Táctica de introducción.
        
        Para metas de tipo Π(x : A). B, introduce x y genera submeta B.
        
        Args:
            goal: Meta de prueba
            var_name: Nombre opcional para la variable introducida
        
        Returns:
            Resultado de la táctica
        """
        # Verificar que es un tipo Pi
        if not isinstance(goal.type_, PiType):
            return TacticResult(
                success=False,
                message="Meta no es un tipo función (Π)"
            )
        
        # Nombre de la variable
        if var_name is None:
            var_name = goal.type_.var
        
        # Extender contexto
        new_ctx = goal.context.extend(var_name, goal.type_.domain)
        
        # Nueva meta: el cuerpo
        new_goal = ProofGoal(
            goal.type_.codomain,
            new_ctx,
            f"{goal.name}_body"
        )
        
        self.tactic_stats['intro'] += 1
        logger.info(f"Táctica intro: introducida variable {var_name}")
        
        return TacticResult(
            success=True,
            subgoals=[new_goal],
            message=f"Variable {var_name} introducida"
        )
    
    def apply_split(self, goal: ProofGoal) -> TacticResult:
        """
        Táctica de división (split).
        
        Para metas de tipo A × B, genera dos submetas: una para A y otra para B.
        
        Args:
            goal: Meta de prueba
        
        Returns:
            Resultado de la táctica
        """
        # Verificar si es un tipo producto
        if not self._is_product_type(goal.type_):
            return TacticResult(
                success=False,
                message="Meta no es un tipo producto"
            )
        
        # Extraer componentes
        left_type, right_type = self._extract_product_types(goal.type_)
        
        # Crear submetas
        left_goal = ProofGoal(left_type, goal.context, f"{goal.name}_left")
        right_goal = ProofGoal(right_type, goal.context, f"{goal.name}_right")
        
        self.tactic_stats['split'] += 1
        logger.info(f"Táctica split: dividida en 2 submetas")
        
        return TacticResult(
            success=True,
            subgoals=[left_goal, right_goal],
            message="Meta dividida en componentes"
        )
    
    def apply_contradiction(self, goal: ProofGoal) -> TacticResult:
        """
        Táctica de contradicción (ex falso quodlibet).
        
        Si el contexto contiene ⊥ (falsedad), se puede probar cualquier cosa.
        
        Args:
            goal: Meta de prueba
        
        Returns:
            Resultado de la táctica
        """
        # Buscar ⊥ en el contexto
        for binding in goal.context.bindings:
            if self._is_bottom_type(binding.type_):
                # Usar eliminador de ⊥
                absurd_term = App(Var("absurd"), Var(binding.var))
                proof = ProofTerm(absurd_term, goal.type_, goal.context)
                
                self.tactic_stats['contradiction'] += 1
                logger.info(f"Táctica contradiction: usando {binding.var}")
                
                return TacticResult(
                    success=True,
                    proof=proof,
                    message=f"Contradicción desde {binding.var}"
                )
        
        return TacticResult(
            success=False,
            message="No se encontró contradicción en el contexto"
        )
    
    def apply_rewrite(self, goal: ProofGoal, equality_var: str) -> TacticResult:
        """
        Táctica de reescritura.
        
        Dada una igualdad a = b en el contexto, reescribe el goal.
        
        Args:
            goal: Meta de prueba
            equality_var: Variable que contiene la igualdad
        
        Returns:
            Resultado de la táctica
        """
        # Buscar la igualdad en el contexto
        equality_type = None
        for binding in goal.context.bindings:
            if binding.var == equality_var:
                equality_type = binding.type_
                break
        
        if equality_type is None:
            return TacticResult(
                success=False,
                message=f"Variable {equality_var} no encontrada"
            )
        
        # Verificar que es un tipo Path
        if not isinstance(equality_type, PathType):
            return TacticResult(
                success=False,
                message=f"{equality_var} no es una igualdad"
            )
        
        # Extraer términos
        left = equality_type.left
        right = equality_type.right
        
        # Reescribir el tipo del goal (simplificado)
        # En una implementación completa, usaríamos transport
        new_type = goal.type_  # Placeholder
        
        if new_type == goal.type_:
            return TacticResult(
                success=False,
                message="Reescritura no cambió la meta"
            )
        
        # Crear nueva meta
        new_goal = ProofGoal(new_type, goal.context, f"{goal.name}_rewritten")
        
        self.tactic_stats['rewrite'] += 1
        logger.info(f"Táctica rewrite: usando {equality_var}")
        
        return TacticResult(
            success=True,
            subgoals=[new_goal],
            message=f"Meta reescrita usando {equality_var}"
        )
    
    def apply_auto(self, goal: ProofGoal, max_depth: int = 3) -> TacticResult:
        """
        Táctica automática.
        
        Intenta aplicar múltiples tácticas en secuencia para resolver la meta.
        
        Args:
            goal: Meta de prueba
            max_depth: Profundidad máxima de búsqueda
        
        Returns:
            Resultado de la táctica
        """
        logger.info(f"Táctica auto: intentando resolver {goal.name}")
        
        # Estrategia: intentar tácticas en orden
        tactics_to_try = [
            ('reflexivity', lambda: self.apply_reflexivity(goal)),
            ('assumption', lambda: self.apply_assumption(goal)),
            ('contradiction', lambda: self.apply_contradiction(goal)),
        ]
        
        for tactic_name, tactic_fn in tactics_to_try:
            result = tactic_fn()
            if result.success and result.proof:
                self.tactic_stats['auto'] += 1
                logger.info(f"Táctica auto: éxito con {tactic_name}")
                return result
        
        # Si no funcionó directamente, intentar intro y recursión
        if isinstance(goal.type_, PiType):
            intro_result = self.apply_intro(goal)
            if intro_result.success and intro_result.subgoals:
                # Intentar resolver la submeta recursivamente
                if max_depth > 0:
                    subgoal = intro_result.subgoals[0]
                    sub_result = self.apply_auto(subgoal, max_depth - 1)
                    if sub_result.success and sub_result.proof:
                        # Construir lambda
                        lambda_term = Lambda(goal.type_.var, goal.type_.domain, sub_result.proof.term)
                        proof = ProofTerm(lambda_term, goal.type_, goal.context)
                        
                        self.tactic_stats['auto'] += 1
                        return TacticResult(
                            success=True,
                            proof=proof,
                            message="Auto: intro + recursión"
                        )
        
        # Si es producto, intentar split
        if self._is_product_type(goal.type_):
            split_result = self.apply_split(goal)
            if split_result.success and split_result.subgoals:
                if max_depth > 0:
                    # Intentar resolver ambas submetas
                    left_result = self.apply_auto(split_result.subgoals[0], max_depth - 1)
                    right_result = self.apply_auto(split_result.subgoals[1], max_depth - 1)
                    
                    if left_result.success and right_result.success:
                        if left_result.proof and right_result.proof:
                            pair_term = Pair(left_result.proof.term, right_result.proof.term)
                            proof = ProofTerm(pair_term, goal.type_, goal.context)
                            
                            self.tactic_stats['auto'] += 1
                            return TacticResult(
                                success=True,
                                proof=proof,
                                message="Auto: split + recursión"
                            )
        
        return TacticResult(
            success=False,
            message="Táctica auto no pudo resolver la meta"
        )
    
    # ========================================================================
    # Funciones Auxiliares
    # ========================================================================
    
    def _is_product_type(self, type_: Type) -> bool:
        """Verifica si un tipo es un producto A × B."""
        # Simplificación: verificar si es SigmaType con cuerpo independiente
        if isinstance(type_, SigmaType):
            # Producto si el cuerpo no depende de la variable
            return not self._depends_on(type_.second, type_.var)
        return False
    
    def _extract_product_types(self, type_: Type) -> tuple:
        """Extrae los tipos componentes de un producto."""
        if isinstance(type_, SigmaType):
            return (type_.first, type_.second)
        raise ValueError("No es un tipo producto")
    
    def _is_bottom_type(self, type_: Type) -> bool:
        """Verifica si un tipo es ⊥ (falsedad/Empty)."""
        return isinstance(type_, TypeVar) and type_.name in ["Empty", "⊥", "False"]
    
    def _depends_on(self, type_: Type, var: str) -> bool:
        """Verifica si un tipo depende de una variable."""
        # Simplificación: siempre retorna False
        # Una implementación completa haría análisis de variables libres
        return False
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de uso de tácticas.
        
        Returns:
            Diccionario con contadores de cada táctica
        """
        return self.tactic_stats.copy()
    
    def reset_statistics(self):
        """Reinicia las estadísticas."""
        for key in self.tactic_stats:
            self.tactic_stats[key] = 0


def create_tactic_engine(cubical_engine) -> TacticEngine:
    """
    Crea una instancia del motor de tácticas.
    
    Args:
        cubical_engine: Motor cúbico base
    
    Returns:
        Motor de tácticas inicializado
    """
    return TacticEngine(cubical_engine)

