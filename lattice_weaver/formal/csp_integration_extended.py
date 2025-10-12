# lattice_weaver/formal/csp_integration_extended.py

"""
Integración Completa CSP-HoTT - Fase: Integración Sistema Formal

Extiende CSPHoTTBridge con traducción completa de problemas CSP a tipos HoTT
y conversión de soluciones a pruebas formales.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import logging

from .cubical_syntax import *
from .cubical_engine import CubicalEngine, ProofGoal, ProofTerm
from .csp_integration import CSPProblem, CSPSolution, CSPHoTTBridge

logger = logging.getLogger(__name__)


class ExtendedCSPHoTTBridge(CSPHoTTBridge):
    """
    Puente extendido CSP-HoTT con traducción completa.
    
    Extiende CSPHoTTBridge con:
    - Traducción completa de problemas CSP a tipos Sigma anidados
    - Conversión de soluciones a pruebas formales
    - Construcción de tipos para restricciones
    """
    
    def __init__(self):
        """Inicializa el puente extendido."""
        super().__init__()
        self.constraint_cache: Dict[Tuple[str, str], Type] = {}
    
    def translate_csp_to_type(self, problem: CSPProblem) -> Type:
        """
        Traduce un problema CSP completo a un tipo HoTT.
        
        El tipo resultante representa la proposición:
        "Existe una asignación que satisface todas las restricciones"
        
        Σ(x1 : Dom1). Σ(x2 : Dom2). ... Σ(xn : Domn). 
          (C1 x1 x2) × (C2 x2 x3) × ... × (Ck xn-1 xn)
        
        Args:
            problem: Problema CSP
        
        Returns:
            Tipo HoTT que representa el problema
        """
        if not problem.variables:
            return Universe(0)
        
        # Construir tipo Sigma anidado para las variables
        var_types = []
        for var in problem.variables:
            domain_type = self._domain_to_type(problem.domains[var], var)
            var_types.append((var, domain_type))
        
        # Construir tipo producto para las restricciones
        constraint_types = []
        for var1, var2, relation in problem.constraints:
            constraint_type = self._constraint_to_type(var1, var2, relation)
            constraint_types.append(constraint_type)
        
        # Combinar: Σ(variables). Π(constraints)
        result_type = self._build_nested_sigma(var_types, constraint_types)
        
        logger.info(f"Traducido CSP a tipo: {result_type}")
        
        return result_type
    
    def _domain_to_type(self, domain: Set, var_name: str) -> Type:
        """
        Convierte un dominio CSP en un tipo HoTT.
        
        Estrategias:
        1. Dominio finito pequeño → Tipo suma (coproducto)
        2. Dominio finito grande → Tipo abstracto con axiomas
        3. Dominio infinito → Tipo base (Nat, Int, etc.)
        
        Args:
            domain: Conjunto de valores del dominio
            var_name: Nombre de la variable
        
        Returns:
            Tipo que representa el dominio
        """
        domain_list = list(domain)
        
        if len(domain_list) <= 10:
            # Tipo suma: Dom = v1 + v2 + ... + vn
            # En HoTT: Σ(i : Fin n). Unit
            type_name = f"Dom_{var_name}_{len(domain_list)}"
            return TypeVar(type_name)
        else:
            # Tipo abstracto
            type_name = f"Dom_{var_name}"
            return TypeVar(type_name)
    
    def _constraint_to_type(self, var1: str, var2: str, relation: Callable) -> Type:
        """
        Convierte una restricción en un tipo (proposición).
        
        Una restricción R(x, y) se interpreta como:
        Π(x : Dom1). Π(y : Dom2). R x y → Type
        
        Donde R x y es un tipo habitado si la restricción se satisface.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            relation: Función de relación
        
        Returns:
            Tipo que representa la restricción
        """
        # Buscar en caché
        key = (var1, var2)
        if key in self.constraint_cache:
            return self.constraint_cache[key]
        
        # Obtener tipos de las variables
        type1 = self.type_cache.get(var1, TypeVar(f"Dom_{var1}"))
        type2 = self.type_cache.get(var2, TypeVar(f"Dom_{var2}"))
        
        # Crear tipo función: Dom1 → Dom2 → Type
        # (representa la relación)
        relation_type = PiType(
            var1, type1,
            PiType(var2, type2, Universe(0))
        )
        
        # Cachear
        self.constraint_cache[key] = relation_type
        
        return relation_type
    
    def _build_nested_sigma(self, var_types: List[Tuple[str, Type]], 
                           constraint_types: List[Type]) -> Type:
        """
        Construye un tipo Sigma anidado con restricciones.
        
        Σ(x1 : T1). Σ(x2 : T2). ... Σ(xn : Tn). C1 × C2 × ... × Ck
        
        Args:
            var_types: Lista de (nombre_variable, tipo)
            constraint_types: Lista de tipos de restricciones
        
        Returns:
            Tipo Sigma anidado
        """
        if not var_types:
            # Solo restricciones
            return self._build_product_type(constraint_types)
        
        # Construir Sigma anidado
        var_name, var_type = var_types[0]
        rest_vars = var_types[1:]
        
        if rest_vars or constraint_types:
            # Tipo del cuerpo: resto de variables + restricciones
            body_type = self._build_nested_sigma(rest_vars, constraint_types)
            return SigmaType(var_name, var_type, body_type)
        else:
            # Última variable
            return var_type
    
    def _build_product_type(self, types: List[Type]) -> Type:
        """
        Construye un tipo producto: T1 × T2 × ... × Tn
        
        Args:
            types: Lista de tipos
        
        Returns:
            Tipo producto
        """
        if not types:
            return Universe(0)  # Unit type
        if len(types) == 1:
            return types[0]
        
        # Producto anidado usando product_type
        result = types[0]
        for t in types[1:]:
            result = product_type(result, t)
        
        return result
    
    def solution_to_proof_complete(self, solution: CSPSolution, 
                                   problem: CSPProblem) -> Optional[ProofTerm]:
        """
        Convierte una solución CSP en una prueba formal completa.
        
        La prueba debe:
        1. Proporcionar valores para todas las variables
        2. Demostrar que cada restricción se satisface
        
        Args:
            solution: Solución CSP
            problem: Problema CSP original
        
        Returns:
            Prueba formal o None si la solución es inválida
        """
        if not solution.is_consistent:
            logger.warning("Solución inconsistente, no se puede generar prueba")
            return None
        
        # Crear contexto
        ctx = self.csp_to_context(problem)
        
        # Construir término para la solución
        # Σ(x1 : Dom1). ... → (val1, (val2, (..., proofs)))
        
        # 1. Valores de variables
        value_terms = []
        for var in problem.variables:
            if var not in solution.assignment:
                logger.warning(f"Variable {var} no asignada en solución")
                return None
            
            value = solution.assignment[var]
            value_term = Var(f"val_{var}_{value}")
            value_terms.append(value_term)
        
        # 2. Pruebas de restricciones
        constraint_proofs = []
        for var1, var2, relation in problem.constraints:
            val1 = solution.assignment.get(var1)
            val2 = solution.assignment.get(var2)
            
            if val1 is None or val2 is None:
                logger.warning(f"Variables {var1} o {var2} no asignadas")
                return None
            
            # Verificar que la restricción se satisface
            if not relation(val1, val2):
                logger.warning(f"Restricción {var1}={val1}, {var2}={val2} no satisfecha")
                return None  # Solución inválida
            
            # Crear prueba (por ahora, un axioma)
            proof_term = Var(f"proof_{var1}_{var2}_{val1}_{val2}")
            constraint_proofs.append(proof_term)
        
        # Combinar en un término Sigma anidado
        solution_term = self._build_nested_pair(value_terms + constraint_proofs)
        
        # Tipo de la solución
        solution_type = self.translate_csp_to_type(problem)
        
        logger.info(f"Generada prueba para solución: {solution.assignment}")
        
        return ProofTerm(solution_term, solution_type, ctx)
    
    def _build_nested_pair(self, terms: List[Term]) -> Term:
        """
        Construye un par anidado: (t1, (t2, (..., tn)))
        
        Args:
            terms: Lista de términos
        
        Returns:
            Par anidado
        """
        if not terms:
            return Var("unit")  # Término unit
        if len(terms) == 1:
            return terms[0]
        
        # Par anidado desde el final
        result = terms[-1]
        for t in reversed(terms[:-1]):
            result = Pair(t, result)
        
        return result
    
    def verify_solution_type_checks(self, solution: CSPSolution, 
                                    problem: CSPProblem) -> bool:
        """
        Verifica que la prueba generada para una solución type-checks.
        
        Args:
            solution: Solución CSP
            problem: Problema CSP
        
        Returns:
            True si la prueba es válida según el verificador de tipos
        """
        proof = self.solution_to_proof_complete(solution, problem)
        
        if proof is None:
            return False
        
        # Verificar usando el motor cúbico
        try:
            is_valid = self.engine.verify_proof(proof)
            logger.info(f"Verificación de prueba: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Error verificando prueba: {e}")
            return False
    
    def extract_constraints_as_propositions(self, problem: CSPProblem) -> List[Type]:
        """
        Extrae las restricciones del problema como proposiciones lógicas.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Lista de tipos que representan las restricciones
        """
        propositions = []
        
        for var1, var2, relation in problem.constraints:
            prop_type = self._constraint_to_type(var1, var2, relation)
            propositions.append(prop_type)
        
        return propositions
    
    def get_translation_statistics(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de la traducción.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'cached_types': len(self.type_cache),
            'cached_constraints': len(self.constraint_cache)
        }


def create_extended_bridge() -> ExtendedCSPHoTTBridge:
    """
    Crea una instancia del puente extendido CSP-HoTT.
    
    Returns:
        Puente extendido inicializado
    """
    return ExtendedCSPHoTTBridge()


# ============================================================================
# Ejemplos de uso
# ============================================================================

def example_graph_coloring_translation():
    """
    Ejemplo: Traducir problema de coloración de grafos a tipo HoTT.
    """
    # Problema: Colorear 3 nodos con 2 colores (rojo, azul)
    # Restricción: nodos adyacentes deben tener colores diferentes
    
    problem = CSPProblem(
        variables=['n1', 'n2', 'n3'],
        domains={
            'n1': {'red', 'blue'},
            'n2': {'red', 'blue'},
            'n3': {'red', 'blue'}
        },
        constraints=[
            ('n1', 'n2', lambda a, b: a != b),  # n1 y n2 adyacentes
            ('n2', 'n3', lambda a, b: a != b),  # n2 y n3 adyacentes
        ]
    )
    
    # Traducir a tipo
    bridge = create_extended_bridge()
    problem_type = bridge.translate_csp_to_type(problem)
    
    print(f"Tipo del problema: {problem_type}")
    
    # Solución válida
    solution = CSPSolution(
        assignment={'n1': 'red', 'n2': 'blue', 'n3': 'red'},
        is_consistent=True
    )
    
    # Convertir a prueba
    proof = bridge.solution_to_proof_complete(solution, problem)
    
    if proof:
        print(f"Prueba generada: {proof.term}")
        print(f"Tipo de la prueba: {proof.type}")
    
    return problem, solution, proof


def example_invalid_solution():
    """
    Ejemplo: Solución inválida no genera prueba.
    """
    problem = CSPProblem(
        variables=['x', 'y'],
        domains={'x': {1, 2}, 'y': {1, 2}},
        constraints=[('x', 'y', lambda a, b: a != b)]
    )
    
    # Solución inválida (x = y)
    invalid_solution = CSPSolution(
        assignment={'x': 1, 'y': 1},
        is_consistent=False
    )
    
    bridge = create_extended_bridge()
    proof = bridge.solution_to_proof_complete(invalid_solution, problem)
    
    assert proof is None, "Solución inválida no debe generar prueba"
    print("✓ Solución inválida correctamente rechazada")
    
    return problem, invalid_solution

