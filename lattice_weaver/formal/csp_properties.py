# lattice_weaver/formal/csp_properties.py

"""
Verificación Formal de Propiedades CSP

Define y verifica propiedades formales de problemas CSP usando HoTT.

Propiedades verificables:
- Arc-consistencia
- Consistencia global
- Correctitud de soluciones
- Generación de invariantes

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass
import logging

from .cubical_syntax import *
from .cubical_engine import CubicalEngine, ProofGoal, ProofTerm
from .csp_integration import CSPProblem, CSPSolution
from .csp_integration_extended import ExtendedCSPHoTTBridge

logger = logging.getLogger(__name__)


@dataclass
class PropertyVerificationResult:
    """
    Resultado de verificación de una propiedad.
    
    Attributes:
        property_name: Nombre de la propiedad verificada
        is_valid: Si la propiedad es válida
        proof: Prueba formal (si existe)
        message: Mensaje descriptivo
    """
    property_name: str
    is_valid: bool
    proof: Optional[ProofTerm] = None
    message: str = ""


class CSPPropertyVerifier:
    """
    Verificador de propiedades formales de CSP.
    
    Permite definir y verificar propiedades como:
    - Consistencia
    - Completitud
    - Arc-consistencia
    - Path-consistencia
    """
    
    def __init__(self):
        """Inicializa el verificador."""
        self.engine = CubicalEngine()
        self.bridge = ExtendedCSPHoTTBridge()
        self.verification_cache: Dict[str, PropertyVerificationResult] = {}
    
    # ========================================================================
    # Verificación de Arc-Consistencia
    # ========================================================================
    
    def verify_arc_consistency(self, problem: CSPProblem, 
                               var1: str, var2: str) -> PropertyVerificationResult:
        """
        Verifica formalmente que un arco es consistente.
        
        Propiedad: ∀x ∈ Dom(var1). ∃y ∈ Dom(var2). C(x, y)
        
        Args:
            problem: Problema CSP
            var1: Primera variable del arco
            var2: Segunda variable del arco
        
        Returns:
            Resultado de verificación
        """
        property_name = f"arc_consistency_{var1}_{var2}"
        
        # Verificar en caché
        if property_name in self.verification_cache:
            return self.verification_cache[property_name]
        
        logger.info(f"Verificando arc-consistencia: {var1} → {var2}")
        
        # Buscar la restricción
        constraint_fn = None
        for v1, v2, fn in problem.constraints:
            if (v1 == var1 and v2 == var2) or (v1 == var2 and v2 == var1):
                constraint_fn = fn
                break
        
        if constraint_fn is None:
            result = PropertyVerificationResult(
                property_name=property_name,
                is_valid=False,
                message=f"No existe restricción entre {var1} y {var2}"
            )
            self.verification_cache[property_name] = result
            return result
        
        # Verificar arc-consistencia computacionalmente
        # Para cada valor en Dom(var1), debe existir al menos un valor en Dom(var2)
        dom1 = problem.domains[var1]
        dom2 = problem.domains[var2]
        
        is_arc_consistent = True
        for val1 in dom1:
            has_support = False
            for val2 in dom2:
                if constraint_fn(val1, val2):
                    has_support = True
                    break
            
            if not has_support:
                is_arc_consistent = False
                logger.warning(f"Valor {val1} de {var1} no tiene soporte en {var2}")
                break
        
        # Construir el tipo de la propiedad (simplificado)
        # En una implementación completa, construiríamos el tipo Pi/Sigma
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=is_arc_consistent,
            message=f"Arc-consistencia {var1}→{var2}: {'válida' if is_arc_consistent else 'inválida'}"
        )
        
        self.verification_cache[property_name] = result
        logger.info(result.message)
        
        return result
    
    def verify_all_arcs_consistent(self, problem: CSPProblem) -> PropertyVerificationResult:
        """
        Verifica que todos los arcos del problema son consistentes.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Resultado de verificación
        """
        property_name = "all_arcs_consistent"
        
        logger.info("Verificando consistencia de todos los arcos")
        
        all_consistent = True
        inconsistent_arcs = []
        
        for var1, var2, _ in problem.constraints:
            result = self.verify_arc_consistency(problem, var1, var2)
            if not result.is_valid:
                all_consistent = False
                inconsistent_arcs.append((var1, var2))
        
        message = "Todos los arcos son consistentes" if all_consistent else \
                  f"Arcos inconsistentes: {inconsistent_arcs}"
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=all_consistent,
            message=message
        )
        
        logger.info(message)
        return result
    
    # ========================================================================
    # Verificación de Consistencia Global
    # ========================================================================
    
    def verify_global_consistency(self, problem: CSPProblem) -> PropertyVerificationResult:
        """
        Verifica formalmente que el problema es globalmente consistente.
        
        Propiedad: ∃(x1, x2, ..., xn). C1 ∧ C2 ∧ ... ∧ Ck
        
        Args:
            problem: Problema CSP
        
        Returns:
            Resultado de verificación
        """
        property_name = "global_consistency"
        
        logger.info("Verificando consistencia global")
        
        # El tipo del problema representa la consistencia global
        property_type = self.bridge.translate_csp_to_type(problem)
        
        # Crear contexto
        ctx = self.bridge.csp_to_context(problem)
        
        # Crear meta
        goal = ProofGoal(property_type, ctx, property_name)
        
        # Intentar probar (con tácticas)
        proof = self.engine.search_proof_with_tactics(goal, max_depth=5)
        
        is_valid = proof is not None
        message = "Problema globalmente consistente (prueba encontrada)" if is_valid else \
                  "No se pudo probar consistencia global"
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=is_valid,
            proof=proof,
            message=message
        )
        
        logger.info(message)
        return result
    
    # ========================================================================
    # Verificación de Soluciones
    # ========================================================================
    
    def verify_solution_correctness(self, solution: CSPSolution, 
                                    problem: CSPProblem) -> PropertyVerificationResult:
        """
        Verifica formalmente que una solución es correcta.
        
        Args:
            solution: Solución propuesta
            problem: Problema CSP
        
        Returns:
            Resultado de verificación
        """
        property_name = f"solution_correctness_{id(solution)}"
        
        logger.info("Verificando correctitud de solución")
        
        # Convertir solución a prueba
        proof = self.bridge.solution_to_proof_complete(solution, problem)
        
        if proof is None:
            result = PropertyVerificationResult(
                property_name=property_name,
                is_valid=False,
                message="Solución inválida: no se pudo generar prueba"
            )
            logger.warning(result.message)
            return result
        
        # Verificar la prueba mediante type-checking
        try:
            is_valid = self.engine.verify_proof(proof)
            message = "Solución formalmente correcta" if is_valid else \
                      "Solución inválida: prueba no verifica"
            
            result = PropertyVerificationResult(
                property_name=property_name,
                is_valid=is_valid,
                proof=proof if is_valid else None,
                message=message
            )
            
            logger.info(message)
            return result
            
        except Exception as e:
            logger.error(f"Error verificando prueba: {e}")
            return PropertyVerificationResult(
                property_name=property_name,
                is_valid=False,
                message=f"Error en verificación: {e}"
            )
    
    def verify_solution_completeness(self, solution: CSPSolution,
                                     problem: CSPProblem) -> PropertyVerificationResult:
        """
        Verifica que la solución asigna todas las variables.
        
        Args:
            solution: Solución propuesta
            problem: Problema CSP
        
        Returns:
            Resultado de verificación
        """
        property_name = "solution_completeness"
        
        logger.info("Verificando completitud de solución")
        
        missing_vars = []
        for var in problem.variables:
            if var not in solution.assignment:
                missing_vars.append(var)
        
        is_complete = len(missing_vars) == 0
        message = "Solución completa" if is_complete else \
                  f"Variables sin asignar: {missing_vars}"
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=is_complete,
            message=message
        )
        
        logger.info(message)
        return result
    
    # ========================================================================
    # Generación de Invariantes
    # ========================================================================
    
    def generate_invariants(self, problem: CSPProblem) -> List[Type]:
        """
        Genera invariantes del problema CSP.
        
        Un invariante es una propiedad que se mantiene durante la resolución.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Lista de tipos que representan invariantes
        """
        logger.info("Generando invariantes del problema")
        
        invariants = []
        
        # Invariante 1: Los dominios nunca están vacíos
        for var in problem.variables:
            dom_type = TypeVar(f"Dom_{var}")
            # ∃x. x : Dom
            non_empty_type = SigmaType("x", dom_type, Universe(0))
            invariants.append(non_empty_type)
        
        # Invariante 2: Las restricciones se mantienen
        for var1, var2, _ in problem.constraints:
            # C(var1, var2) siempre es válida
            constraint_type = TypeVar(f"C_{var1}_{var2}")
            invariants.append(constraint_type)
        
        # Invariante 3: Cada variable tiene exactamente un valor (en solución)
        # ∀v. ∃!x. assignment(v) = x
        # (Simplificado: no implementado completamente)
        
        logger.info(f"Generados {len(invariants)} invariantes")
        
        return invariants
    
    def verify_invariant(self, invariant: Type, 
                        problem: CSPProblem) -> PropertyVerificationResult:
        """
        Verifica que un invariante se cumple.
        
        Args:
            invariant: Tipo que representa el invariante
            problem: Problema CSP
        
        Returns:
            Resultado de verificación
        """
        property_name = f"invariant_{invariant}"
        
        logger.info(f"Verificando invariante: {invariant}")
        
        # Crear contexto
        ctx = self.bridge.csp_to_context(problem)
        
        # Crear meta
        goal = ProofGoal(invariant, ctx, property_name)
        
        # Intentar probar
        proof = self.engine.search_proof_with_tactics(goal, max_depth=3)
        
        is_valid = proof is not None
        message = f"Invariante {'válido' if is_valid else 'no probado'}"
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=is_valid,
            proof=proof,
            message=message
        )
        
        logger.info(message)
        return result
    
    # ========================================================================
    # Propiedades Adicionales
    # ========================================================================
    
    def verify_domain_consistency(self, problem: CSPProblem) -> PropertyVerificationResult:
        """
        Verifica que todos los dominios son no vacíos.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Resultado de verificación
        """
        property_name = "domain_consistency"
        
        logger.info("Verificando consistencia de dominios")
        
        empty_domains = []
        for var, domain in problem.domains.items():
            if len(domain) == 0:
                empty_domains.append(var)
        
        is_valid = len(empty_domains) == 0
        message = "Todos los dominios son no vacíos" if is_valid else \
                  f"Dominios vacíos: {empty_domains}"
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=is_valid,
            message=message
        )
        
        logger.info(message)
        return result
    
    def verify_constraint_symmetry(self, problem: CSPProblem,
                                   var1: str, var2: str) -> PropertyVerificationResult:
        """
        Verifica si una restricción es simétrica.
        
        Propiedad: C(x, y) ↔ C(y, x)
        
        Args:
            problem: Problema CSP
            var1: Primera variable
            var2: Segunda variable
        
        Returns:
            Resultado de verificación
        """
        property_name = f"constraint_symmetry_{var1}_{var2}"
        
        logger.info(f"Verificando simetría de restricción: {var1}, {var2}")
        
        # Buscar restricción
        constraint_fn = None
        for v1, v2, fn in problem.constraints:
            if (v1 == var1 and v2 == var2):
                constraint_fn = fn
                break
        
        if constraint_fn is None:
            return PropertyVerificationResult(
                property_name=property_name,
                is_valid=False,
                message=f"No existe restricción {var1}-{var2}"
            )
        
        # Verificar simetría
        dom1 = problem.domains[var1]
        dom2 = problem.domains[var2]
        
        is_symmetric = True
        for val1 in dom1:
            for val2 in dom2:
                if constraint_fn(val1, val2) != constraint_fn(val2, val1):
                    is_symmetric = False
                    break
            if not is_symmetric:
                break
        
        message = f"Restricción {var1}-{var2} es {'simétrica' if is_symmetric else 'asimétrica'}"
        
        result = PropertyVerificationResult(
            property_name=property_name,
            is_valid=is_symmetric,
            message=message
        )
        
        logger.info(message)
        return result
    
    # ========================================================================
    # Utilidades
    # ========================================================================
    
    def get_verification_statistics(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de verificaciones realizadas.
        
        Returns:
            Diccionario con estadísticas
        """
        total = len(self.verification_cache)
        valid = sum(1 for r in self.verification_cache.values() if r.is_valid)
        invalid = total - valid
        
        return {
            'total_verifications': total,
            'valid_properties': valid,
            'invalid_properties': invalid,
            'cached_results': total
        }
    
    def clear_cache(self):
        """Limpia la caché de verificaciones."""
        self.verification_cache.clear()
        logger.info("Caché de verificaciones limpiada")


def create_property_verifier() -> CSPPropertyVerifier:
    """
    Crea una instancia del verificador de propiedades.
    
    Returns:
        Verificador inicializado
    """
    return CSPPropertyVerifier()

