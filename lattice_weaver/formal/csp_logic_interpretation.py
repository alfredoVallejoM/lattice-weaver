"""
Interpretación Lógica Completa de CSP en HoTT - Fase 10

Proporciona una interpretación lógica completa de problemas CSP en el marco
de la Teoría de Tipos Homotópica (HoTT), incluyendo:
- Semántica denotacional de CSP en HoTT
- Interpretación de restricciones como tipos dependientes
- Traducción de algoritmos de propagación a transformaciones de tipos
- Correspondencia Curry-Howard para CSP
- Interpretación categórica de dominios y restricciones

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .cubical_syntax import *
from .cubical_engine import CubicalEngine, ProofGoal, ProofTerm
from .csp_integration import CSPProblem, CSPSolution
from .csp_integration_extended import ExtendedCSPHoTTBridge

logger = logging.getLogger(__name__)


# ============================================================================
# Semántica Denotacional de CSP en HoTT
# ============================================================================

class CSPSemantics(Enum):
    """Diferentes semánticas para interpretar CSP en HoTT."""
    PROPOSITIONAL = "propositional"  # Restricciones como proposiciones
    PROOF_RELEVANT = "proof_relevant"  # Pruebas como datos
    HOMOTOPICAL = "homotopical"  # Soluciones como puntos en espacio
    CATEGORICAL = "categorical"  # Interpretación categórica


@dataclass
class DomainInterpretation:
    """
    Interpretación de un dominio CSP en HoTT.
    
    Attributes:
        domain_type: Tipo HoTT que representa el dominio
        elements: Mapeo de valores del dominio a términos
        axioms: Axiomas sobre el dominio
        semantics: Semántica utilizada
    """
    domain_type: Type
    elements: Dict[Any, Term]
    axioms: List[Type]
    semantics: CSPSemantics


@dataclass
class ConstraintInterpretation:
    """
    Interpretación de una restricción CSP en HoTT.
    
    Attributes:
        constraint_type: Tipo HoTT que representa la restricción
        proof_constructor: Función que construye pruebas de satisfacción
        semantics: Semántica utilizada
    """
    constraint_type: Type
    proof_constructor: Callable[[Any, Any], Optional[Term]]
    semantics: CSPSemantics


@dataclass
class PropagationInterpretation:
    """
    Interpretación de propagación de restricciones como transformación de tipos.
    
    Attributes:
        before_type: Tipo antes de la propagación
        after_type: Tipo después de la propagación
        transformation: Función de transformación de términos
        correctness_proof: Prueba de correctitud de la transformación
    """
    before_type: Type
    after_type: Type
    transformation: Callable[[Term], Term]
    correctness_proof: Optional[ProofTerm]


class CSPLogicInterpreter:
    """
    Intérprete lógico completo de CSP en HoTT.
    
    Proporciona interpretación semántica completa de problemas CSP
    en el marco de la Teoría de Tipos Homotópica.
    """
    
    def __init__(self, semantics: CSPSemantics = CSPSemantics.PROOF_RELEVANT):
        """
        Inicializa el intérprete.
        
        Args:
            semantics: Semántica a utilizar
        """
        self.semantics = semantics
        self.bridge = ExtendedCSPHoTTBridge()
        self.engine = CubicalEngine()
        
        # Cachés de interpretaciones
        self.domain_interpretations: Dict[str, DomainInterpretation] = {}
        self.constraint_interpretations: Dict[Tuple[str, str], ConstraintInterpretation] = {}
        self.propagation_interpretations: List[PropagationInterpretation] = []
    
    # ========================================================================
    # Interpretación de Dominios
    # ========================================================================
    
    def interpret_domain(self, var_name: str, domain: Set, 
                        problem: CSPProblem) -> DomainInterpretation:
        """
        Interpreta un dominio CSP en HoTT según la semántica elegida.
        
        Args:
            var_name: Nombre de la variable
            domain: Conjunto de valores del dominio
            problem: Problema CSP completo (para contexto)
        
        Returns:
            Interpretación del dominio
        """
        if var_name in self.domain_interpretations:
            return self.domain_interpretations[var_name]
        
        if self.semantics == CSPSemantics.PROPOSITIONAL:
            interp = self._interpret_domain_propositional(var_name, domain)
        elif self.semantics == CSPSemantics.PROOF_RELEVANT:
            interp = self._interpret_domain_proof_relevant(var_name, domain)
        elif self.semantics == CSPSemantics.HOMOTOPICAL:
            interp = self._interpret_domain_homotopical(var_name, domain)
        else:  # CATEGORICAL
            interp = self._interpret_domain_categorical(var_name, domain)
        
        self.domain_interpretations[var_name] = interp
        return interp
    
    def _interpret_domain_propositional(self, var_name: str, 
                                       domain: Set) -> DomainInterpretation:
        """
        Interpretación proposicional: dominio como tipo suma.
        
        Dom(x) = v1 + v2 + ... + vn
        """
        domain_list = sorted(list(domain))
        
        # Tipo suma (coproducto)
        if len(domain_list) == 1:
            domain_type = TypeVar(f"Unit_{var_name}")
        else:
            # Construir suma iterativamente
            domain_type = TypeVar(f"Dom_{var_name}")
        
        # Elementos como inyecciones
        elements = {}
        for i, val in enumerate(domain_list):
            elements[val] = Var(f"inj_{i}_{var_name}_{val}")
        
        # Axioma: dominio es finito
        axioms = [
            TypeVar(f"finite_{var_name}")
        ]
        
        return DomainInterpretation(
            domain_type=domain_type,
            elements=elements,
            axioms=axioms,
            semantics=CSPSemantics.PROPOSITIONAL
        )
    
    def _interpret_domain_proof_relevant(self, var_name: str, 
                                        domain: Set) -> DomainInterpretation:
        """
        Interpretación proof-relevant: dominio como tipo Sigma.
        
        Dom(x) = Σ(v : Values). (v ∈ domain)
        """
        domain_list = sorted(list(domain))
        
        # Tipo base de valores
        value_type = TypeVar(f"Value_{var_name}")
        
        # Predicado de pertenencia
        membership_pred = PiType(
            "v", value_type,
            Universe(0)  # v ∈ domain : Prop
        )
        
        # Tipo Sigma: valores con prueba de pertenencia
        domain_type = SigmaType(
            "v", value_type,
            App(Var("in_domain"), Var("v"))
        )
        
        # Elementos como pares (valor, prueba)
        elements = {}
        for val in domain_list:
            val_term = Var(f"val_{var_name}_{val}")
            proof_term = Var(f"proof_in_{var_name}_{val}")
            elements[val] = Pair(val_term, proof_term)
        
        # Axiomas: decidibilidad de pertenencia
        axioms = [membership_pred]
        
        return DomainInterpretation(
            domain_type=domain_type,
            elements=elements,
            axioms=axioms,
            semantics=CSPSemantics.PROOF_RELEVANT
        )
    
    def _interpret_domain_homotopical(self, var_name: str, 
                                     domain: Set) -> DomainInterpretation:
        """
        Interpretación homotópica: dominio como espacio discreto.
        
        Dom(x) es un tipo con estructura de igualdad trivial.
        """
        domain_list = sorted(list(domain))
        
        # Tipo base
        domain_type = TypeVar(f"Dom_{var_name}")
        
        # Elementos como puntos en el espacio
        elements = {}
        for val in domain_list:
            elements[val] = Var(f"point_{var_name}_{val}")
        
        # Axiomas: espacio discreto (decidibilidad de igualdad)
        axioms = [
            PiType("x", domain_type,
                  PiType("y", domain_type,
                        TypeVar(f"decidable_eq_{var_name}")))
        ]
        
        return DomainInterpretation(
            domain_type=domain_type,
            elements=elements,
            axioms=axioms,
            semantics=CSPSemantics.HOMOTOPICAL
        )
    
    def _interpret_domain_categorical(self, var_name: str, 
                                     domain: Set) -> DomainInterpretation:
        """
        Interpretación categórica: dominio como objeto en categoría.
        
        Dom(x) es un objeto con morfismos hacia otros dominios.
        """
        domain_type = TypeVar(f"Obj_{var_name}")
        
        # Elementos como morfismos desde terminal
        elements = {}
        for val in domain:
            elements[val] = Var(f"morph_1_{var_name}_{val}")
        
        axioms = []
        
        return DomainInterpretation(
            domain_type=domain_type,
            elements=elements,
            axioms=axioms,
            semantics=CSPSemantics.CATEGORICAL
        )
    
    # ========================================================================
    # Interpretación de Restricciones
    # ========================================================================
    
    def interpret_constraint(self, var1: str, var2: str, 
                           relation: Callable,
                           problem: CSPProblem) -> ConstraintInterpretation:
        """
        Interpreta una restricción CSP en HoTT.
        
        Args:
            var1: Primera variable
            var2: Segunda variable
            relation: Función de relación
            problem: Problema CSP completo
        
        Returns:
            Interpretación de la restricción
        """
        key = (var1, var2)
        if key in self.constraint_interpretations:
            return self.constraint_interpretations[key]
        
        if self.semantics == CSPSemantics.PROPOSITIONAL:
            interp = self._interpret_constraint_propositional(var1, var2, relation)
        elif self.semantics == CSPSemantics.PROOF_RELEVANT:
            interp = self._interpret_constraint_proof_relevant(var1, var2, relation)
        elif self.semantics == CSPSemantics.HOMOTOPICAL:
            interp = self._interpret_constraint_homotopical(var1, var2, relation)
        else:  # CATEGORICAL
            interp = self._interpret_constraint_categorical(var1, var2, relation)
        
        self.constraint_interpretations[key] = interp
        return interp
    
    def _interpret_constraint_propositional(self, var1: str, var2: str,
                                          relation: Callable) -> ConstraintInterpretation:
        """
        Interpretación proposicional: restricción como proposición.
        
        C(x, y) : Prop
        """
        dom1 = TypeVar(f"Dom_{var1}")
        dom2 = TypeVar(f"Dom_{var2}")
        
        # Tipo función: Dom1 → Dom2 → Prop
        constraint_type = PiType(
            var1, dom1,
            PiType(var2, dom2, Universe(0))
        )
        
        # Constructor de pruebas (axiomático)
        def proof_constructor(val1, val2):
            if relation(val1, val2):
                return Var(f"proof_{var1}_{var2}_{val1}_{val2}")
            return None
        
        return ConstraintInterpretation(
            constraint_type=constraint_type,
            proof_constructor=proof_constructor,
            semantics=CSPSemantics.PROPOSITIONAL
        )
    
    def _interpret_constraint_proof_relevant(self, var1: str, var2: str,
                                           relation: Callable) -> ConstraintInterpretation:
        """
        Interpretación proof-relevant: restricción como tipo de datos.
        
        C(x, y) es un tipo habitado si la restricción se satisface.
        """
        dom1 = TypeVar(f"Dom_{var1}")
        dom2 = TypeVar(f"Dom_{var2}")
        
        # Tipo función que devuelve tipo (no proposición)
        constraint_type = PiType(
            var1, dom1,
            PiType(var2, dom2, Universe(0))
        )
        
        def proof_constructor(val1, val2):
            if relation(val1, val2):
                # Prueba con datos relevantes
                return Pair(
                    Var(f"witness_{val1}_{val2}"),
                    Var(f"proof_{var1}_{var2}_{val1}_{val2}")
                )
            return None
        
        return ConstraintInterpretation(
            constraint_type=constraint_type,
            proof_constructor=proof_constructor,
            semantics=CSPSemantics.PROOF_RELEVANT
        )
    
    def _interpret_constraint_homotopical(self, var1: str, var2: str,
                                        relation: Callable) -> ConstraintInterpretation:
        """
        Interpretación homotópica: restricción como fibración.
        
        C es una fibración sobre Dom1 × Dom2.
        """
        dom1 = TypeVar(f"Dom_{var1}")
        dom2 = TypeVar(f"Dom_{var2}")
        
        # Tipo producto
        base = product_type(dom1, dom2)
        
        # Fibración: Base → Type
        constraint_type = PiType("p", base, Universe(0))
        
        def proof_constructor(val1, val2):
            if relation(val1, val2):
                return Var(f"fiber_{var1}_{var2}_{val1}_{val2}")
            return None
        
        return ConstraintInterpretation(
            constraint_type=constraint_type,
            proof_constructor=proof_constructor,
            semantics=CSPSemantics.HOMOTOPICAL
        )
    
    def _interpret_constraint_categorical(self, var1: str, var2: str,
                                        relation: Callable) -> ConstraintInterpretation:
        """
        Interpretación categórica: restricción como morfismo.
        
        C : Dom1 × Dom2 → Ω (clasificador de subobjetos)
        """
        dom1 = TypeVar(f"Obj_{var1}")
        dom2 = TypeVar(f"Obj_{var2}")
        
        # Morfismo al clasificador
        omega = TypeVar("Omega")  # Clasificador de subobjetos
        constraint_type = PiType(
            "p", product_type(dom1, dom2),
            omega
        )
        
        def proof_constructor(val1, val2):
            if relation(val1, val2):
                return Var(f"true_{var1}_{var2}")
            return Var(f"false_{var1}_{var2}")
        
        return ConstraintInterpretation(
            constraint_type=constraint_type,
            proof_constructor=proof_constructor,
            semantics=CSPSemantics.CATEGORICAL
        )
    
    # ========================================================================
    # Interpretación de Propagación
    # ========================================================================
    
    def interpret_arc_consistency_step(self, var: str, domain_before: Set,
                                      domain_after: Set,
                                      problem: CSPProblem) -> PropagationInterpretation:
        """
        Interpreta un paso de arc-consistency como transformación de tipos.
        
        Args:
            var: Variable afectada
            domain_before: Dominio antes de la propagación
            domain_after: Dominio después de la propagación
            problem: Problema CSP
        
        Returns:
            Interpretación de la propagación
        """
        # Tipos antes y después
        before_interp = self.interpret_domain(var, domain_before, problem)
        after_interp = self.interpret_domain(var, domain_after, problem)
        
        before_type = before_interp.domain_type
        after_type = after_interp.domain_type
        
        # Transformación: proyección (eliminación de valores)
        def transformation(term: Term) -> Term:
            # En la práctica, esto sería una función de proyección
            return App(Var(f"project_{var}"), term)
        
        # Prueba de correctitud: after ⊆ before
        correctness_proof = self._build_subset_proof(
            var, domain_after, domain_before
        )
        
        interp = PropagationInterpretation(
            before_type=before_type,
            after_type=after_type,
            transformation=transformation,
            correctness_proof=correctness_proof
        )
        
        self.propagation_interpretations.append(interp)
        
        return interp
    
    def _build_subset_proof(self, var: str, subset: Set, 
                          superset: Set) -> Optional[ProofTerm]:
        """
        Construye una prueba de que subset ⊆ superset.
        
        Args:
            var: Variable
            subset: Subconjunto
            superset: Superconjunto
        
        Returns:
            Prueba formal o None
        """
        if not subset.issubset(superset):
            return None
        
        # Tipo: ∀x ∈ subset. x ∈ superset
        proof_type = PiType(
            "x", TypeVar(f"Dom_{var}"),
            Universe(0)
        )
        
        # Término de prueba (axiomático)
        proof_term = Lambda("x", TypeVar(f"Dom_{var}"), Var(f"proof_subset_{var}"))
        
        ctx = {}
        
        return ProofTerm(proof_term, proof_type, ctx)
    
    # ========================================================================
    # Correspondencia Curry-Howard para CSP
    # ========================================================================
    
    def curry_howard_correspondence(self, problem: CSPProblem) -> Dict[str, Any]:
        """
        Establece la correspondencia Curry-Howard para un problema CSP.
        
        Correspondencia:
        - Dominios ↔ Tipos
        - Valores ↔ Términos
        - Restricciones ↔ Proposiciones/Tipos
        - Soluciones ↔ Pruebas/Habitantes
        - Arc-consistency ↔ Normalización de tipos
        - Backtracking ↔ Búsqueda de pruebas
        
        Args:
            problem: Problema CSP
        
        Returns:
            Diccionario con la correspondencia
        """
        correspondence = {
            'domains_to_types': {},
            'values_to_terms': {},
            'constraints_to_propositions': {},
            'solutions_to_proofs': 'Σ-type inhabitants',
            'arc_consistency_to': 'Type normalization',
            'backtracking_to': 'Proof search',
            'semantics': self.semantics.value
        }
        
        # Dominios → Tipos
        for var in problem.variables:
            domain = problem.domains[var]
            interp = self.interpret_domain(var, domain, problem)
            correspondence['domains_to_types'][var] = str(interp.domain_type)
            correspondence['values_to_terms'][var] = {
                val: str(term) for val, term in interp.elements.items()
            }
        
        # Restricciones → Proposiciones
        for var1, var2, relation in problem.constraints:
            interp = self.interpret_constraint(var1, var2, relation, problem)
            key = f"{var1}_{var2}"
            correspondence['constraints_to_propositions'][key] = str(interp.constraint_type)
        
        return correspondence
    
    # ========================================================================
    # Análisis y Estadísticas
    # ========================================================================
    
    def get_interpretation_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la interpretación.
        
        Returns:
            Diccionario con estadísticas
        """
        return {
            'semantics': self.semantics.value,
            'domains_interpreted': len(self.domain_interpretations),
            'constraints_interpreted': len(self.constraint_interpretations),
            'propagations_interpreted': len(self.propagation_interpretations),
            'total_axioms': sum(
                len(interp.axioms) 
                for interp in self.domain_interpretations.values()
            )
        }
    
    def explain_interpretation(self, problem: CSPProblem) -> str:
        """
        Genera una explicación textual de la interpretación.
        
        Args:
            problem: Problema CSP
        
        Returns:
            Explicación en texto
        """
        lines = []
        lines.append(f"Interpretación Lógica de CSP en HoTT")
        lines.append(f"Semántica: {self.semantics.value}")
        lines.append("")
        
        lines.append("Dominios:")
        for var in problem.variables:
            domain = problem.domains[var]
            interp = self.interpret_domain(var, domain, problem)
            lines.append(f"  {var} : {interp.domain_type}")
            lines.append(f"    Valores: {len(domain)}")
            lines.append(f"    Axiomas: {len(interp.axioms)}")
        
        lines.append("")
        lines.append("Restricciones:")
        for var1, var2, relation in problem.constraints:
            interp = self.interpret_constraint(var1, var2, relation, problem)
            lines.append(f"  C({var1}, {var2}) : {interp.constraint_type}")
        
        lines.append("")
        lines.append("Correspondencia Curry-Howard:")
        corr = self.curry_howard_correspondence(problem)
        for key, value in corr.items():
            if isinstance(value, dict):
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_logic_interpreter(semantics: CSPSemantics = CSPSemantics.PROOF_RELEVANT) -> CSPLogicInterpreter:
    """
    Crea un intérprete lógico de CSP.
    
    Args:
        semantics: Semántica a utilizar
    
    Returns:
        Intérprete inicializado
    """
    return CSPLogicInterpreter(semantics)


def compare_semantics(problem: CSPProblem) -> Dict[str, Dict[str, Any]]:
    """
    Compara las diferentes semánticas para un problema CSP.
    
    Args:
        problem: Problema CSP
    
    Returns:
        Comparación de semánticas
    """
    results = {}
    
    for semantics in CSPSemantics:
        interpreter = create_logic_interpreter(semantics)
        
        # Interpretar problema
        for var in problem.variables:
            interpreter.interpret_domain(var, problem.domains[var], problem)
        
        for var1, var2, relation in problem.constraints:
            interpreter.interpret_constraint(var1, var2, relation, problem)
        
        # Obtener estadísticas
        stats = interpreter.get_interpretation_statistics()
        correspondence = interpreter.curry_howard_correspondence(problem)
        
        results[semantics.value] = {
            'statistics': stats,
            'correspondence': correspondence
        }
    
    return results

