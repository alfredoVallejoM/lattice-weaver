"""
Módulo Formal de LatticeWeaver

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from .heyting_algebra import HeytingAlgebra, HeytingElement, create_power_set_algebra
from .heyting_optimized import OptimizedHeytingAlgebra
from .lattice_to_heyting import lattice_to_heyting, concept_to_proposition, heyting_to_logic_table
from .cubical_syntax import (
    Type, Universe, TypeVar, PiType, SigmaType, PathType,
    Term, Var, Lambda, App, Pair, Fst, Snd, Refl, PathAbs, PathApp,
    Context, Binding,
    arrow_type, product_type, identity_type
)
from .cubical_operations import (
    beta_reduce, normalize, alpha_equivalent,
    identity_function, compose_functions, constant_function, swap_pair,
    path_inverse, path_compose, transport, j_eliminator,
    is_function_type, is_product_type, is_path_type
)
from .type_checker import TypeChecker, TypeCheckError, type_check, infer, well_typed
from .cubical_engine import CubicalEngine, ProofGoal, ProofTerm, create_engine
from .tactics import TacticEngine, TacticResult, create_tactic_engine
from .csp_integration import (
    CSPHoTTBridge, CSPProblem, CSPSolution,
    create_bridge, simple_csp_example, simple_solution_example
)
from .csp_integration_extended import (
    ExtendedCSPHoTTBridge,
    create_extended_bridge,
    example_graph_coloring_translation,
    example_invalid_solution
)
from .csp_properties import (
    CSPPropertyVerifier, PropertyVerificationResult,
    create_property_verifier
)
from .csp_logic_interpretation import (
    CSPLogicInterpreter, CSPSemantics, DomainInterpretation,
    ConstraintInterpretation, PropagationInterpretation,
    create_logic_interpreter, compare_semantics
)

__all__ = [
    # Heyting Algebra
    'HeytingAlgebra',
    'HeytingElement',
    'OptimizedHeytingAlgebra',
    'create_power_set_algebra',
    'lattice_to_heyting',
    'concept_to_proposition',
    'heyting_to_logic_table',
    
    # Cubical Syntax - Types
    'Type', 'Universe', 'TypeVar', 'PiType', 'SigmaType', 'PathType',
    
    # Cubical Syntax - Terms
    'Term', 'Var', 'Lambda', 'App', 'Pair', 'Fst', 'Snd', 
    'Refl', 'PathAbs', 'PathApp',
    
    # Cubical Syntax - Context
    'Context', 'Binding',
    
    # Cubical Syntax - Utilities
    'arrow_type', 'product_type', 'identity_type',
    
    # Cubical Operations
    'beta_reduce', 'normalize', 'alpha_equivalent',
    'identity_function', 'compose_functions', 'constant_function', 'swap_pair',
    'path_inverse', 'path_compose', 'transport', 'j_eliminator',
    'is_function_type', 'is_product_type', 'is_path_type',
    
    # Type Checker
    'TypeChecker', 'TypeCheckError', 'type_check', 'infer', 'well_typed',
    
    # Cubical Engine
    'CubicalEngine', 'ProofGoal', 'ProofTerm', 'create_engine',
    
    # Tactics
    'TacticEngine', 'TacticResult', 'create_tactic_engine',
    
    # CSP Integration
    'CSPHoTTBridge', 'CSPProblem', 'CSPSolution',
    'create_bridge', 'simple_csp_example', 'simple_solution_example',
    
    # CSP Integration Extended
    'ExtendedCSPHoTTBridge', 'create_extended_bridge',
    'example_graph_coloring_translation', 'example_invalid_solution',
    
    # CSP Properties
    'CSPPropertyVerifier', 'PropertyVerificationResult',
    'create_property_verifier',
    
    # CSP Logic Interpretation
    'CSPLogicInterpreter', 'CSPSemantics', 'DomainInterpretation',
    'ConstraintInterpretation', 'PropagationInterpretation',
    'create_logic_interpreter', 'compare_semantics'
]
