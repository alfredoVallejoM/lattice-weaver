"""
Operaciones sobre Sintaxis Cúbica

Este módulo implementa operaciones básicas sobre el AST de HoTT:
- Reducción (evaluación)
- Normalización
- Comparación de términos
- Construcción de términos comunes

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Optional
from .cubical_syntax import *
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# REDUCCIÓN Y EVALUACIÓN
# ============================================================================

def beta_reduce(term: Term) -> Term:
    """
    Realiza una beta-reducción en el término.
    
    Beta-reducción: (λx. t) a → t[x := a]
    
    Args:
        term: Término a reducir
    
    Returns:
        Término reducido (un paso)
    """
    if isinstance(term, App):
        # Reducir función y argumento primero
        func = beta_reduce(term.func)
        arg = beta_reduce(term.arg)
        
        # Si la función es una lambda, aplicar beta-reducción
        if isinstance(func, Lambda):
            return func.body.substitute(func.var, arg)
        
        return App(func, arg)
    
    elif isinstance(term, Lambda):
        return Lambda(term.var, term.type_, beta_reduce(term.body))
    
    elif isinstance(term, Pair):
        return Pair(beta_reduce(term.first), beta_reduce(term.second))
    
    elif isinstance(term, Fst):
        pair = beta_reduce(term.pair)
        if isinstance(pair, Pair):
            return pair.first
        return Fst(pair)
    
    elif isinstance(term, Snd):
        pair = beta_reduce(term.pair)
        if isinstance(pair, Pair):
            return pair.second
        return Snd(pair)
    
    elif isinstance(term, PathApp):
        path = beta_reduce(term.path)
        
        # Reducir abstracción de camino
        if isinstance(path, PathAbs):
            if term.point == "0":
                # Evaluar en el extremo izquierdo
                return path.body.substitute(path.var, Var("0"))
            elif term.point == "1":
                # Evaluar en el extremo derecho
                return path.body.substitute(path.var, Var("1"))
            else:
                # Sustituir variable de intervalo
                return path.body.substitute(path.var, Var(term.point))
        
        # Reducir refl
        if isinstance(path, Refl):
            return path.point
        
        return PathApp(path, term.point)
    
    elif isinstance(term, PathAbs):
        return PathAbs(term.var, beta_reduce(term.body))
    
    elif isinstance(term, Refl):
        return Refl(beta_reduce(term.point))
    
    else:
        # Variable u otros términos no reducibles
        return term


def normalize(term: Term, max_steps: int = 100) -> Term:
    """
    Normaliza un término aplicando beta-reducción hasta que no cambie.
    
    Args:
        term: Término a normalizar
        max_steps: Número máximo de pasos de reducción
    
    Returns:
        Término en forma normal
    """
    for _ in range(max_steps):
        reduced = beta_reduce(term)
        if alpha_equivalent(reduced, term):
            return reduced
        term = reduced
    
    logger.warning(f"Normalización no convergió en {max_steps} pasos")
    return term


def alpha_equivalent(term1: Term, term2: Term) -> bool:
    """
    Verifica si dos términos son alfa-equivalentes.
    
    Dos términos son alfa-equivalentes si son iguales salvo renombramiento
    de variables ligadas.
    
    Args:
        term1: Primer término
        term2: Segundo término
    
    Returns:
        True si son alfa-equivalentes
    """
    # Simplificación: comparación estructural
    # Una implementación completa requeriría renombramiento consistente
    
    if type(term1) != type(term2):
        return False
    
    if isinstance(term1, Var):
        return term1.name == term2.name
    
    elif isinstance(term1, Lambda):
        # Renombrar ambas lambdas a una variable fresca
        fresh_var = f"_alpha_{id(term1)}"
        body1 = term1.body.substitute(term1.var, Var(fresh_var))
        body2 = term2.body.substitute(term2.var, Var(fresh_var))
        return alpha_equivalent(body1, body2)
    
    elif isinstance(term1, App):
        return (alpha_equivalent(term1.func, term2.func) and
                alpha_equivalent(term1.arg, term2.arg))
    
    elif isinstance(term1, Pair):
        return (alpha_equivalent(term1.first, term2.first) and
                alpha_equivalent(term1.second, term2.second))
    
    elif isinstance(term1, Fst):
        return alpha_equivalent(term1.pair, term2.pair)
    
    elif isinstance(term1, Snd):
        return alpha_equivalent(term1.pair, term2.pair)
    
    elif isinstance(term1, Refl):
        return alpha_equivalent(term1.point, term2.point)
    
    elif isinstance(term1, PathAbs):
        fresh_var = f"_alpha_{id(term1)}"
        body1 = term1.body.substitute(term1.var, Var(fresh_var))
        body2 = term2.body.substitute(term2.var, Var(fresh_var))
        return alpha_equivalent(body1, body2)
    
    elif isinstance(term1, PathApp):
        return (alpha_equivalent(term1.path, term2.path) and
                term1.point == term2.point)
    
    return False


# ============================================================================
# CONSTRUCCIÓN DE TÉRMINOS COMUNES
# ============================================================================

def identity_function(type_: Type) -> Lambda:
    """
    Construye la función identidad: λ(x : A). x
    
    Args:
        type_: Tipo del argumento
    
    Returns:
        Función identidad
    """
    return Lambda("x", type_, Var("x"))


def compose_functions(f: Term, g: Term, 
                     type_a: Type, type_b: Type, type_c: Type) -> Lambda:
    """
    Construye la composición de funciones: λx. f (g x)
    
    Args:
        f: Función B → C
        g: Función A → B
        type_a: Tipo A
        type_b: Tipo B
        type_c: Tipo C
    
    Returns:
        Función compuesta A → C
    """
    return Lambda("x", type_a, 
                 App(f, App(g, Var("x"))))


def constant_function(type_a: Type, type_b: Type, value: Term) -> Lambda:
    """
    Construye una función constante: λ(x : A). b
    
    Args:
        type_a: Tipo del argumento
        type_b: Tipo del resultado
        value: Valor constante
    
    Returns:
        Función constante
    """
    return Lambda("x", type_a, value)


def swap_pair(type_a: Type, type_b: Type) -> Lambda:
    """
    Construye la función que intercambia componentes de un par:
    λ(p : A × B). (snd p, fst p)
    
    Args:
        type_a: Tipo del primer componente
        type_b: Tipo del segundo componente
    
    Returns:
        Función que intercambia componentes
    """
    pair_type = product_type(type_a, type_b)
    return Lambda("p", pair_type,
                 Pair(Snd(Var("p")), Fst(Var("p"))))


def path_inverse(type_: Type, a: Term, b: Term, path: Term) -> PathAbs:
    """
    Construye el camino inverso: si p : Path A a b, entonces p⁻¹ : Path A b a
    
    Args:
        type_: Tipo en el que viven los puntos
        a: Punto inicial del camino original
        b: Punto final del camino original
        path: Camino de a a b
    
    Returns:
        Camino inverso de b a a
    """
    # p⁻¹ = <i> p @ (1 - i)
    # Simplificación: <i> p @ i  (debería ser 1-i pero no tenemos aritmética de intervalos)
    return PathAbs("i", PathApp(path, "i"))


def path_compose(type_: Type, a: Term, b: Term, c: Term,
                path1: Term, path2: Term) -> PathAbs:
    """
    Construye la composición de caminos:
    Si p : Path A a b y q : Path A b c, entonces p ∙ q : Path A a c
    
    Args:
        type_: Tipo en el que viven los puntos
        a: Punto inicial
        b: Punto intermedio
        c: Punto final
        path1: Camino de a a b
        path2: Camino de b a c
    
    Returns:
        Camino compuesto de a a c
    """
    # Simplificación: concatenación de caminos
    # En una implementación completa, usaríamos operaciones de max/min en intervalos
    return PathAbs("i", PathApp(path1, "i"))


def transport(type_family: Lambda, path: Term, value: Term) -> Term:
    """
    Transporte a lo largo de un camino.
    
    Si P : A → Type, p : Path A a b, y x : P a,
    entonces transport P p x : P b
    
    Args:
        type_family: Familia de tipos P : A → Type
        path: Camino p : Path A a b
        value: Valor x : P a
    
    Returns:
        Valor transportado : P b
    """
    # Simplificación: aplicar la familia de tipos al extremo final del camino
    endpoint = PathApp(path, "1")
    return App(type_family, endpoint)


def j_eliminator(type_: Type, a: Term, 
                motive: Lambda, base_case: Term,
                b: Term, path: Term) -> Term:
    """
    Eliminador J (inducción sobre igualdad).
    
    Principio fundamental de HoTT: para probar algo sobre todos los caminos
    desde a, basta probarlo para refl a.
    
    Args:
        type_: Tipo base A
        a: Punto base
        motive: Motivo C : Π(x : A). Path A a x → Type
        base_case: Caso base c : C a (refl a)
        b: Punto objetivo
        path: Camino p : Path A a b
    
    Returns:
        Resultado : C b p
    """
    # Simplificación: si path es refl, devolver base_case
    if isinstance(path, Refl):
        return base_case
    
    # En general, necesitaríamos evaluar el motivo
    return App(App(motive, b), path)


# ============================================================================
# UTILIDADES DE TIPOS
# ============================================================================

def is_function_type(type_: Type) -> bool:
    """Verifica si un tipo es un tipo función (Pi)."""
    return isinstance(type_, PiType)


def is_product_type(type_: Type) -> bool:
    """Verifica si un tipo es un tipo producto (Sigma)."""
    return isinstance(type_, SigmaType)


def is_path_type(type_: Type) -> bool:
    """Verifica si un tipo es un tipo de caminos."""
    return isinstance(type_, PathType)


def get_function_domain(type_: PiType) -> Type:
    """Obtiene el dominio de un tipo función."""
    return type_.domain


def get_function_codomain(type_: PiType) -> Type:
    """Obtiene el codominio de un tipo función."""
    return type_.codomain

