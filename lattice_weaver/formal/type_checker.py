"""
Verificador de Tipos para Teoría de Tipos Homotópica (HoTT)

Este módulo implementa un verificador de tipos (type-checker) para el
sistema de tipos dependientes con semántica cúbica.

El verificador valida que los términos estén bien tipados según las
reglas de inferencia de HoTT, garantizando la correctitud de las pruebas
y construcciones.

Reglas principales implementadas:
- Universos y jerarquía de tipos
- Tipos dependientes (Pi y Sigma)
- Tipos de caminos (igualdad)
- Formación, introducción, eliminación y cómputo

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Optional, Tuple
from .cubical_syntax import *
from .cubical_operations import normalize, alpha_equivalent
import logging

logger = logging.getLogger(__name__)


class TypeCheckError(Exception):
    """Excepción lanzada cuando falla la verificación de tipos."""
    pass


class TypeChecker:
    """
    Verificador de tipos para HoTT.
    
    Implementa las reglas de tipado de la teoría de tipos dependientes
    con tipos de caminos.
    """
    
    def __init__(self):
        """Inicializa el verificador de tipos."""
        self.trace = []  # Para debugging
    
    def check_type(self, ctx: Context, term: Term, expected_type: Type) -> bool:
        """
        Verifica que un término tiene un tipo esperado.
        
        Modo de verificación (checking): Γ ⊢ t : A
        
        Args:
            ctx: Contexto de tipado
            term: Término a verificar
            expected_type: Tipo esperado
        
        Returns:
            True si el término tiene el tipo esperado
        
        Raises:
            TypeCheckError: Si el término no tiene el tipo esperado
        """
        inferred_type = self.infer_type(ctx, term)
        
        if not self.types_equal(ctx, inferred_type, expected_type):
            raise TypeCheckError(
                f"Type mismatch: expected {expected_type}, got {inferred_type}"
            )
        
        return True
    
    def infer_type(self, ctx: Context, term: Term) -> Type:
        """
        Infiere el tipo de un término.
        
        Modo de inferencia (synthesis): Γ ⊢ t ⇒ A
        
        Args:
            ctx: Contexto de tipado
            term: Término cuyo tipo se va a inferir
        
        Returns:
            Tipo inferido del término
        
        Raises:
            TypeCheckError: Si no se puede inferir el tipo
        """
        # Variable
        if isinstance(term, Var):
            return self._infer_var(ctx, term)
        
        # Lambda
        elif isinstance(term, Lambda):
            return self._infer_lambda(ctx, term)
        
        # Aplicación
        elif isinstance(term, App):
            return self._infer_app(ctx, term)
        
        # Par
        elif isinstance(term, Pair):
            return self._infer_pair(ctx, term)
        
        # Primera proyección
        elif isinstance(term, Fst):
            return self._infer_fst(ctx, term)
        
        # Segunda proyección
        elif isinstance(term, Snd):
            return self._infer_snd(ctx, term)
        
        # Reflexividad
        elif isinstance(term, Refl):
            return self._infer_refl(ctx, term)
        
        # Abstracción de camino
        elif isinstance(term, PathAbs):
            return self._infer_path_abs(ctx, term)
        
        # Aplicación de camino
        elif isinstance(term, PathApp):
            return self._infer_path_app(ctx, term)
        
        else:
            raise TypeCheckError(f"Cannot infer type of {term}")
    
    def check_type_formation(self, ctx: Context, type_: Type) -> bool:
        """
        Verifica que un tipo está bien formado.
        
        Regla de formación: Γ ⊢ A type
        
        Args:
            ctx: Contexto de tipado
            type_: Tipo a verificar
        
        Returns:
            True si el tipo está bien formado
        
        Raises:
            TypeCheckError: Si el tipo no está bien formado
        """
        # Universo
        if isinstance(type_, Universe):
            return True
        
        # Variable de tipo
        elif isinstance(type_, TypeVar):
            if ctx.lookup(type_.name) is None:
                raise TypeCheckError(f"Type variable {type_.name} not in context")
            return True
        
        # Tipo Pi
        elif isinstance(type_, PiType):
            # Verificar dominio
            self.check_type_formation(ctx, type_.domain)
            
            # Verificar codominio en contexto extendido
            ctx_ext = ctx.extend(type_.var, type_.domain)
            self.check_type_formation(ctx_ext, type_.codomain)
            
            return True
        
        # Tipo Sigma
        elif isinstance(type_, SigmaType):
            # Verificar primer componente
            self.check_type_formation(ctx, type_.first)
            
            # Verificar segundo componente en contexto extendido
            ctx_ext = ctx.extend(type_.var, type_.first)
            self.check_type_formation(ctx_ext, type_.second)
            
            return True
        
        # Tipo de caminos
        elif isinstance(type_, PathType):
            # Verificar que el tipo base está bien formado
            self.check_type_formation(ctx, type_.type_)
            
            # Verificar que los extremos tienen el tipo correcto
            self.check_type(ctx, type_.left, type_.type_)
            self.check_type(ctx, type_.right, type_.type_)
            
            return True
        
        else:
            raise TypeCheckError(f"Unknown type constructor: {type_}")
    
    def types_equal(self, ctx: Context, type1: Type, type2: Type) -> bool:
        """
        Verifica si dos tipos son iguales (definitionally equal).
        
        En teoría de tipos dependientes, la igualdad de tipos es igualdad
        definitional (por normalización).
        
        Args:
            ctx: Contexto de tipado
            type1: Primer tipo
            type2: Segundo tipo
        
        Returns:
            True si los tipos son iguales
        """
        # Simplificación: comparación estructural
        # Una implementación completa requeriría normalización de tipos
        
        if type(type1) != type(type2):
            return False
        
        if isinstance(type1, Universe):
            return type1.level == type2.level
        
        elif isinstance(type1, TypeVar):
            return type1.name == type2.name
        
        elif isinstance(type1, PiType):
            # Dominio debe ser igual
            if not self.types_equal(ctx, type1.domain, type2.domain):
                return False
            
            # Codominio debe ser igual en contexto extendido
            ctx_ext = ctx.extend(type1.var, type1.domain)
            return self.types_equal(ctx_ext, type1.codomain, type2.codomain)
        
        elif isinstance(type1, SigmaType):
            if not self.types_equal(ctx, type1.first, type2.first):
                return False
            
            ctx_ext = ctx.extend(type1.var, type1.first)
            return self.types_equal(ctx_ext, type1.second, type2.second)
        
        elif isinstance(type1, PathType):
            return (self.types_equal(ctx, type1.type_, type2.type_) and
                    self.terms_equal(ctx, type1.left, type2.left) and
                    self.terms_equal(ctx, type1.right, type2.right))
        
        return False
    
    def terms_equal(self, ctx: Context, term1: Term, term2: Term) -> bool:
        """
        Verifica si dos términos son iguales (definitionally equal).
        
        Args:
            ctx: Contexto de tipado
            term1: Primer término
            term2: Segundo término
        
        Returns:
            True si los términos son iguales
        """
        # Normalizar y comparar
        norm1 = normalize(term1)
        norm2 = normalize(term2)
        return alpha_equivalent(norm1, norm2)
    
    # ========================================================================
    # Reglas de inferencia específicas
    # ========================================================================
    
    def _infer_var(self, ctx: Context, var: Var) -> Type:
        """Infiere el tipo de una variable."""
        type_ = ctx.lookup(var.name)
        if type_ is None:
            raise TypeCheckError(f"Variable {var.name} not in context")
        return type_
    
    def _infer_lambda(self, ctx: Context, lam: Lambda) -> Type:
        """Infiere el tipo de una lambda."""
        # Verificar que el tipo del argumento está bien formado
        self.check_type_formation(ctx, lam.type_)
        
        # Inferir el tipo del cuerpo en contexto extendido
        ctx_ext = ctx.extend(lam.var, lam.type_)
        body_type = self.infer_type(ctx_ext, lam.body)
        
        # El tipo de la lambda es Π(x : A). B
        return PiType(lam.var, lam.type_, body_type)
    
    def _infer_app(self, ctx: Context, app: App) -> Type:
        """Infiere el tipo de una aplicación."""
        # Inferir el tipo de la función
        func_type = self.infer_type(ctx, app.func)
        
        # Debe ser un tipo Pi
        if not isinstance(func_type, PiType):
            raise TypeCheckError(f"Expected function type, got {func_type}")
        
        # Verificar que el argumento tiene el tipo correcto
        self.check_type(ctx, app.arg, func_type.domain)
        
        # El tipo resultado es el codominio con el argumento sustituido
        return func_type.codomain.substitute(func_type.var, app.arg)
    
    def _infer_pair(self, ctx: Context, pair: Pair) -> Type:
        """Infiere el tipo de un par."""
        # Inferir tipos de los componentes
        first_type = self.infer_type(ctx, pair.first)
        second_type = self.infer_type(ctx, pair.second)
        
        # El tipo es Σ(x : A). B, donde B no depende de x
        return SigmaType("_", first_type, second_type)
    
    def _infer_fst(self, ctx: Context, fst: Fst) -> Type:
        """Infiere el tipo de la primera proyección."""
        # Inferir el tipo del par
        pair_type = self.infer_type(ctx, fst.pair)
        
        # Debe ser un tipo Sigma
        if not isinstance(pair_type, SigmaType):
            raise TypeCheckError(f"Expected pair type, got {pair_type}")
        
        return pair_type.first
    
    def _infer_snd(self, ctx: Context, snd: Snd) -> Type:
        """Infiere el tipo de la segunda proyección."""
        # Inferir el tipo del par
        pair_type = self.infer_type(ctx, snd.pair)
        
        # Debe ser un tipo Sigma
        if not isinstance(pair_type, SigmaType):
            raise TypeCheckError(f"Expected pair type, got {pair_type}")
        
        # El tipo del segundo componente puede depender del primero
        # Sustituir con fst del par
        return pair_type.second.substitute(pair_type.var, Fst(snd.pair))
    
    def _infer_refl(self, ctx: Context, refl: Refl) -> Type:
        """Infiere el tipo de la reflexividad."""
        # Inferir el tipo del punto
        point_type = self.infer_type(ctx, refl.point)
        
        # El tipo es Path A a a
        return PathType(point_type, refl.point, refl.point)
    
    def _infer_path_abs(self, ctx: Context, path_abs: PathAbs) -> Type:
        """Infiere el tipo de una abstracción de camino."""
        # Simplificación: asumir que la variable de intervalo tiene tipo I
        # En una implementación completa, I sería un tipo especial
        
        # Inferir el tipo del cuerpo
        # (En realidad necesitaríamos un contexto con la variable de intervalo)
        body_type = self.infer_type(ctx, path_abs.body)
        
        # Evaluar en los extremos
        left = path_abs.body.substitute(path_abs.var, Var("0"))
        right = path_abs.body.substitute(path_abs.var, Var("1"))
        
        # El tipo es Path A left right
        return PathType(body_type, left, right)
    
    def _infer_path_app(self, ctx: Context, path_app: PathApp) -> Type:
        """Infiere el tipo de una aplicación de camino."""
        # Inferir el tipo del camino
        path_type = self.infer_type(ctx, path_app.path)
        
        # Debe ser un tipo de caminos
        if not isinstance(path_type, PathType):
            raise TypeCheckError(f"Expected path type, got {path_type}")
        
        # El tipo resultado es el tipo base del camino
        return path_type.type_


# ============================================================================
# Funciones de utilidad
# ============================================================================

def type_check(ctx: Context, term: Term, expected_type: Type) -> bool:
    """
    Función de conveniencia para verificar tipos.
    
    Args:
        ctx: Contexto de tipado
        term: Término a verificar
        expected_type: Tipo esperado
    
    Returns:
        True si el término tiene el tipo esperado
    """
    checker = TypeChecker()
    return checker.check_type(ctx, term, expected_type)


def infer(ctx: Context, term: Term) -> Type:
    """
    Función de conveniencia para inferir tipos.
    
    Args:
        ctx: Contexto de tipado
        term: Término cuyo tipo se va a inferir
    
    Returns:
        Tipo inferido
    """
    checker = TypeChecker()
    return checker.infer_type(ctx, term)


def well_typed(ctx: Context, term: Term) -> Tuple[bool, Optional[Type]]:
    """
    Verifica si un término está bien tipado.
    
    Args:
        ctx: Contexto de tipado
        term: Término a verificar
    
    Returns:
        Tupla (está bien tipado, tipo inferido o None)
    """
    try:
        checker = TypeChecker()
        type_ = checker.infer_type(ctx, term)
        return (True, type_)
    except TypeCheckError:
        return (False, None)

