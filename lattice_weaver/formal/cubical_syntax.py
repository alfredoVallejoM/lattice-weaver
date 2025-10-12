"""
Sintaxis Cúbica: AST para Teoría de Tipos Homotópica (HoTT)

Este módulo implementa un Abstract Syntax Tree (AST) para representar
términos y tipos de la Teoría de Tipos Homotópica con semántica cúbica.

La sintaxis cúbica es una forma de implementar HoTT que utiliza cubos
(intervalos unitarios) en lugar de simplices. Esto permite una mejor
computabilidad y verificación de pruebas.

Elementos principales:
- Tipos (Type, Pi, Sigma, Path, etc.)
- Términos (Var, Lambda, App, Pair, etc.)
- Contextos (Context)
- Sustituciones (Substitution)

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Optional, List, Dict, Set, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TIPOS (Types)
# ============================================================================

class Type(ABC):
    """Clase base para todos los tipos en HoTT."""
    
    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Retorna el conjunto de variables libres en el tipo."""
        pass
    
    @abstractmethod
    def substitute(self, var: str, term: 'Term') -> 'Type':
        """Sustituye una variable por un término."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Representación en string del tipo."""
        pass


@dataclass(frozen=True)
class Universe(Type):
    """
    Universo de tipos: Type_i
    
    En teoría de tipos, los tipos mismos son objetos de universos.
    Esto evita paradojas como la de Russell.
    
    Attributes:
        level: Nivel del universo (0, 1, 2, ...)
    """
    level: int = 0
    
    def free_vars(self) -> Set[str]:
        return set()
    
    def substitute(self, var: str, term: 'Term') -> 'Type':
        return self
    
    def __str__(self) -> str:
        return f"Type_{self.level}" if self.level > 0 else "Type"


@dataclass(frozen=True)
class TypeVar(Type):
    """
    Variable de tipo.
    
    Attributes:
        name: Nombre de la variable
    """
    name: str
    
    def free_vars(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, term: 'Term') -> 'Type':
        # Las variables de tipo no se sustituyen por términos
        return self
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PiType(Type):
    """
    Tipo dependiente Pi: Π(x : A). B(x)
    
    Generaliza el tipo función. Si B no depende de x, es A → B.
    
    Attributes:
        var: Variable ligada
        domain: Tipo del dominio (A)
        codomain: Tipo del codominio (B), que puede depender de var
    """
    var: str
    domain: Type
    codomain: Type
    
    def free_vars(self) -> Set[str]:
        return self.domain.free_vars() | (self.codomain.free_vars() - {self.var})
    
    def substitute(self, var: str, term: 'Term') -> 'Type':
        if var == self.var:
            # La variable está ligada, no sustituir en el codominio
            return PiType(self.var, self.domain.substitute(var, term), self.codomain)
        else:
            return PiType(
                self.var,
                self.domain.substitute(var, term),
                self.codomain.substitute(var, term)
            )
    
    def __str__(self) -> str:
        # Si el codominio no depende de var, usar notación →
        if self.var not in self.codomain.free_vars():
            return f"({self.domain} → {self.codomain})"
        return f"Π({self.var} : {self.domain}). {self.codomain}"


@dataclass(frozen=True)
class SigmaType(Type):
    """
    Tipo dependiente Sigma: Σ(x : A). B(x)
    
    Generaliza el tipo producto. Si B no depende de x, es A × B.
    
    Attributes:
        var: Variable ligada
        first: Tipo del primer componente (A)
        second: Tipo del segundo componente (B), que puede depender de var
    """
    var: str
    first: Type
    second: Type
    
    def free_vars(self) -> Set[str]:
        return self.first.free_vars() | (self.second.free_vars() - {self.var})
    
    def substitute(self, var: str, term: 'Term') -> 'Type':
        if var == self.var:
            return SigmaType(self.var, self.first.substitute(var, term), self.second)
        else:
            return SigmaType(
                self.var,
                self.first.substitute(var, term),
                self.second.substitute(var, term)
            )
    
    def __str__(self) -> str:
        # Si el segundo no depende de var, usar notación ×
        if self.var not in self.second.free_vars():
            return f"({self.first} × {self.second})"
        return f"Σ({self.var} : {self.first}). {self.second}"


@dataclass(frozen=True)
class PathType(Type):
    """
    Tipo de caminos (igualdad): Path A a b
    
    En HoTT, la igualdad es un tipo de caminos entre puntos.
    Path A a b representa los caminos de a a b en el tipo A.
    
    Attributes:
        type_: Tipo en el que viven los puntos
        left: Punto inicial
        right: Punto final
    """
    type_: Type
    left: 'Term'
    right: 'Term'
    
    def free_vars(self) -> Set[str]:
        return self.type_.free_vars() | self.left.free_vars() | self.right.free_vars()
    
    def substitute(self, var: str, term: 'Term') -> 'Type':
        return PathType(
            self.type_.substitute(var, term),
            self.left.substitute(var, term),
            self.right.substitute(var, term)
        )
    
    def __str__(self) -> str:
        return f"Path {self.type_} {self.left} {self.right}"


# ============================================================================
# TÉRMINOS (Terms)
# ============================================================================

class Term(ABC):
    """Clase base para todos los términos en HoTT."""
    
    @abstractmethod
    def free_vars(self) -> Set[str]:
        """Retorna el conjunto de variables libres en el término."""
        pass
    
    @abstractmethod
    def substitute(self, var: str, term: 'Term') -> 'Term':
        """Sustituye una variable por un término."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Representación en string del término."""
        pass


@dataclass(frozen=True)
class Var(Term):
    """
    Variable.
    
    Attributes:
        name: Nombre de la variable
    """
    name: str
    
    def free_vars(self) -> Set[str]:
        return {self.name}
    
    def substitute(self, var: str, term: Term) -> Term:
        if self.name == var:
            return term
        return self
    
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Lambda(Term):
    """
    Abstracción lambda: λ(x : A). t
    
    Attributes:
        var: Variable ligada
        type_: Tipo de la variable
        body: Cuerpo de la función
    """
    var: str
    type_: Type
    body: Term
    
    def free_vars(self) -> Set[str]:
        return (self.type_.free_vars() | self.body.free_vars()) - {self.var}
    
    def substitute(self, var: str, term: Term) -> Term:
        if var == self.var:
            # La variable está ligada, no sustituir en el cuerpo
            return Lambda(self.var, self.type_.substitute(var, term), self.body)
        else:
            return Lambda(
                self.var,
                self.type_.substitute(var, term),
                self.body.substitute(var, term)
            )
    
    def __str__(self) -> str:
        return f"λ({self.var} : {self.type_}). {self.body}"


@dataclass(frozen=True)
class App(Term):
    """
    Aplicación de función: f a
    
    Attributes:
        func: Función
        arg: Argumento
    """
    func: Term
    arg: Term
    
    def free_vars(self) -> Set[str]:
        return self.func.free_vars() | self.arg.free_vars()
    
    def substitute(self, var: str, term: Term) -> Term:
        return App(
            self.func.substitute(var, term),
            self.arg.substitute(var, term)
        )
    
    def __str__(self) -> str:
        return f"({self.func} {self.arg})"


@dataclass(frozen=True)
class Pair(Term):
    """
    Par: (a, b)
    
    Attributes:
        first: Primer componente
        second: Segundo componente
    """
    first: Term
    second: Term
    
    def free_vars(self) -> Set[str]:
        return self.first.free_vars() | self.second.free_vars()
    
    def substitute(self, var: str, term: Term) -> Term:
        return Pair(
            self.first.substitute(var, term),
            self.second.substitute(var, term)
        )
    
    def __str__(self) -> str:
        return f"({self.first}, {self.second})"


@dataclass(frozen=True)
class Fst(Term):
    """
    Primera proyección: fst p
    
    Attributes:
        pair: Par del que extraer el primer componente
    """
    pair: Term
    
    def free_vars(self) -> Set[str]:
        return self.pair.free_vars()
    
    def substitute(self, var: str, term: Term) -> Term:
        return Fst(self.pair.substitute(var, term))
    
    def __str__(self) -> str:
        return f"fst {self.pair}"


@dataclass(frozen=True)
class Snd(Term):
    """
    Segunda proyección: snd p
    
    Attributes:
        pair: Par del que extraer el segundo componente
    """
    pair: Term
    
    def free_vars(self) -> Set[str]:
        return self.pair.free_vars()
    
    def substitute(self, var: str, term: Term) -> Term:
        return Snd(self.pair.substitute(var, term))
    
    def __str__(self) -> str:
        return f"snd {self.pair}"


@dataclass(frozen=True)
class Refl(Term):
    """
    Reflexividad: refl a
    
    El camino trivial de un punto a sí mismo.
    
    Attributes:
        point: Punto
    """
    point: Term
    
    def free_vars(self) -> Set[str]:
        return self.point.free_vars()
    
    def substitute(self, var: str, term: Term) -> Term:
        return Refl(self.point.substitute(var, term))
    
    def __str__(self) -> str:
        return f"refl {self.point}"


@dataclass(frozen=True)
class PathAbs(Term):
    """
    Abstracción de camino: <i> t
    
    Construye un camino parametrizado por la variable de intervalo i.
    
    Attributes:
        var: Variable de intervalo (i ∈ [0,1])
        body: Cuerpo del camino
    """
    var: str
    body: Term
    
    def free_vars(self) -> Set[str]:
        return self.body.free_vars() - {self.var}
    
    def substitute(self, var: str, term: Term) -> Term:
        if var == self.var:
            return self
        return PathAbs(self.var, self.body.substitute(var, term))
    
    def __str__(self) -> str:
        return f"<{self.var}> {self.body}"


@dataclass(frozen=True)
class PathApp(Term):
    """
    Aplicación de camino: p @ i
    
    Evalúa un camino en un punto del intervalo.
    
    Attributes:
        path: Camino
        point: Punto del intervalo (0, 1, o variable)
    """
    path: Term
    point: str  # "0", "1", o nombre de variable
    
    def free_vars(self) -> Set[str]:
        vars = self.path.free_vars()
        if self.point not in ["0", "1"]:
            vars.add(self.point)
        return vars
    
    def substitute(self, var: str, term: Term) -> Term:
        # No sustituir puntos del intervalo
        return PathApp(self.path.substitute(var, term), self.point)
    
    def __str__(self) -> str:
        return f"({self.path} @ {self.point})"


# ============================================================================
# CONTEXTOS Y JUICIOS
# ============================================================================

@dataclass
class Binding:
    """
    Ligadura de variable en un contexto.
    
    Attributes:
        var: Nombre de la variable
        type_: Tipo de la variable
    """
    var: str
    type_: Type
    
    def __str__(self) -> str:
        return f"{self.var} : {self.type_}"


class Context:
    """
    Contexto de tipado: Γ = x₁:A₁, x₂:A₂, ..., xₙ:Aₙ
    
    Mantiene las asunciones sobre los tipos de las variables.
    """
    
    def __init__(self):
        """Inicializa un contexto vacío."""
        self.bindings: List[Binding] = []
    
    def extend(self, var: str, type_: Type) -> 'Context':
        """
        Extiende el contexto con una nueva ligadura.
        
        Args:
            var: Variable a añadir
            type_: Tipo de la variable
        
        Returns:
            Nuevo contexto extendido
        """
        new_ctx = Context()
        new_ctx.bindings = self.bindings + [Binding(var, type_)]
        return new_ctx
    
    def lookup(self, var: str) -> Optional[Type]:
        """
        Busca el tipo de una variable en el contexto.
        
        Args:
            var: Variable a buscar
        
        Returns:
            Tipo de la variable, o None si no está en el contexto
        """
        for binding in reversed(self.bindings):
            if binding.var == var:
                return binding.type_
        return None
    
    def __str__(self) -> str:
        if not self.bindings:
            return "∅"
        return ", ".join(str(b) for b in self.bindings)
    
    def __repr__(self) -> str:
        return f"Context({self})"


# ============================================================================
# UTILIDADES
# ============================================================================

def arrow_type(domain: Type, codomain: Type) -> PiType:
    """
    Construye un tipo función simple A → B.
    
    Args:
        domain: Tipo del dominio
        codomain: Tipo del codominio
    
    Returns:
        Tipo Pi no dependiente
    """
    return PiType("_", domain, codomain)


def product_type(first: Type, second: Type) -> SigmaType:
    """
    Construye un tipo producto simple A × B.
    
    Args:
        first: Tipo del primer componente
        second: Tipo del segundo componente
    
    Returns:
        Tipo Sigma no dependiente
    """
    return SigmaType("_", first, second)


def identity_type(type_: Type, left: Term, right: Term) -> PathType:
    """
    Construye un tipo de identidad Id_A(a, b).
    
    Args:
        type_: Tipo en el que viven los términos
        left: Término izquierdo
        right: Término derecho
    
    Returns:
        Tipo de caminos
    """
    return PathType(type_, left, right)

