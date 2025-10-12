"""
Álgebra de Heyting para Lógica Intuicionista

Este módulo implementa un álgebra de Heyting, que es la estructura algebraica
subyacente a la lógica intuicionista. Un álgebra de Heyting es un retículo
distributivo acotado con una operación de implicación (→) que satisface:

    a ∧ b ≤ c  ⟺  a ≤ (b → c)

Esta propiedad se conoce como la adjunción de Galois o propiedad residual.

En el contexto de LatticeWeaver, el álgebra de Heyting proporciona:
1. Una semántica lógica para el retículo de conceptos (Capa 1)
2. Operaciones lógicas intuicionistas sobre conceptos
3. Una base formal para razonamiento constructivo

Diferencias con álgebra booleana:
- No se asume el tercio excluso (a ∨ ¬a = ⊤)
- La negación es constructiva: ¬a = (a → ⊥)
- La implicación no es definible desde otras operaciones

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import Any, Optional, Set, FrozenSet
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeytingElement:
    """
    Elemento de un álgebra de Heyting.
    
    Representa un concepto o proposición en la lógica intuicionista.
    Es inmutable (frozen) para poder usarse como clave de diccionario.
    
    Attributes:
        name: Nombre del elemento
        value: Valor opcional (puede ser un conjunto, un número, etc.)
    """
    name: str
    value: Optional[FrozenSet] = None
    
    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.name}={{{', '.join(map(str, sorted(self.value)))}}}"
        return self.name
    
    def __repr__(self) -> str:
        return f"HeytingElement({self.name!r})"
    
    def __lt__(self, other: 'HeytingElement') -> bool:
        """Orden lexicográfico para sorting."""
        return self.name < other.name


class HeytingAlgebra:
    """
    Álgebra de Heyting: retículo distributivo acotado con implicación.
    
    Un álgebra de Heyting es una estructura (H, ∧, ∨, →, ⊥, ⊤) donde:
    - (H, ∧, ∨, ⊥, ⊤) es un retículo distributivo acotado
    - → es la implicación de Heyting que satisface: a ∧ b ≤ c ⟺ a ≤ (b → c)
    - ¬a se define como a → ⊥
    
    Attributes:
        elements: Conjunto de elementos del álgebra
        bottom: Elemento mínimo (⊥, falsedad)
        top: Elemento máximo (⊤, verdad)
        _leq: Relación de orden parcial (≤)
        _meet_table: Tabla de ínfimos (∧, conjunción)
        _join_table: Tabla de supremos (∨, disyunción)
        _impl_table: Tabla de implicaciones (→)
    """
    
    def __init__(self, name: str = "H"):
        """
        Inicializa un álgebra de Heyting vacía.
        
        Args:
            name: Nombre del álgebra
        """
        self.name = name
        self.elements: Set[HeytingElement] = set()
        self.bottom: Optional[HeytingElement] = None
        self.top: Optional[HeytingElement] = None
        
        # Tablas de operaciones
        self._leq: Set[tuple[HeytingElement, HeytingElement]] = set()
        self._meet_table: dict[tuple[HeytingElement, HeytingElement], HeytingElement] = {}
        self._join_table: dict[tuple[HeytingElement, HeytingElement], HeytingElement] = {}
        self._impl_table: dict[tuple[HeytingElement, HeytingElement], HeytingElement] = {}
    
    def add_element(self, element: HeytingElement):
        """
        Añade un elemento al álgebra.
        
        Args:
            element: Elemento a añadir
        """
        self.elements.add(element)
        logger.debug(f"Añadido elemento {element} al álgebra {self.name}")
    
    def set_bottom(self, element: HeytingElement):
        """
        Establece el elemento mínimo (⊥).
        
        Args:
            element: Elemento mínimo
        """
        if element not in self.elements:
            self.add_element(element)
        self.bottom = element
        
        # ⊥ ≤ a para todo a
        for e in self.elements:
            self.add_order(element, e)
        
        logger.debug(f"Establecido ⊥ = {element}")
    
    def set_top(self, element: HeytingElement):
        """
        Establece el elemento máximo (⊤).
        
        Args:
            element: Elemento máximo
        """
        if element not in self.elements:
            self.add_element(element)
        self.top = element
        
        # a ≤ ⊤ para todo a
        for e in self.elements:
            self.add_order(e, element)
        
        logger.debug(f"Establecido ⊤ = {element}")
    
    def add_order(self, a: HeytingElement, b: HeytingElement):
        """
        Añade una relación de orden a ≤ b.
        
        Args:
            a: Elemento menor o igual
            b: Elemento mayor o igual
        """
        self._leq.add((a, b))
    
    def leq(self, a: HeytingElement, b: HeytingElement) -> bool:
        """
        Verifica si a ≤ b.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            True si a ≤ b, False en caso contrario
        """
        # Reflexividad
        if a == b:
            return True
        
        # Relación directa
        if (a, b) in self._leq:
            return True
        
        # Transitividad
        for c in self.elements:
            if (a, c) in self._leq and (c, b) in self._leq:
                return True
        
        return False
    
    def meet(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula el ínfimo (meet) a ∧ b.
        
        La conjunción lógica en el álgebra de Heyting.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Ínfimo de a y b
        """
        key = (a, b)
        if key in self._meet_table:
            return self._meet_table[key]
        
        # Buscar el mayor elemento c tal que c ≤ a y c ≤ b
        candidates = [c for c in self.elements if self.leq(c, a) and self.leq(c, b)]
        
        if not candidates:
            if self.bottom:
                return self.bottom
            raise ValueError(f"No se encontró ínfimo para {a} ∧ {b}")
        
        # Encontrar el máximo de los candidatos
        result = max(candidates, key=lambda c: sum(1 for d in self.elements if self.leq(d, c)))
        
        self._meet_table[key] = result
        self._meet_table[(b, a)] = result  # Conmutatividad
        
        return result
    
    def join(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula el supremo (join) a ∨ b.
        
        La disyunción lógica en el álgebra de Heyting.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            Supremo de a y b
        """
        # Caso especial: a ∨ a = a
        if a == b:
            return a
        
        key = (a, b)
        if key in self._join_table:
            return self._join_table[key]
        
        # Buscar el menor elemento c tal que a ≤ c y b ≤ c
        candidates = [c for c in self.elements if self.leq(a, c) and self.leq(b, c)]
        
        if not candidates:
            if self.top:
                return self.top
            raise ValueError(f"No se encontró supremo para {a} ∨ {b}")
        
        # Si los elementos tienen valores (conjuntos), usar la unión
        if a.value is not None and b.value is not None:
            union = a.value | b.value
            for c in candidates:
                if c.value == union:
                    self._join_table[key] = c
                    self._join_table[(b, a)] = c
                    return c
        
        # Encontrar el mínimo de los candidatos
        # (el que tiene menos elementos por encima de él)
        result = min(candidates, key=lambda c: sum(1 for d in self.elements if self.leq(c, d) and c != d))
        
        self._join_table[key] = result
        self._join_table[(b, a)] = result  # Conmutatividad
        
        return result
    
    def implies(self, a: HeytingElement, b: HeytingElement) -> HeytingElement:
        """
        Calcula la implicación de Heyting a → b.
        
        La implicación intuicionista, definida por:
            a ∧ c ≤ b  ⟺  c ≤ (a → b)
        
        Args:
            a: Antecedente
            b: Consecuente
        
        Returns:
            Implicación a → b
        """
        key = (a, b)
        if key in self._impl_table:
            return self._impl_table[key]
        
        # Buscar el mayor c tal que a ∧ c ≤ b
        candidates = [c for c in self.elements if self.leq(self.meet(a, c), b)]
        
        if not candidates:
            if self.bottom:
                return self.bottom
            raise ValueError(f"No se encontró implicación para {a} → {b}")
        
        # Encontrar el máximo de los candidatos
        result = max(candidates, key=lambda c: sum(1 for d in self.elements if self.leq(d, c)))
        
        self._impl_table[key] = result
        
        return result
    
    def neg(self, a: HeytingElement) -> HeytingElement:
        """
        Calcula la negación intuicionista ¬a.
        
        Definida como ¬a = (a → ⊥).
        
        Args:
            a: Elemento a negar
        
        Returns:
            Negación de a
        """
        if self.bottom is None:
            raise ValueError("No se puede calcular negación sin elemento ⊥")
        
        return self.implies(a, self.bottom)
    
    def is_valid(self) -> bool:
        """
        Verifica si la estructura es un álgebra de Heyting válida.
        
        Returns:
            True si es válida, False en caso contrario
        """
        if self.bottom is None or self.top is None:
            logger.warning("Álgebra sin ⊥ o ⊤")
            return False
        
        # Verificar que ⊥ ≤ a ≤ ⊤ para todo a
        for a in self.elements:
            if not self.leq(self.bottom, a) or not self.leq(a, self.top):
                logger.warning(f"Elemento {a} no cumple ⊥ ≤ a ≤ ⊤")
                return False
        
        # Verificar distributividad (simplificado)
        # a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
        for a in list(self.elements)[:5]:  # Muestra para eficiencia
            for b in list(self.elements)[:5]:
                for c in list(self.elements)[:5]:
                    try:
                        lhs = self.meet(a, self.join(b, c))
                        rhs = self.join(self.meet(a, b), self.meet(a, c))
                        if lhs != rhs:
                            logger.warning(f"Falla distributividad: {a} ∧ ({b} ∨ {c}) ≠ ({a} ∧ {b}) ∨ ({a} ∧ {c})")
                            return False
                    except ValueError:
                        continue
        
        return True
    
    def __repr__(self) -> str:
        """Representación en string del álgebra."""
        return f"HeytingAlgebra({self.name}, |H|={len(self.elements)})"
    
    def __str__(self) -> str:
        """Representación detallada del álgebra."""
        lines = [f"Álgebra de Heyting '{self.name}'"]
        lines.append(f"  Elementos: {len(self.elements)}")
        lines.append(f"  ⊥ = {self.bottom}")
        lines.append(f"  ⊤ = {self.top}")
        return "\n".join(lines)


def create_power_set_algebra(base_set: Set[Any], name: str = "P") -> HeytingAlgebra:
    """
    Crea un álgebra de Heyting desde el conjunto potencia de un conjunto base.
    
    El conjunto potencia con la inclusión (⊆) forma un álgebra de Heyting donde:
    - A ∧ B = A ∩ B (intersección)
    - A ∨ B = A ∪ B (unión)
    - A → B = (A^c ∪ B) (complemento de A unión B)
    - ¬A = A^c (complemento)
    
    Args:
        base_set: Conjunto base
        name: Nombre del álgebra
    
    Returns:
        Álgebra de Heyting construida
    """
    algebra = HeytingAlgebra(name)
    
    # Generar conjunto potencia
    base_list = list(base_set)
    n = len(base_list)
    
    # Mapeo de subconjuntos a elementos
    subset_to_element = {}
    
    for i in range(2**n):
        subset = frozenset(base_list[j] for j in range(n) if (i >> j) & 1)
        
        # Evitar duplicados
        if subset not in subset_to_element:
            element = HeytingElement(f"S{i}", subset)
            algebra.add_element(element)
            subset_to_element[subset] = element
    
    # Establecer ⊥ y ⊤ usando los elementos ya creados
    bottom = subset_to_element[frozenset()]
    top = subset_to_element[frozenset(base_set)]
    
    # No llamar a set_bottom/set_top porque ya están en elements
    # Solo establecer las referencias
    algebra.bottom = bottom
    algebra.top = top
    
    # Añadir órdenes para ⊥ y ⊤
    for elem in algebra.elements:
        algebra.add_order(bottom, elem)  # ⊥ ≤ a para todo a
        algebra.add_order(elem, top)     # a ≤ ⊤ para todo a
    
    # Establecer orden (inclusión)
    for a in algebra.elements:
        for b in algebra.elements:
            if a.value is not None and b.value is not None:
                if a.value.issubset(b.value):
                    algebra.add_order(a, b)
    
    return algebra

