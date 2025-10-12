"""
Locale Theory - Topología sin Puntos (Point-Free Topology)

Este módulo implementa la teoría de Locales y Frames, proporcionando una base
formal para el razonamiento topológico constructivo en LatticeWeaver.

Estructuras implementadas:
- PartialOrder: Conjunto parcialmente ordenado (poset)
- CompleteLattice: Retículo completo con supremos/ínfimos arbitrarios
- Frame: Retículo completo de Heyting con ley distributiva infinita
- Locale: Dual categórico de un Frame (espacio topológico sin puntos)

Fundamentos matemáticos:
- Los Frames son álgebras de Heyting completas
- Los Locales representan espacios topológicos de forma constructiva
- Conexión profunda con Formal Concept Analysis (FCA)
- Base para la teoría de Sheaves (Meseta 2)

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Set, FrozenSet, Tuple, Dict, Any, Hashable, Optional, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Utilidades
# ============================================================================

class FrozenDict(dict):
    """
    Diccionario inmutable (hashable).
    
    Permite usar diccionarios como claves o en conjuntos frozen.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hash = None
    
    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self.items()))
        return self._hash
    
    def __setitem__(self, key, value):
        raise TypeError("FrozenDict is immutable")
    
    def __delitem__(self, key):
        raise TypeError("FrozenDict is immutable")
    
    def clear(self):
        raise TypeError("FrozenDict is immutable")
    
    def pop(self, *args):
        raise TypeError("FrozenDict is immutable")
    
    def popitem(self):
        raise TypeError("FrozenDict is immutable")
    
    def setdefault(self, key, default=None):
        raise TypeError("FrozenDict is immutable")
    
    def update(self, *args, **kwargs):
        raise TypeError("FrozenDict is immutable")


# ============================================================================
# PartialOrder - Conjunto Parcialmente Ordenado
# ============================================================================

@dataclass(frozen=True)
class PartialOrder:
    """
    Conjunto parcialmente ordenado (poset).
    
    Un poset es una estructura (P, ≤) donde:
    1. Reflexividad: ∀a ∈ P, a ≤ a
    2. Antisimetría: ∀a,b ∈ P, (a ≤ b ∧ b ≤ a) → a = b
    3. Transitividad: ∀a,b,c ∈ P, (a ≤ b ∧ b ≤ c) → a ≤ c
    
    Attributes:
        elements: Conjunto de elementos del poset
        leq: Relación de orden ≤ (como conjunto de pares)
        _cache: Caché para operaciones costosas (no participa en comparación)
    
    Examples:
        >>> # Crear poset de divisores de 12
        >>> elements = frozenset({1, 2, 3, 4, 6, 12})
        >>> leq = frozenset({
        ...     (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 12),
        ...     (2, 2), (2, 4), (2, 6), (2, 12),
        ...     (3, 3), (3, 6), (3, 12),
        ...     (4, 4), (4, 12),
        ...     (6, 6), (6, 12),
        ...     (12, 12)
        ... })
        >>> poset = PartialOrder(elements, leq)
    
    Notes:
        - La relación leq debe ser completa (incluir todos los pares)
        - La verificación de axiomas se hace en __post_init__
        - La estructura es inmutable (thread-safe)
    """
    
    elements: FrozenSet[Hashable]
    leq: FrozenSet[Tuple[Hashable, Hashable]]
    _cache: Dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    
    def __post_init__(self):
        """
        Verifica los axiomas de orden parcial.
        
        Raises:
            ValueError: Si algún axioma es violado
        """
        self._verify_reflexive()
        self._verify_antisymmetric()
        self._verify_transitive()
        
        logger.debug(f"PartialOrder creado con {len(self.elements)} elementos")
    
    def _verify_reflexive(self):
        """
        Verifica reflexividad: ∀a ∈ P, a ≤ a.
        
        Raises:
            ValueError: Si la reflexividad es violada
        """
        for elem in self.elements:
            if (elem, elem) not in self.leq:
                raise ValueError(
                    f"Reflexividad violada: ({elem}, {elem}) no está en la relación"
                )
    
    def _verify_antisymmetric(self):
        """
        Verifica antisimetría: ∀a,b ∈ P, (a ≤ b ∧ b ≤ a) → a = b.
        
        Raises:
            ValueError: Si la antisimetría es violada
        """
        for a in self.elements:
            for b in self.elements:
                if a != b and (a, b) in self.leq and (b, a) in self.leq:
                    raise ValueError(
                        f"Antisimetría violada: {a} ≤ {b} y {b} ≤ {a} pero {a} ≠ {b}"
                    )
    
    def _verify_transitive(self):
        """
        Verifica transitividad: ∀a,b,c ∈ P, (a ≤ b ∧ b ≤ c) → a ≤ c.
        
        Raises:
            ValueError: Si la transitividad es violada
        """
        for a in self.elements:
            for b in self.elements:
                if (a, b) in self.leq:
                    for c in self.elements:
                        if (b, c) in self.leq and (a, c) not in self.leq:
                            raise ValueError(
                                f"Transitividad violada: {a} ≤ {b} y {b} ≤ {c} "
                                f"pero {a} ≰ {c}"
                            )
    
    def is_leq(self, a: Hashable, b: Hashable) -> bool:
        """
        Verifica si a ≤ b.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            True si a ≤ b, False en caso contrario
        
        Raises:
            ValueError: Si a o b no están en el poset
        
        Examples:
            >>> poset.is_leq(2, 6)  # 2 divide a 6
            True
            >>> poset.is_leq(3, 4)  # 3 no divide a 4
            False
        """
        if a not in self.elements:
            raise ValueError(f"{a} no está en el poset")
        if b not in self.elements:
            raise ValueError(f"{b} no está en el poset")
        
        return (a, b) in self.leq
    
    def is_less(self, a: Hashable, b: Hashable) -> bool:
        """
        Verifica si a < b (orden estricto).
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            True si a < b (a ≤ b y a ≠ b), False en caso contrario
        """
        return a != b and self.is_leq(a, b)
    
    def upper_bounds(self, subset: Set[Hashable]) -> Set[Hashable]:
        """
        Calcula las cotas superiores de un subconjunto.
        
        Una cota superior de S es un elemento b tal que ∀a ∈ S, a ≤ b.
        
        Args:
            subset: Subconjunto de elementos
        
        Returns:
            Conjunto de cotas superiores
        
        Examples:
            >>> poset.upper_bounds({2, 3})
            {6, 12}  # 6 y 12 son divisibles por 2 y 3
        
        Notes:
            - Resultado cacheado para eficiencia
            - Complejidad: O(|P| * |S|)
        """
        # Validar subset
        if not subset.issubset(self.elements):
            invalid = subset - self.elements
            raise ValueError(f"Elementos no válidos en subset: {invalid}")
        
        # Caso especial: conjunto vacío
        if not subset:
            return set(self.elements)
        
        # Buscar en caché
        key = ('upper_bounds', frozenset(subset))
        if key in self._cache:
            return self._cache[key].copy()
        
        # Calcular cotas superiores
        upper = {
            b for b in self.elements
            if all(self.is_leq(a, b) for a in subset)
        }
        
        # Cachear resultado
        self._cache[key] = upper.copy()
        
        return upper
    
    def lower_bounds(self, subset: Set[Hashable]) -> Set[Hashable]:
        """
        Calcula las cotas inferiores de un subconjunto.
        
        Una cota inferior de S es un elemento a tal que ∀b ∈ S, a ≤ b.
        
        Args:
            subset: Subconjunto de elementos
        
        Returns:
            Conjunto de cotas inferiores
        
        Examples:
            >>> poset.lower_bounds({6, 12})
            {1, 2, 3, 6}  # Divisores comunes de 6 y 12
        
        Notes:
            - Resultado cacheado para eficiencia
            - Complejidad: O(|P| * |S|)
        """
        # Validar subset
        if not subset.issubset(self.elements):
            invalid = subset - self.elements
            raise ValueError(f"Elementos no válidos en subset: {invalid}")
        
        # Caso especial: conjunto vacío
        if not subset:
            return set(self.elements)
        
        # Buscar en caché
        key = ('lower_bounds', frozenset(subset))
        if key in self._cache:
            return self._cache[key].copy()
        
        # Calcular cotas inferiores
        lower = {
            a for a in self.elements
            if all(self.is_leq(a, b) for b in subset)
        }
        
        # Cachear resultado
        self._cache[key] = lower.copy()
        
        return lower
    
    def minimal_elements(self, subset: Optional[Set[Hashable]] = None) -> Set[Hashable]:
        """
        Calcula los elementos minimales de un subconjunto.
        
        Un elemento a es minimal en S si no existe b ∈ S con b < a.
        
        Args:
            subset: Subconjunto de elementos (None = todo el poset)
        
        Returns:
            Conjunto de elementos minimales
        
        Examples:
            >>> poset.minimal_elements()
            {1}  # 1 es el único minimal en divisores de 12
        """
        if subset is None:
            subset = set(self.elements)
        
        minimal = set()
        for a in subset:
            is_minimal = True
            for b in subset:
                if b != a and self.is_less(b, a):
                    is_minimal = False
                    break
            if is_minimal:
                minimal.add(a)
        
        return minimal
    
    def maximal_elements(self, subset: Optional[Set[Hashable]] = None) -> Set[Hashable]:
        """
        Calcula los elementos maximales de un subconjunto.
        
        Un elemento a es maximal en S si no existe b ∈ S con a < b.
        
        Args:
            subset: Subconjunto de elementos (None = todo el poset)
        
        Returns:
            Conjunto de elementos maximales
        
        Examples:
            >>> poset.maximal_elements()
            {12}  # 12 es el único maximal en divisores de 12
        """
        if subset is None:
            subset = set(self.elements)
        
        maximal = set()
        for a in subset:
            is_maximal = True
            for b in subset:
                if b != a and self.is_less(a, b):
                    is_maximal = False
                    break
            if is_maximal:
                maximal.add(a)
        
        return maximal
    
    def comparable(self, a: Hashable, b: Hashable) -> bool:
        """
        Verifica si dos elementos son comparables.
        
        a y b son comparables si a ≤ b o b ≤ a.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            True si son comparables, False en caso contrario
        """
        return self.is_leq(a, b) or self.is_leq(b, a)
    
    def is_chain(self, subset: Set[Hashable]) -> bool:
        """
        Verifica si un subconjunto es una cadena (totalmente ordenado).
        
        Un subconjunto S es una cadena si todo par de elementos es comparable.
        
        Args:
            subset: Subconjunto de elementos
        
        Returns:
            True si es una cadena, False en caso contrario
        
        Examples:
            >>> poset.is_chain({1, 2, 4, 12})
            True  # 1 | 2 | 4 | 12
            >>> poset.is_chain({2, 3})
            False  # 2 y 3 no son comparables
        """
        for a in subset:
            for b in subset:
                if not self.comparable(a, b):
                    return False
        return True
    
    def is_antichain(self, subset: Set[Hashable]) -> bool:
        """
        Verifica si un subconjunto es una anticadena.
        
        Un subconjunto S es una anticadena si ningún par de elementos distintos
        es comparable.
        
        Args:
            subset: Subconjunto de elementos
        
        Returns:
            True si es una anticadena, False en caso contrario
        
        Examples:
            >>> poset.is_antichain({2, 3})
            True  # 2 y 3 no son comparables
            >>> poset.is_antichain({2, 4})
            False  # 2 ≤ 4
        """
        for a in subset:
            for b in subset:
                if a != b and self.comparable(a, b):
                    return False
        return True
    
    def hasse_diagram_edges(self) -> Set[Tuple[Hashable, Hashable]]:
        """
        Calcula las aristas del diagrama de Hasse.
        
        El diagrama de Hasse muestra solo las relaciones de cobertura:
        a cubre b si a < b y no existe c con a < c < b.
        
        Returns:
            Conjunto de aristas (a, b) donde a cubre b
        
        Notes:
            - Útil para visualización
            - Reduce la relación de orden a su forma minimal
        """
        covers = set()
        
        for a in self.elements:
            for b in self.elements:
                if self.is_less(a, b):
                    # Verificar si a cubre b (no hay elemento intermedio)
                    is_cover = True
                    for c in self.elements:
                        if c != a and c != b:
                            if self.is_less(a, c) and self.is_less(c, b):
                                is_cover = False
                                break
                    
                    if is_cover:
                        covers.add((a, b))
        
        return covers
    
    def __len__(self) -> int:
        """Retorna el número de elementos en el poset."""
        return len(self.elements)
    
    def __contains__(self, elem: Hashable) -> bool:
        """Verifica si un elemento está en el poset."""
        return elem in self.elements
    
    def __iter__(self) -> Iterator[Hashable]:
        """Itera sobre los elementos del poset."""
        return iter(self.elements)
    
    def __repr__(self) -> str:
        """Representación en string del poset."""
        return f"PartialOrder(|P|={len(self.elements)})"


# ============================================================================
# CompleteLattice - Retículo Completo
# ============================================================================

@dataclass(frozen=True)
class CompleteLattice:
    """
    Retículo completo.
    
    Un retículo completo es un poset donde todo subconjunto (incluso infinito)
    tiene supremo (join) e ínfimo (meet).
    
    Propiedades:
    - Todo retículo completo tiene elemento máximo (top) y mínimo (bottom)
    - top = ⋁ P (supremo de todos los elementos)
    - bottom = ⋀ P (ínfimo de todos los elementos)
    - ⋁ ∅ = bottom
    - ⋀ ∅ = top
    
    Attributes:
        poset: Orden parcial subyacente
        top: Elemento máximo (⊤)
        bottom: Elemento mínimo (⊥)
        _join_cache: Caché de supremos calculados
        _meet_cache: Caché de ínfimos calculados
    
    Examples:
        >>> # Retículo de divisores de 12
        >>> lattice = CompleteLattice(poset, top=12, bottom=1)
        >>> lattice.join({2, 3})  # mcm(2, 3) = 6
        6
        >>> lattice.meet({6, 12})  # mcd(6, 12) = 6
        6
    
    Notes:
        - Los supremos/ínfimos se calculan como cotas superiores/inferiores minimales
        - Resultados cacheados para eficiencia
        - Inmutable y thread-safe
    """
    
    poset: PartialOrder
    top: Hashable
    bottom: Hashable
    _join_cache: Dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    _meet_cache: Dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    
    def __post_init__(self):
        """
        Verifica que top y bottom sean válidos.
        
        Raises:
            ValueError: Si top o bottom no están en el poset o son incorrectos
        """
        if self.top not in self.poset.elements:
            raise ValueError(f"Top {self.top} no está en el poset")
        if self.bottom not in self.poset.elements:
            raise ValueError(f"Bottom {self.bottom} no está en el poset")
        
        # Verificar que top es máximo
        for elem in self.poset.elements:
            if not self.poset.is_leq(elem, self.top):
                raise ValueError(
                    f"Top {self.top} no es máximo: {elem} ≰ {self.top}"
                )
        
        # Verificar que bottom es mínimo
        for elem in self.poset.elements:
            if not self.poset.is_leq(self.bottom, elem):
                raise ValueError(
                    f"Bottom {self.bottom} no es mínimo: {self.bottom} ≰ {elem}"
                )
        
        logger.debug(f"CompleteLattice creado: ⊤={self.top}, ⊥={self.bottom}")
    
    def join(self, subset: Set[Hashable]) -> Hashable:
        """
        Calcula el supremo (join, ⋁) de un subconjunto.
        
        El supremo de S es la menor cota superior de S.
        
        Args:
            subset: Subconjunto de elementos
        
        Returns:
            Supremo del subconjunto
        
        Raises:
            ValueError: Si el subconjunto no tiene supremo (no debería pasar en retículo completo)
        
        Examples:
            >>> lattice.join({2, 3})
            6  # mcm(2, 3) = 6
            >>> lattice.join(set())
            1  # ⋁ ∅ = ⊥
            >>> lattice.join({2})
            2  # ⋁ {a} = a
        
        Notes:
            - Complejidad: O(|P|²) en el peor caso
            - Resultado cacheado
            - ⋁ ∅ = ⊥ por convención
        """
        # Caso especial: conjunto vacío
        if not subset:
            return self.bottom
        
        # Caso especial: singleton
        if len(subset) == 1:
            return next(iter(subset))
        
        # Buscar en caché
        key = frozenset(subset)
        if key in self._join_cache:
            return self._join_cache[key]
        
        # Calcular cotas superiores
        upper = self.poset.upper_bounds(subset)
        
        if not upper:
            raise ValueError(f"No hay cotas superiores para {subset}")
        
        # El supremo es la cota superior minimal
        # (en un retículo completo, es única)
        minimal_upper = self.poset.minimal_elements(upper)
        
        if len(minimal_upper) != 1:
            # Esto no debería pasar en un retículo completo
            raise ValueError(
                f"Múltiples cotas superiores minimales para {subset}: {minimal_upper}"
            )
        
        supremum = next(iter(minimal_upper))
        
        # Cachear resultado
        self._join_cache[key] = supremum
        
        return supremum
    
    def meet(self, subset: Set[Hashable]) -> Hashable:
        """
        Calcula el ínfimo (meet, ⋀) de un subconjunto.
        
        El ínfimo de S es la mayor cota inferior de S.
        
        Args:
            subset: Subconjunto de elementos
        
        Returns:
            Ínfimo del subconjunto
        
        Raises:
            ValueError: Si el subconjunto no tiene ínfimo (no debería pasar en retículo completo)
        
        Examples:
            >>> lattice.meet({6, 12})
            6  # mcd(6, 12) = 6
            >>> lattice.meet(set())
            12  # ⋀ ∅ = ⊤
            >>> lattice.meet({3})
            3  # ⋀ {a} = a
        
        Notes:
            - Complejidad: O(|P|²) en el peor caso
            - Resultado cacheado
            - ⋀ ∅ = ⊤ por convención
        """
        # Caso especial: conjunto vacío
        if not subset:
            return self.top
        
        # Caso especial: singleton
        if len(subset) == 1:
            return next(iter(subset))
        
        # Buscar en caché
        key = frozenset(subset)
        if key in self._meet_cache:
            return self._meet_cache[key]
        
        # Calcular cotas inferiores
        lower = self.poset.lower_bounds(subset)
        
        if not lower:
            raise ValueError(f"No hay cotas inferiores para {subset}")
        
        # El ínfimo es la cota inferior maximal
        # (en un retículo completo, es única)
        maximal_lower = self.poset.maximal_elements(lower)
        
        if len(maximal_lower) != 1:
            # Esto no debería pasar en un retículo completo
            raise ValueError(
                f"Múltiples cotas inferiores maximales para {subset}: {maximal_lower}"
            )
        
        infimum = next(iter(maximal_lower))
        
        # Cachear resultado
        self._meet_cache[key] = infimum
        
        return infimum
    
    def join_binary(self, a: Hashable, b: Hashable) -> Hashable:
        """
        Supremo binario: a ∨ b.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            a ∨ b
        """
        return self.join({a, b})
    
    def meet_binary(self, a: Hashable, b: Hashable) -> Hashable:
        """
        Ínfimo binario: a ∧ b.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            a ∧ b
        """
        return self.meet({a, b})
    
    def __repr__(self) -> str:
        """Representación en string del retículo."""
        return f"CompleteLattice(|L|={len(self.poset.elements)}, ⊤={self.top}, ⊥={self.bottom})"


# ============================================================================
# Construcciones de Retículos
# ============================================================================

class LatticeBuilder:
    """
    Constructor de retículos completos desde diferentes fuentes.
    
    Proporciona métodos estáticos para construir retículos desde:
    - Conjuntos potencia (powerset)
    - Divisores de un número
    - Subgrupos de un grupo
    - Contextos formales (FCA)
    """
    
    @staticmethod
    def from_powerset(base_set: Set[Hashable]) -> CompleteLattice:
        """
        Construye el retículo del conjunto potencia P(S).
        
        El conjunto potencia con inclusión (⊆) forma un retículo completo donde:
        - join = unión (∪)
        - meet = intersección (∩)
        - top = S
        - bottom = ∅
        
        Args:
            base_set: Conjunto base S
        
        Returns:
            Retículo completo P(S)
        
        Examples:
            >>> lattice = LatticeBuilder.from_powerset({1, 2, 3})
            >>> lattice.join({frozenset({1}), frozenset({2})})
            frozenset({1, 2})
        
        Notes:
            - Tamaño: 2^|S|
            - Útil para modelar atributos en FCA
        """
        # Generar conjunto potencia
        base_list = list(base_set)
        powerset_elements = set()
        
        for r in range(len(base_list) + 1):
            for subset in itertools.combinations(base_list, r):
                powerset_elements.add(frozenset(subset))
        
        powerset_elements = frozenset(powerset_elements)
        
        # Relación de orden: inclusión
        leq = frozenset(
            (a, b)
            for a in powerset_elements
            for b in powerset_elements
            if a.issubset(b)
        )
        
        # Crear poset
        poset = PartialOrder(powerset_elements, leq)
        
        # Top y bottom
        top = frozenset(base_set)
        bottom = frozenset()
        
        return CompleteLattice(poset, top, bottom)
    
    @staticmethod
    def from_divisors(n: int) -> CompleteLattice:
        """
        Construye el retículo de divisores de n.
        
        Los divisores de n con la relación de divisibilidad (|) forman un
        retículo completo donde:
        - join = mcm (mínimo común múltiplo)
        - meet = mcd (máximo común divisor)
        - top = n
        - bottom = 1
        
        Args:
            n: Número entero positivo
        
        Returns:
            Retículo de divisores de n
        
        Examples:
            >>> lattice = LatticeBuilder.from_divisors(12)
            >>> lattice.join({2, 3})
            6  # mcm(2, 3) = 6
        
        Notes:
            - Solo para n > 0
            - Útil para ejemplos y tests
        """
        if n <= 0:
            raise ValueError("n debe ser positivo")
        
        # Calcular divisores
        divisors = set()
        for i in range(1, n + 1):
            if n % i == 0:
                divisors.add(i)
        
        divisors = frozenset(divisors)
        
        # Relación de orden: divisibilidad
        leq = frozenset(
            (a, b)
            for a in divisors
            for b in divisors
            if b % a == 0
        )
        
        # Crear poset
        poset = PartialOrder(divisors, leq)
        
        return CompleteLattice(poset, top=n, bottom=1)




# ============================================================================
# Frame - Retículo Completo de Heyting
# ============================================================================

@dataclass(frozen=True)
class Frame(CompleteLattice):
    """
    Frame (retículo completo de Heyting).
    
    Un Frame es un retículo completo que satisface la **ley distributiva infinita**:
        a ∧ (⋁ S) = ⋁ {a ∧ s | s ∈ S}
    
    Los Frames son las álgebras de Heyting completas, lo que significa que tienen:
    - Implicación intuicionista: a → b
    - Negación intuicionista: ¬a = a → ⊥
    
    Propiedades:
    - Todo Frame es un retículo completo
    - La ley distributiva infinita es la característica definitoria
    - Los morfismos de Frames preservan supremos arbitrarios e ínfimos finitos
    
    Attributes:
        Hereda de CompleteLattice
        _implication_cache: Caché de implicaciones de Heyting
    
    Examples:
        >>> # Frame de abiertos de un espacio topológico
        >>> # (simplificado: powerset con unión e intersección)
        >>> frame = FrameBuilder.from_powerset({1, 2, 3})
        >>> a = frozenset({1})
        >>> b = frozenset({1, 2})
        >>> frame.heyting_implication(a, b)
        frozenset({1, 2, 3})  # a → b = ⊤ cuando a ⊆ b
    
    Notes:
        - La verificación completa de la ley distributiva infinita es intratable
        - Verificamos una muestra representativa en __post_init__
        - Inmutable y thread-safe
    """
    
    _implication_cache: Dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    
    def __post_init__(self):
        """
        Verifica la ley distributiva infinita (muestra).
        
        Raises:
            ValueError: Si la ley distributiva es violada
        """
        # Llamar a __post_init__ de CompleteLattice
        super().__post_init__()
        
        # Verificar ley distributiva infinita en muestra
        self._verify_infinite_distributivity_sample()
        
        logger.debug(f"Frame creado y verificado")
    
    def _verify_infinite_distributivity_sample(self):
        """
        Verifica la ley distributiva infinita en una muestra.
        
        Ley: a ∧ (⋁ S) = ⋁ {a ∧ s | s ∈ S}
        
        Verificamos para:
        - Subconjuntos pequeños (|S| ≤ 3)
        - Muestra de elementos a
        
        Raises:
            ValueError: Si la ley es violada
        
        Notes:
            - Verificación completa es O(2^|P|), intratable
            - Esta muestra detecta errores obvios
        """
        # Limitar verificación a posets pequeños
        if len(self.poset.elements) > 20:
            logger.warning(
                f"Frame grande ({len(self.poset.elements)} elementos), "
                "verificación de distributividad limitada"
            )
            sample_size = 5
        else:
            sample_size = min(10, len(self.poset.elements))
        
        # Muestra de elementos
        sample_elements = list(self.poset.elements)[:sample_size]
        
        for a in sample_elements:
            # Verificar para subconjuntos de tamaño 1, 2, 3
            for subset_size in range(1, min(4, len(self.poset.elements) + 1)):
                for subset_tuple in itertools.combinations(self.poset.elements, subset_size):
                    subset = set(subset_tuple)
                    
                    # LHS: a ∧ (⋁ S)
                    join_s = self.join(subset)
                    lhs = self.meet_binary(a, join_s)
                    
                    # RHS: ⋁ {a ∧ s | s ∈ S}
                    meets = {self.meet_binary(a, s) for s in subset}
                    rhs = self.join(meets)
                    
                    if lhs != rhs:
                        raise ValueError(
                            f"Ley distributiva infinita violada:\n"
                            f"  {a} ∧ (⋁ {subset}) = {lhs}\n"
                            f"  ⋁ {{{a} ∧ s | s ∈ {subset}}} = {rhs}\n"
                            f"  {lhs} ≠ {rhs}"
                        )
    
    def heyting_implication(self, a: Hashable, b: Hashable) -> Hashable:
        """
        Calcula la implicación de Heyting a → b.
        
        En un Frame, la implicación se define como:
            a → b = ⋁ {c ∈ L | c ∧ a ≤ b}
        
        Propiedades:
        - a → b = ⊤ si y solo si a ≤ b
        - a → ⊥ = ¬a (negación)
        - Residuación: c ≤ (a → b) ⟺ c ∧ a ≤ b
        
        Args:
            a: Premisa
            b: Conclusión
        
        Returns:
            Implicación a → b
        
        Examples:
            >>> # En powerset: A → B = ¬A ∪ B
            >>> frame.heyting_implication(frozenset({1}), frozenset({1, 2}))
            frozenset({2, 3})  # ¬{1} ∪ {1,2} = {2,3} ∪ {1,2} = {1,2,3}
        
        Notes:
            - Resultado cacheado
            - Complejidad: O(|P|²) en el peor caso
        """
        # Buscar en caché
        key = (a, b)
        if key in self._implication_cache:
            return self._implication_cache[key]
        
        # Calcular conjunto de c tal que c ∧ a ≤ b
        candidates = {
            c for c in self.poset.elements
            if self.poset.is_leq(self.meet_binary(c, a), b)
        }
        
        # La implicación es el supremo de los candidatos
        implication = self.join(candidates)
        
        # Cachear
        self._implication_cache[key] = implication
        
        return implication
    
    def heyting_negation(self, a: Hashable) -> Hashable:
        """
        Calcula la negación de Heyting ¬a.
        
        La negación se define como:
            ¬a = a → ⊥
        
        Propiedades:
        - ¬¬a ≥ a (pero no necesariamente ¬¬a = a, lógica intuicionista)
        - ¬⊤ = ⊥
        - ¬⊥ = ⊤
        
        Args:
            a: Elemento a negar
        
        Returns:
            Negación ¬a
        
        Examples:
            >>> # En powerset: ¬A = complemento de A
            >>> frame.heyting_negation(frozenset({1}))
            frozenset({2, 3})
        """
        return self.heyting_implication(a, self.bottom)
    
    def is_regular(self, a: Hashable) -> bool:
        """
        Verifica si un elemento es regular.
        
        Un elemento a es regular si ¬¬a = a.
        Los elementos regulares forman un álgebra de Boole.
        
        Args:
            a: Elemento a verificar
        
        Returns:
            True si a es regular, False en caso contrario
        
        Notes:
            - En un álgebra de Boole, todos los elementos son regulares
            - En un Frame general, puede haber elementos no regulares
        """
        neg_neg_a = self.heyting_negation(self.heyting_negation(a))
        return neg_neg_a == a
    
    def regular_elements(self) -> Set[Hashable]:
        """
        Calcula el conjunto de elementos regulares.
        
        Returns:
            Conjunto de elementos a tal que ¬¬a = a
        
        Notes:
            - Los elementos regulares forman un sub-álgebra de Boole
            - Útil para análisis de la estructura del Frame
        """
        return {a for a in self.poset.elements if self.is_regular(a)}
    
    def __repr__(self) -> str:
        """Representación en string del Frame."""
        return f"Frame(|L|={len(self.poset.elements)}, ⊤={self.top}, ⊥={self.bottom})"


# ============================================================================
# Locale - Espacio Topológico sin Puntos
# ============================================================================

@dataclass(frozen=True)
class Locale:
    """
    Locale (espacio topológico sin puntos).
    
    Un Locale es el **dual categórico** de un Frame. Representa un espacio
    topológico de forma constructiva, sin necesidad de puntos individuales.
    
    Interpretación:
    - Los elementos del Frame subyacente se interpretan como "abiertos"
    - El supremo (⋁) corresponde a la unión de abiertos
    - El ínfimo finito (⋀) corresponde a la intersección de abiertos
    - Los morfismos de Locales van en dirección opuesta a los de Frames
    
    Ventajas de la topología sin puntos:
    - Razonamiento constructivo (no requiere axioma de elección)
    - Generalización de espacios topológicos
    - Base para la teoría de Sheaves
    - Conexión natural con lógica intuicionista
    
    Attributes:
        frame: Frame subyacente (retículo de abiertos)
        name: Nombre del Locale (opcional, para debugging)
        _interior_cache: Caché de operadores interior
        _closure_cache: Caché de operadores clausura
    
    Examples:
        >>> # Locale del espacio discreto {1, 2, 3}
        >>> frame = FrameBuilder.from_powerset({1, 2, 3})
        >>> locale = Locale(frame, name="Discrete({1,2,3})")
        >>> 
        >>> # Unión de abiertos
        >>> locale.union({frozenset({1}), frozenset({2})})
        frozenset({1, 2})
    
    Notes:
        - Un Locale no necesariamente proviene de un espacio topológico
        - Hay Locales "sin puntos" que no tienen representación espacial
        - Inmutable y thread-safe
    """
    
    frame: Frame
    name: Optional[str] = None
    _interior_cache: Dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    _closure_cache: Dict = field(default_factory=dict, compare=False, hash=False, repr=False)
    
    def __post_init__(self):
        """Inicializa el Locale."""
        logger.debug(f"Locale creado: {self.name or 'unnamed'}")
    
    def opens(self) -> FrozenSet[Hashable]:
        """
        Retorna el conjunto de abiertos.
        
        Los abiertos son los elementos del Frame subyacente.
        
        Returns:
            Conjunto de abiertos
        """
        return self.frame.poset.elements
    
    def is_open(self, element: Hashable) -> bool:
        """
        Verifica si un elemento es un abierto.
        
        Args:
            element: Elemento a verificar
        
        Returns:
            True si es un abierto, False en caso contrario
        """
        return element in self.frame.poset.elements
    
    def union(self, opens_set: Set[Hashable]) -> Hashable:
        """
        Unión de abiertos.
        
        En un Locale, la unión de abiertos es el supremo en el Frame.
        La unión puede ser de un conjunto arbitrario (incluso infinito).
        
        Args:
            opens_set: Conjunto de abiertos
        
        Returns:
            Unión de los abiertos (también un abierto)
        
        Examples:
            >>> locale.union({frozenset({1}), frozenset({2}), frozenset({3})})
            frozenset({1, 2, 3})
        
        Notes:
            - ⋃ ∅ = ∅ (abierto vacío = bottom)
            - ⋃ {U} = U
            - La unión es asociativa y conmutativa
        """
        return self.frame.join(opens_set)
    
    def intersection(self, opens_set: Set[Hashable]) -> Hashable:
        """
        Intersección finita de abiertos.
        
        En un Locale, la intersección de abiertos es el ínfimo en el Frame.
        **Importante:** Solo se garantiza para intersecciones finitas.
        
        Args:
            opens_set: Conjunto **finito** de abiertos
        
        Returns:
            Intersección de los abiertos (también un abierto)
        
        Raises:
            ValueError: Si el conjunto es demasiado grande (límite de seguridad)
        
        Examples:
            >>> locale.intersection({frozenset({1, 2}), frozenset({2, 3})})
            frozenset({2})
        
        Notes:
            - ⋂ ∅ = X (abierto total = top)
            - ⋂ {U} = U
            - La intersección infinita de abiertos NO es necesariamente abierta
        """
        # Límite de seguridad para intersecciones
        if len(opens_set) > 100:
            raise ValueError(
                f"Intersección de {len(opens_set)} abiertos excede el límite de 100. "
                "Las intersecciones infinitas no están garantizadas en Locales."
            )
        
        return self.frame.meet(opens_set)
    
    def interior(self, element: Hashable) -> Hashable:
        """
        Operador interior (◇).
        
        El interior de un elemento es el mayor abierto contenido en él.
        
        Definición:
            int(a) = ⋁ {U ∈ opens | U ≤ a}
        
        Propiedades:
        - int(a) ≤ a
        - int(int(a)) = int(a) (idempotente)
        - int(a ∧ b) = int(a) ∧ int(b)
        
        Args:
            element: Elemento
        
        Returns:
            Interior del elemento
        
        Examples:
            >>> # En espacio discreto, todo elemento es abierto
            >>> locale.interior(frozenset({1, 2}))
            frozenset({1, 2})
        
        Notes:
            - Resultado cacheado
            - Corresponde al operador modal ◇ (posibilidad) en lógica modal S4
        """
        # Buscar en caché
        if element in self._interior_cache:
            return self._interior_cache[element]
        
        # Calcular interior: supremo de abiertos contenidos
        contained_opens = {
            u for u in self.opens()
            if self.frame.poset.is_leq(u, element)
        }
        
        interior = self.frame.join(contained_opens)
        
        # Cachear
        self._interior_cache[element] = interior
        
        return interior
    
    def closure(self, element: Hashable) -> Hashable:
        """
        Operador clausura (□).
        
        La clausura de un elemento es el menor cerrado que lo contiene.
        Un cerrado es el complemento de un abierto.
        
        Definición:
            cl(a) = ¬int(¬a)
        
        Propiedades:
        - a ≤ cl(a)
        - cl(cl(a)) = cl(a) (idempotente)
        - cl(a ∨ b) = cl(a) ∨ cl(b)
        
        Args:
            element: Elemento
        
        Returns:
            Clausura del elemento
        
        Examples:
            >>> # En espacio discreto, cl(A) = A
            >>> locale.closure(frozenset({1, 2}))
            frozenset({1, 2})
        
        Notes:
            - Resultado cacheado
            - Corresponde al operador modal □ (necesidad) en lógica modal S4
        """
        # Buscar en caché
        if element in self._closure_cache:
            return self._closure_cache[element]
        
        # Calcular clausura: ¬int(¬element)
        neg_element = self.frame.heyting_negation(element)
        interior_neg = self.interior(neg_element)
        closure = self.frame.heyting_negation(interior_neg)
        
        # Cachear
        self._closure_cache[element] = closure
        
        return closure
    
    def boundary(self, element: Hashable) -> Hashable:
        """
        Operador frontera (∂).
        
        La frontera de un elemento es la intersección de su clausura con
        la clausura de su complemento.
        
        Definición:
            ∂a = cl(a) ∧ cl(¬a)
        
        Propiedades:
        - ∂a = cl(a) ∧ ¬int(a) (definición alternativa)
        - ∂∂a ≤ ∂a
        - a es abierto y cerrado ⟺ ∂a = ⊥
        
        Args:
            element: Elemento
        
        Returns:
            Frontera del elemento
        
        Examples:
            >>> # En espacio discreto, ∂A = ∅ (todo es abierto y cerrado)
            >>> locale.boundary(frozenset({1, 2}))
            frozenset()
        
        Notes:
            - La frontera caracteriza la "interfaz" entre el elemento y su complemento
        """
        closure_elem = self.closure(element)
        neg_element = self.frame.heyting_negation(element)
        closure_neg = self.closure(neg_element)
        
        return self.frame.meet_binary(closure_elem, closure_neg)
    
    def is_dense(self, element: Hashable) -> bool:
        """
        Verifica si un elemento es denso.
        
        Un elemento a es denso si cl(a) = ⊤.
        
        Args:
            element: Elemento a verificar
        
        Returns:
            True si es denso, False en caso contrario
        
        Examples:
            >>> # En espacio discreto, solo ⊤ es denso
            >>> locale.is_dense(locale.frame.top)
            True
        """
        return self.closure(element) == self.frame.top
    
    def is_nowhere_dense(self, element: Hashable) -> bool:
        """
        Verifica si un elemento es nowhere dense.
        
        Un elemento a es nowhere dense si int(cl(a)) = ⊥.
        
        Args:
            element: Elemento a verificar
        
        Returns:
            True si es nowhere dense, False en caso contrario
        """
        closure = self.closure(element)
        interior_closure = self.interior(closure)
        return interior_closure == self.frame.bottom
    
    def __repr__(self) -> str:
        """Representación en string del Locale."""
        name_str = f" '{self.name}'" if self.name else ""
        return f"Locale{name_str}(|opens|={len(self.opens())})"


# ============================================================================
# FrameBuilder - Constructor de Frames
# ============================================================================

class FrameBuilder:
    """
    Constructor de Frames desde diferentes fuentes.
    
    Proporciona métodos estáticos para construir Frames desde:
    - Conjuntos potencia (powerset)
    - Retículos completos (verificando distributividad)
    - Contextos formales (FCA)
    """
    
    @staticmethod
    def from_powerset(base_set: Set[Hashable]) -> Frame:
        """
        Construye el Frame del conjunto potencia P(S).
        
        El conjunto potencia con inclusión forma un Frame donde:
        - join = unión (∪)
        - meet = intersección (∩)
        - top = S
        - bottom = ∅
        - La ley distributiva infinita se satisface automáticamente
        
        Args:
            base_set: Conjunto base S
        
        Returns:
            Frame P(S)
        
        Examples:
            >>> frame = FrameBuilder.from_powerset({1, 2, 3})
            >>> frame.join({frozenset({1}), frozenset({2})})
            frozenset({1, 2})
        
        Notes:
            - Este es el ejemplo canónico de Frame
            - Tamaño: 2^|S|
            - Útil para modelar topologías discretas
        """
        # Construir retículo completo
        lattice = LatticeBuilder.from_powerset(base_set)
        
        # Convertir a Frame (el powerset siempre satisface distributividad infinita)
        frame = Frame(
            poset=lattice.poset,
            top=lattice.top,
            bottom=lattice.bottom
        )
        
        return frame
    
    @staticmethod
    def from_complete_lattice(lattice: CompleteLattice, verify: bool = True) -> Frame:
        """
        Intenta construir un Frame desde un retículo completo.
        
        Verifica (opcionalmente) que el retículo satisface la ley distributiva infinita.
        
        Args:
            lattice: Retículo completo
            verify: Si True, verifica la ley distributiva (puede ser costoso)
        
        Returns:
            Frame construido desde el retículo
        
        Raises:
            ValueError: Si la ley distributiva no se satisface
        
        Notes:
            - No todo retículo completo es un Frame
            - La verificación completa es intratable para retículos grandes
        """
        frame = Frame(
            poset=lattice.poset,
            top=lattice.top,
            bottom=lattice.bottom
        )
        
        # La verificación se hace en Frame.__post_init__
        
        return frame


# ============================================================================
# LocaleBuilder - Constructor de Locales
# ============================================================================

class LocaleBuilder:
    """
    Constructor de Locales desde diferentes fuentes.
    
    Proporciona métodos estáticos para construir Locales desde:
    - Frames
    - Espacios topológicos (representados como conjuntos de abiertos)
    - Contextos formales (FCA)
    """
    
    @staticmethod
    def from_frame(frame: Frame, name: Optional[str] = None) -> Locale:
        """
        Construye un Locale desde un Frame.
        
        Args:
            frame: Frame subyacente
            name: Nombre del Locale (opcional)
        
        Returns:
            Locale construido
        """
        return Locale(frame=frame, name=name)
    
    @staticmethod
    def discrete_locale(base_set: Set[Hashable], name: Optional[str] = None) -> Locale:
        """
        Construye el Locale discreto sobre un conjunto.
        
        En el espacio discreto, todo subconjunto es abierto.
        
        Args:
            base_set: Conjunto base
            name: Nombre del Locale (opcional)
        
        Returns:
            Locale discreto
        
        Examples:
            >>> locale = LocaleBuilder.discrete_locale({1, 2, 3})
            >>> # Todo subconjunto es abierto
            >>> locale.is_open(frozenset({1}))
            True
        """
        frame = FrameBuilder.from_powerset(base_set)
        
        if name is None:
            name = f"Discrete({base_set})"
        
        return Locale(frame=frame, name=name)
    
    @staticmethod
    def trivial_locale(base_set: Set[Hashable], name: Optional[str] = None) -> Locale:
        """
        Construye el Locale trivial (indiscreto) sobre un conjunto.
        
        En el espacio trivial, solo ∅ y X son abiertos.
        
        Args:
            base_set: Conjunto base
            name: Nombre del Locale (opcional)
        
        Returns:
            Locale trivial
        
        Examples:
            >>> locale = LocaleBuilder.trivial_locale({1, 2, 3})
            >>> # Solo ∅ y {1,2,3} son abiertos
            >>> len(locale.opens())
            2
        """
        # Elementos: solo ∅ y X
        empty = frozenset()
        full = frozenset(base_set)
        elements = frozenset({empty, full})
        
        # Orden: ∅ ≤ X
        leq = frozenset({
            (empty, empty),
            (empty, full),
            (full, full)
        })
        
        # Crear poset
        poset = PartialOrder(elements, leq)
        
        # Crear Frame
        frame = Frame(poset=poset, top=full, bottom=empty)
        
        if name is None:
            name = f"Trivial({base_set})"
        
        return Locale(frame=frame, name=name)

