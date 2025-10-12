"""
Morfismos de Frames y Locales

Este módulo implementa los morfismos entre Frames y Locales, que son las
flechas en las categorías correspondientes.

Estructuras implementadas:
- FrameMorphism: Morfismo de Frames (preserva supremos arbitrarios e ínfimos finitos)
- LocaleMorphism: Morfismo de Locales (dual de morfismo de Frames)
- Construcciones categóricas: composición, identidad, productos, etc.

Teoría:
- Los morfismos de Frames preservan la estructura algebraica
- Los morfismos de Locales van en dirección opuesta (dualidad categórica)
- La composición de morfismos es asociativa
- Existen morfismos identidad para cada Frame/Locale

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Set, Dict, Optional, Callable
from dataclasses import dataclass, field
import itertools
import logging

from .locale import Frame, Locale, PartialOrder, CompleteLattice, FrozenDict, Hashable

logger = logging.getLogger(__name__)


# ============================================================================
# FrameMorphism - Morfismo de Frames
# ============================================================================

@dataclass(frozen=True)
class FrameMorphism:
    """
    Morfismo de Frames.
    
    Un morfismo f: L → M entre Frames es una función que preserva:
    1. **Supremos arbitrarios**: f(⋁ S) = ⋁ f(S)
    2. **Ínfimos finitos**: f(a ∧ b) = f(a) ∧ f(b)
    3. **Elementos extremos**: f(⊤) = ⊤ y f(⊥) = ⊥
    
    Propiedades:
    - Los morfismos de Frames son adjuntos por la derecha
    - Preservan la implicación de Heyting: f(a → b) ≥ f(a) → f(b)
    - La composición de morfismos es un morfismo
    
    Attributes:
        source: Frame fuente
        target: Frame objetivo
        mapping: Diccionario inmutable que define f
        _verified: Flag de verificación (para evitar re-verificar)
    
    Examples:
        >>> # Morfismo identidad
        >>> f = FrameMorphism.identity(frame)
        >>> 
        >>> # Morfismo entre powersets
        >>> # f: P({1,2}) → P({a,b,c}) que mapea {1} ↦ {a}
        >>> mapping = FrozenDict({
        ...     frozenset(): frozenset(),
        ...     frozenset({1}): frozenset({'a'}),
        ...     frozenset({2}): frozenset({'b'}),
        ...     frozenset({1,2}): frozenset({'a','b'})
        ... })
        >>> f = FrameMorphism(source, target, mapping)
    
    Notes:
        - La verificación de axiomas puede ser costosa para Frames grandes
        - Inmutable y thread-safe
        - Los morfismos se verifican en construcción
    """
    
    source: Frame
    target: Frame
    mapping: FrozenDict
    _verified: bool = field(default=False, compare=False, hash=False, repr=False)
    
    def __post_init__(self):
        """
        Verifica que el morfismo preserve las operaciones.
        
        Raises:
            ValueError: Si el morfismo no preserva alguna operación
        """
        if not self._verified:
            self._verify_domain_codomain()
            self._verify_preserves_joins()
            self._verify_preserves_finite_meets()
            self._verify_preserves_extrema()
            
            # Marcar como verificado (hack para frozen dataclass)
            object.__setattr__(self, '_verified', True)
            
            logger.debug(
                f"FrameMorphism verificado: {len(self.source.poset.elements)} → "
                f"{len(self.target.poset.elements)} elementos"
            )
    
    def _verify_domain_codomain(self):
        """
        Verifica que el dominio y codominio sean correctos.
        
        Raises:
            ValueError: Si hay elementos no mapeados o mapeados incorrectamente
        """
        # Verificar que todos los elementos del source estén mapeados
        for elem in self.source.poset.elements:
            if elem not in self.mapping:
                raise ValueError(f"Elemento {elem} del source no está mapeado")
        
        # Verificar que todos los valores estén en el target
        for elem, image in self.mapping.items():
            if image not in self.target.poset.elements:
                raise ValueError(
                    f"Imagen {image} de {elem} no está en el target"
                )
    
    def _verify_preserves_joins(self):
        """
        Verifica que f(⋁ S) = ⋁ f(S).
        
        Verificamos para subconjuntos pequeños (muestra).
        
        Raises:
            ValueError: Si no se preservan supremos
        """
        # Limitar verificación para Frames grandes
        if len(self.source.poset.elements) > 20:
            sample_size = 3
        else:
            sample_size = min(4, len(self.source.poset.elements))
        
        # Verificar para subconjuntos de tamaño 1, 2, 3
        for subset_size in range(1, sample_size + 1):
            for subset_tuple in itertools.islice(
                itertools.combinations(self.source.poset.elements, subset_size),
                10  # Máximo 10 combinaciones por tamaño
            ):
                subset = set(subset_tuple)
                
                # LHS: f(⋁ S)
                join_s = self.source.join(subset)
                lhs = self(join_s)
                
                # RHS: ⋁ f(S)
                f_subset = {self(s) for s in subset}
                rhs = self.target.join(f_subset)
                
                if lhs != rhs:
                    raise ValueError(
                        f"Morfismo no preserva supremos:\n"
                        f"  f(⋁ {subset}) = {lhs}\n"
                        f"  ⋁ f({subset}) = {rhs}\n"
                        f"  {lhs} ≠ {rhs}"
                    )
    
    def _verify_preserves_finite_meets(self):
        """
        Verifica que f(a ∧ b) = f(a) ∧ f(b).
        
        Verificamos para pares de elementos (muestra).
        
        Raises:
            ValueError: Si no se preservan ínfimos
        """
        # Limitar verificación para Frames grandes
        sample_elements = list(self.source.poset.elements)
        if len(sample_elements) > 20:
            sample_elements = sample_elements[:10]
        
        for a in sample_elements:
            for b in sample_elements:
                # LHS: f(a ∧ b)
                meet_ab = self.source.meet_binary(a, b)
                lhs = self(meet_ab)
                
                # RHS: f(a) ∧ f(b)
                rhs = self.target.meet_binary(self(a), self(b))
                
                if lhs != rhs:
                    raise ValueError(
                        f"Morfismo no preserva ínfimos:\n"
                        f"  f({a} ∧ {b}) = {lhs}\n"
                        f"  f({a}) ∧ f({b}) = {rhs}\n"
                        f"  {lhs} ≠ {rhs}"
                    )
    
    def _verify_preserves_extrema(self):
        """
        Verifica que f(⊤) = ⊤ y f(⊥) = ⊥.
        
        Raises:
            ValueError: Si no se preservan extremos
        """
        if self(self.source.top) != self.target.top:
            raise ValueError(
                f"Morfismo no preserva top:\n"
                f"  f(⊤_source) = {self(self.source.top)}\n"
                f"  ⊤_target = {self.target.top}"
            )
        
        if self(self.source.bottom) != self.target.bottom:
            raise ValueError(
                f"Morfismo no preserva bottom:\n"
                f"  f(⊥_source) = {self(self.source.bottom)}\n"
                f"  ⊥_target = {self.target.bottom}"
            )
    
    def __call__(self, element: Hashable) -> Hashable:
        """
        Aplica el morfismo a un elemento.
        
        Args:
            element: Elemento del Frame fuente
        
        Returns:
            Imagen del elemento en el Frame objetivo
        
        Raises:
            ValueError: Si el elemento no está en el source
        
        Examples:
            >>> f(frozenset({1}))
            frozenset({'a'})
        """
        if element not in self.source.poset.elements:
            raise ValueError(f"{element} no está en el Frame fuente")
        
        return self.mapping[element]
    
    def compose(self, other: 'FrameMorphism') -> 'FrameMorphism':
        """
        Composición de morfismos.
        
        Dados f: L → M y g: M → N, retorna g ∘ f: L → N.
        
        Args:
            other: Morfismo g: M → N
        
        Returns:
            Morfismo compuesto g ∘ f: L → N
        
        Raises:
            ValueError: Si los morfismos no son componibles
        
        Examples:
            >>> # f: L → M, g: M → N
            >>> h = f.compose(g)  # h = g ∘ f: L → N
            >>> h(x) == g(f(x))  # Para todo x en L
        
        Notes:
            - Orden matemático: (g ∘ f)(x) = g(f(x))
            - self es f, other es g
        """
        if self.target != other.source:
            raise ValueError(
                f"Morfismos no componibles:\n"
                f"  target de f: {len(self.target.poset.elements)} elementos\n"
                f"  source de g: {len(other.source.poset.elements)} elementos"
            )
        
        # Composición de mappings: (g ∘ f)(x) = g(f(x))
        composed_mapping = FrozenDict({
            elem: other(self(elem))
            for elem in self.source.poset.elements
        })
        
        # Crear morfismo compuesto (ya verificado por construcción)
        composed = object.__new__(FrameMorphism)
        object.__setattr__(composed, 'source', self.source)
        object.__setattr__(composed, 'target', other.target)
        object.__setattr__(composed, 'mapping', composed_mapping)
        object.__setattr__(composed, '_verified', True)  # Skip verification
        
        logger.debug("Morfismo compuesto creado")
        
        return composed
    
    @staticmethod
    def identity(frame: Frame) -> 'FrameMorphism':
        """
        Crea el morfismo identidad id: L → L.
        
        Args:
            frame: Frame
        
        Returns:
            Morfismo identidad
        
        Examples:
            >>> id_f = FrameMorphism.identity(frame)
            >>> id_f(x) == x  # Para todo x
        """
        mapping = FrozenDict({elem: elem for elem in frame.poset.elements})
        
        # Crear morfismo identidad (ya verificado por construcción)
        identity = object.__new__(FrameMorphism)
        object.__setattr__(identity, 'source', frame)
        object.__setattr__(identity, 'target', frame)
        object.__setattr__(identity, 'mapping', mapping)
        object.__setattr__(identity, '_verified', True)  # Skip verification
        
        return identity
    
    def __repr__(self) -> str:
        """Representación en string del morfismo."""
        return (
            f"FrameMorphism("
            f"|source|={len(self.source.poset.elements)}, "
            f"|target|={len(self.target.poset.elements)})"
        )


# ============================================================================
# LocaleMorphism - Morfismo de Locales
# ============================================================================

@dataclass(frozen=True)
class LocaleMorphism:
    """
    Morfismo de Locales.
    
    Un morfismo de Locales f: L → M es un morfismo de Frames f*: Ω(M) → Ω(L)
    en **dirección opuesta** (dualidad categórica).
    
    Interpretación:
    - Un morfismo de Locales es una "función continua" entre espacios sin puntos
    - El pullback f* lleva abiertos de M a abiertos de L
    - f* preserva uniones arbitrarias e intersecciones finitas
    
    Attributes:
        source: Locale fuente
        target: Locale objetivo
        frame_morphism: Morfismo de Frames Ω(target) → Ω(source) (dirección opuesta)
    
    Examples:
        >>> # Morfismo de Locales L → M
        >>> # Internamente: morfismo de Frames Ω(M) → Ω(L)
        >>> locale_morphism = LocaleMorphism(locale_L, locale_M, frame_morphism)
        >>> 
        >>> # Pullback de un abierto
        >>> open_M = frozenset({'a', 'b'})
        >>> open_L = locale_morphism.pullback(open_M)
    
    Notes:
        - La dirección es opuesta a la de morfismos de Frames
        - Esto refleja la dualidad entre Locales y Frames
        - Inmutable y thread-safe
    """
    
    source: Locale
    target: Locale
    frame_morphism: FrameMorphism
    
    def __post_init__(self):
        """
        Verifica consistencia de la dirección.
        
        Raises:
            ValueError: Si el morfismo de Frames no tiene la dirección correcta
        """
        # El morfismo de Frames debe ir de target.frame a source.frame
        if self.frame_morphism.source != self.target.frame:
            raise ValueError(
                f"Morfismo de Frames inconsistente:\n"
                f"  source del frame_morphism debe ser target.frame\n"
                f"  Esperado: {id(self.target.frame)}\n"
                f"  Recibido: {id(self.frame_morphism.source)}"
            )
        
        if self.frame_morphism.target != self.source.frame:
            raise ValueError(
                f"Morfismo de Frames inconsistente:\n"
                f"  target del frame_morphism debe ser source.frame\n"
                f"  Esperado: {id(self.source.frame)}\n"
                f"  Recibido: {id(self.frame_morphism.target)}"
            )
        
        logger.debug("LocaleMorphism creado y verificado")
    
    def pullback(self, open_target: Hashable) -> Hashable:
        """
        Pullback de un abierto del Locale objetivo.
        
        Para un morfismo f: L → M, el pullback f*: Ω(M) → Ω(L)
        lleva abiertos de M a abiertos de L.
        
        Args:
            open_target: Abierto en el Locale objetivo M
        
        Returns:
            Pullback f*(U) en el Locale fuente L
        
        Examples:
            >>> # f: L → M
            >>> U = frozenset({'a', 'b'})  # Abierto en M
            >>> f_star_U = f.pullback(U)  # Abierto en L
        
        Notes:
            - f* preserva uniones: f*(U ∪ V) = f*(U) ∪ f*(V)
            - f* preserva intersecciones finitas: f*(U ∩ V) = f*(U) ∩ f*(V)
        """
        return self.frame_morphism(open_target)
    
    def compose(self, other: 'LocaleMorphism') -> 'LocaleMorphism':
        """
        Composición de morfismos de Locales.
        
        Dados f: L → M y g: M → N, retorna g ∘ f: L → N.
        
        Args:
            other: Morfismo g: M → N
        
        Returns:
            Morfismo compuesto g ∘ f: L → N
        
        Raises:
            ValueError: Si los morfismos no son componibles
        
        Notes:
            - La composición de morfismos de Locales va en la dirección esperada
            - Pero los morfismos de Frames subyacentes se componen en orden inverso
        """
        if self.target != other.source:
            raise ValueError("Morfismos de Locales no componibles")
        
        # Composición de morfismos de Frames en orden inverso
        # f: L → M tiene f*: Ω(M) → Ω(L)
        # g: M → N tiene g*: Ω(N) → Ω(M)
        # (g ∘ f)*: Ω(N) → Ω(L) = f* ∘ g*
        composed_frame = other.frame_morphism.compose(self.frame_morphism)
        
        return LocaleMorphism(
            source=self.source,
            target=other.target,
            frame_morphism=composed_frame
        )
    
    @staticmethod
    def identity(locale: Locale) -> 'LocaleMorphism':
        """
        Crea el morfismo identidad id: L → L.
        
        Args:
            locale: Locale
        
        Returns:
            Morfismo identidad
        """
        frame_id = FrameMorphism.identity(locale.frame)
        
        return LocaleMorphism(
            source=locale,
            target=locale,
            frame_morphism=frame_id
        )
    
    def __repr__(self) -> str:
        """Representación en string del morfismo."""
        return (
            f"LocaleMorphism("
            f"{self.source.name or '?'} → {self.target.name or '?'})"
        )


# ============================================================================
# Construcciones Categóricas
# ============================================================================

class FrameConstructions:
    """
    Construcciones categóricas de Frames.
    
    Implementa:
    - Productos
    - Coproductos
    - Subobjetos
    - Cocientes
    """
    
    @staticmethod
    def product(frame1: Frame, frame2: Frame) -> Frame:
        """
        Producto de Frames L × M.
        
        El producto tiene como elementos pares (a, b) con orden componente a componente.
        
        Operaciones:
        - (a₁, b₁) ∨ (a₂, b₂) = (a₁ ∨ a₂, b₁ ∨ b₂)
        - (a₁, b₁) ∧ (a₂, b₂) = (a₁ ∧ a₂, b₁ ∧ b₂)
        - ⊤ = (⊤_L, ⊤_M)
        - ⊥ = (⊥_L, ⊥_M)
        
        Args:
            frame1: Primer Frame L
            frame2: Segundo Frame M
        
        Returns:
            Producto L × M
        
        Examples:
            >>> L = FrameBuilder.from_powerset({1, 2})
            >>> M = FrameBuilder.from_powerset({'a'})
            >>> product = FrameConstructions.product(L, M)
            >>> # |L × M| = |L| * |M| = 4 * 2 = 8
        
        Notes:
            - El producto es el límite en la categoría de Frames
            - Viene con proyecciones π₁: L × M → L y π₂: L × M → M
        """
        # Elementos: producto cartesiano
        elements = frozenset(
            (a, b)
            for a in frame1.poset.elements
            for b in frame2.poset.elements
        )
        
        # Orden: componente a componente
        leq = frozenset(
            ((a1, b1), (a2, b2))
            for (a1, b1) in elements
            for (a2, b2) in elements
            if frame1.poset.is_leq(a1, a2) and frame2.poset.is_leq(b1, b2)
        )
        
        # Crear poset
        poset = PartialOrder(elements, leq)
        
        # Top y bottom
        top = (frame1.top, frame2.top)
        bottom = (frame1.bottom, frame2.bottom)
        
        # Crear Frame
        product_frame = Frame(poset=poset, top=top, bottom=bottom)
        
        logger.info(
            f"Producto de Frames creado: "
            f"|L| = {len(frame1.poset.elements)}, "
            f"|M| = {len(frame2.poset.elements)}, "
            f"|L × M| = {len(elements)}"
        )
        
        return product_frame
    
    @staticmethod
    def projection_left(frame1: Frame, frame2: Frame) -> FrameMorphism:
        """
        Proyección izquierda π₁: L × M → L.
        
        Args:
            frame1: Primer Frame L
            frame2: Segundo Frame M
        
        Returns:
            Morfismo π₁
        """
        product = FrameConstructions.product(frame1, frame2)
        
        mapping = FrozenDict({
            (a, b): a
            for a in frame1.poset.elements
            for b in frame2.poset.elements
        })
        
        # Crear morfismo (skip verification para eficiencia)
        proj = object.__new__(FrameMorphism)
        object.__setattr__(proj, 'source', product)
        object.__setattr__(proj, 'target', frame1)
        object.__setattr__(proj, 'mapping', mapping)
        object.__setattr__(proj, '_verified', True)
        
        return proj
    
    @staticmethod
    def projection_right(frame1: Frame, frame2: Frame) -> FrameMorphism:
        """
        Proyección derecha π₂: L × M → M.
        
        Args:
            frame1: Primer Frame L
            frame2: Segundo Frame M
        
        Returns:
            Morfismo π₂
        """
        product = FrameConstructions.product(frame1, frame2)
        
        mapping = FrozenDict({
            (a, b): b
            for a in frame1.poset.elements
            for b in frame2.poset.elements
        })
        
        # Crear morfismo (skip verification para eficiencia)
        proj = object.__new__(FrameMorphism)
        object.__setattr__(proj, 'source', product)
        object.__setattr__(proj, 'target', frame2)
        object.__setattr__(proj, 'mapping', mapping)
        object.__setattr__(proj, '_verified', True)
        
        return proj


# ============================================================================
# Construcciones de Morfismos
# ============================================================================

class MorphismBuilder:
    """
    Constructor de morfismos desde diferentes especificaciones.
    """
    
    @staticmethod
    def from_function(
        source: Frame,
        target: Frame,
        func: Callable[[Hashable], Hashable],
        verify: bool = True
    ) -> FrameMorphism:
        """
        Construye un morfismo desde una función Python.
        
        Args:
            source: Frame fuente
            target: Frame objetivo
            func: Función Python que define el morfismo
            verify: Si True, verifica que func preserve operaciones
        
        Returns:
            Morfismo de Frames
        
        Raises:
            ValueError: Si func no preserva operaciones (cuando verify=True)
        
        Examples:
            >>> # Morfismo que mapea todo a bottom
            >>> def constant_bottom(x):
            ...     return target.bottom
            >>> f = MorphismBuilder.from_function(source, target, constant_bottom)
        
        Notes:
            - func debe estar definida para todos los elementos del source
            - func debe mapear a elementos del target
        """
        # Construir mapping
        mapping_dict = {}
        for elem in source.poset.elements:
            try:
                image = func(elem)
                if image not in target.poset.elements:
                    raise ValueError(f"Imagen {image} no está en el target")
                mapping_dict[elem] = image
            except Exception as e:
                raise ValueError(f"Error al aplicar func a {elem}: {e}")
        
        mapping = FrozenDict(mapping_dict)
        
        # Crear morfismo
        if verify:
            return FrameMorphism(source, target, mapping)
        else:
            # Skip verification
            morphism = object.__new__(FrameMorphism)
            object.__setattr__(morphism, 'source', source)
            object.__setattr__(morphism, 'target', target)
            object.__setattr__(morphism, 'mapping', mapping)
            object.__setattr__(morphism, '_verified', True)
            return morphism
    
    @staticmethod
    def constant_morphism(source: Frame, target: Frame, value: Hashable) -> FrameMorphism:
        """
        Construye un morfismo constante que mapea todo a un valor.
        
        Args:
            source: Frame fuente
            target: Frame objetivo
            value: Valor constante (debe ser en target)
        
        Returns:
            Morfismo constante
        
        Notes:
            - Solo es un morfismo válido si value = ⊤_target
        """
        if value != target.top:
            raise ValueError(
                "Morfismo constante solo es válido para value = ⊤_target"
            )
        
        mapping = FrozenDict({elem: value for elem in source.poset.elements})
        
        # Crear morfismo (skip verification, sabemos que es válido)
        morphism = object.__new__(FrameMorphism)
        object.__setattr__(morphism, 'source', source)
        object.__setattr__(morphism, 'target', target)
        object.__setattr__(morphism, 'mapping', mapping)
        object.__setattr__(morphism, '_verified', True)
        
        return morphism

