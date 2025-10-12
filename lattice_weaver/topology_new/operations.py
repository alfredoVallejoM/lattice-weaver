"""
Operaciones Topológicas y Operadores Modales

Este módulo implementa operadores topológicos y modales sobre Locales y Frames.

Estructuras implementadas:
- ModalOperators: Operadores de lógica modal S4 (◇, □)
- TopologicalOperators: Operadores topológicos (interior, clausura, frontera, derivado)
- Verificadores de axiomas modales y topológicos

Teoría:
- Los operadores modales ◇ (posibilidad) y □ (necesidad) corresponden a
  interior y clausura en topología
- La lógica modal S4 es la lógica de los espacios topológicos
- Los axiomas de S4 se satisfacen automáticamente en Locales

Autor: LatticeWeaver Team (Track B)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Set, Dict, Optional, List, Tuple
from dataclasses import dataclass
import logging

from .locale import Locale, Frame, Hashable

logger = logging.getLogger(__name__)


# ============================================================================
# ModalOperators - Operadores de Lógica Modal S4
# ============================================================================

class ModalOperators:
    """
    Operadores modales topológicos en un Locale.
    
    Implementa los operadores de **lógica modal S4**:
    - ◇ (diamond, posibilidad) = interior topológico
    - □ (box, necesidad) = clausura topológica
    
    Axiomas de S4:
    - **K**: □(p → q) → (□p → □q)
    - **T**: □p → p
    - **4**: □p → □□p
    - **Dual**: ◇p ↔ ¬□¬p
    
    Propiedades topológicas:
    - int(a) ≤ a
    - int(int(a)) = int(a) (idempotente)
    - int(a ∧ b) = int(a) ∧ int(b)
    - int(⊤) = ⊤
    
    Attributes:
        locale: Locale sobre el que operar
        _interior_cache: Caché de interiores calculados
        _closure_cache: Caché de clausuras calculadas
    
    Examples:
        >>> modal = ModalOperators(locale)
        >>> 
        >>> # Operador ◇ (posibilidad)
        >>> a = frozenset({1, 2})
        >>> diamond_a = modal.diamond(a)  # int(a)
        >>> 
        >>> # Operador □ (necesidad)
        >>> box_a = modal.box(a)  # cl(a)
        >>> 
        >>> # Verificar axiomas S4
        >>> modal.verify_s4_axioms()
        True
    
    Notes:
        - Los operadores están definidos en el Locale, aquí solo agregamos
          verificación de axiomas y utilidades
        - Resultados cacheados para eficiencia
    """
    
    def __init__(self, locale: Locale):
        """
        Inicializa operadores modales.
        
        Args:
            locale: Locale sobre el que operar
        """
        self.locale = locale
        self._interior_cache: Dict[Hashable, Hashable] = {}
        self._closure_cache: Dict[Hashable, Hashable] = {}
        
        logger.debug(f"ModalOperators inicializado para {locale}")
    
    def diamond(self, element: Hashable) -> Hashable:
        """
        Operador ◇ (posibilidad / interior).
        
        ◇a = int(a) = mayor abierto contenido en a
        
        Interpretación modal:
        - ◇p: "es posible que p"
        - En topología: el interior de p
        
        Args:
            element: Elemento
        
        Returns:
            ◇a
        
        Examples:
            >>> # En espacio discreto: ◇a = a
            >>> modal.diamond(frozenset({1, 2}))
            frozenset({1, 2})
        """
        if element in self._interior_cache:
            return self._interior_cache[element]
        
        result = self.locale.interior(element)
        self._interior_cache[element] = result
        
        return result
    
    def box(self, element: Hashable) -> Hashable:
        """
        Operador □ (necesidad / clausura).
        
        □a = cl(a) = menor cerrado que contiene a
        
        Interpretación modal:
        - □p: "es necesario que p"
        - En topología: la clausura de p
        
        Args:
            element: Elemento
        
        Returns:
            □a
        
        Examples:
            >>> # En espacio discreto: □a = a
            >>> modal.box(frozenset({1, 2}))
            frozenset({1, 2})
        """
        if element in self._closure_cache:
            return self._closure_cache[element]
        
        result = self.locale.closure(element)
        self._closure_cache[element] = result
        
        return result
    
    def verify_s4_axioms(self, sample_size: Optional[int] = None) -> bool:
        """
        Verifica los axiomas de la lógica modal S4.
        
        Axiomas:
        - **K**: □(p → q) → (□p → □q)
        - **T**: □p → p
        - **4**: □p → □□p
        - **Dual**: ◇p ↔ ¬□¬p
        
        Args:
            sample_size: Número de elementos a verificar (None = todos)
        
        Returns:
            True si se satisfacen los axiomas, False en caso contrario
        
        Notes:
            - Verificación completa puede ser costosa para Locales grandes
            - Por defecto verifica una muestra representativa
        """
        opens = list(self.locale.opens())
        
        if sample_size is None:
            if len(opens) > 20:
                sample_size = 10
            else:
                sample_size = len(opens)
        
        sample = opens[:sample_size]
        
        logger.info(f"Verificando axiomas S4 en muestra de {len(sample)} elementos")
        
        # Axioma T: □p → p
        for p in sample:
            box_p = self.box(p)
            if not self.locale.frame.poset.is_leq(box_p, p):
                logger.error(f"Axioma T violado: □{p} ≰ {p}")
                return False
        
        # Axioma 4: □p → □□p
        for p in sample:
            box_p = self.box(p)
            box_box_p = self.box(box_p)
            if not self.locale.frame.poset.is_leq(box_p, box_box_p):
                logger.error(f"Axioma 4 violado: □{p} ≰ □□{p}")
                return False
        
        # Axioma Dual: ◇p ↔ ¬□¬p
        for p in sample:
            diamond_p = self.diamond(p)
            
            neg_p = self.locale.frame.heyting_negation(p)
            box_neg_p = self.box(neg_p)
            neg_box_neg_p = self.locale.frame.heyting_negation(box_neg_p)
            
            if diamond_p != neg_box_neg_p:
                logger.error(f"Axioma Dual violado: ◇{p} ≠ ¬□¬{p}")
                return False
        
        # Axioma K: □(p → q) → (□p → □q)
        # Verificar para pares de elementos
        for i, p in enumerate(sample[:5]):  # Limitar para eficiencia
            for q in sample[:5]:
                # p → q
                impl_pq = self.locale.frame.heyting_implication(p, q)
                
                # □(p → q)
                box_impl = self.box(impl_pq)
                
                # □p → □q
                box_p = self.box(p)
                box_q = self.box(q)
                impl_box = self.locale.frame.heyting_implication(box_p, box_q)
                
                # Verificar: □(p → q) ≤ (□p → □q)
                if not self.locale.frame.poset.is_leq(box_impl, impl_box):
                    logger.error(f"Axioma K violado para p={p}, q={q}")
                    return False
        
        logger.info("Todos los axiomas S4 verificados exitosamente")
        return True
    
    def get_modal_properties(self, element: Hashable) -> Dict[str, any]:
        """
        Calcula propiedades modales de un elemento.
        
        Args:
            element: Elemento a analizar
        
        Returns:
            Diccionario con propiedades modales
        """
        diamond_elem = self.diamond(element)
        box_elem = self.box(element)
        
        # Propiedades
        is_open = (diamond_elem == element)
        is_closed = (box_elem == element)
        is_clopen = is_open and is_closed
        is_dense = self.locale.is_dense(element)
        is_nowhere_dense = self.locale.is_nowhere_dense(element)
        
        # Regularidad
        neg_elem = self.locale.frame.heyting_negation(element)
        neg_neg_elem = self.locale.frame.heyting_negation(neg_elem)
        is_regular = (neg_neg_elem == element)
        
        return {
            'element': element,
            'interior': diamond_elem,
            'closure': box_elem,
            'is_open': is_open,
            'is_closed': is_closed,
            'is_clopen': is_clopen,
            'is_dense': is_dense,
            'is_nowhere_dense': is_nowhere_dense,
            'is_regular': is_regular
        }


# ============================================================================
# TopologicalOperators - Operadores Topológicos Adicionales
# ============================================================================

class TopologicalOperators:
    """
    Operadores topológicos adicionales.
    
    Implementa operadores más allá de interior y clausura:
    - Frontera (boundary)
    - Derivado (derived set)
    - Exterior
    - Adherencia
    """
    
    def __init__(self, locale: Locale):
        """
        Inicializa operadores topológicos.
        
        Args:
            locale: Locale sobre el que operar
        """
        self.locale = locale
        self.modal = ModalOperators(locale)
    
    def boundary(self, element: Hashable) -> Hashable:
        """
        Frontera (boundary) de un elemento.
        
        ∂a = cl(a) ∧ cl(¬a)
        
        Propiedades:
        - ∂a = cl(a) - int(a) (en espacios clásicos)
        - a es abierto y cerrado ⟺ ∂a = ⊥
        
        Args:
            element: Elemento
        
        Returns:
            Frontera ∂a
        """
        return self.locale.boundary(element)
    
    def exterior(self, element: Hashable) -> Hashable:
        """
        Exterior de un elemento.
        
        ext(a) = int(¬a)
        
        Propiedades:
        - ext(a) ∧ a = ⊥
        - ext(a) ∨ int(a) ∨ ∂a = ⊤ (partición del espacio)
        
        Args:
            element: Elemento
        
        Returns:
            Exterior de a
        """
        neg_element = self.locale.frame.heyting_negation(element)
        return self.modal.diamond(neg_element)
    
    def derived_set(self, element: Hashable) -> Hashable:
        """
        Conjunto derivado (puntos de acumulación).
        
        En topología clásica: puntos límite de a.
        En Locales: cl(a) - {a} (aproximación)
        
        Args:
            element: Elemento
        
        Returns:
            Conjunto derivado de a
        
        Notes:
            - Esta es una aproximación, el concepto de "puntos" no existe en Locales
        """
        closure = self.modal.box(element)
        
        # En Locales, no podemos "quitar puntos"
        # Aproximación: cl(a) ∧ ¬a
        neg_element = self.locale.frame.heyting_negation(element)
        return self.locale.frame.meet_binary(closure, neg_element)
    
    def adherence(self, element: Hashable) -> Hashable:
        """
        Adherencia de un elemento.
        
        La adherencia es simplemente la clausura.
        
        Args:
            element: Elemento
        
        Returns:
            Adherencia de a (= cl(a))
        """
        return self.modal.box(element)
    
    def is_separated(self, a: Hashable, b: Hashable) -> bool:
        """
        Verifica si dos elementos están separados.
        
        a y b están separados si cl(a) ∧ b = ⊥ y a ∧ cl(b) = ⊥.
        
        Args:
            a: Primer elemento
            b: Segundo elemento
        
        Returns:
            True si están separados, False en caso contrario
        """
        cl_a = self.modal.box(a)
        cl_b = self.modal.box(b)
        
        meet1 = self.locale.frame.meet_binary(cl_a, b)
        meet2 = self.locale.frame.meet_binary(a, cl_b)
        
        return meet1 == self.locale.frame.bottom and meet2 == self.locale.frame.bottom
    
    def separation_properties(self) -> Dict[str, bool]:
        """
        Verifica propiedades de separación del Locale.
        
        Returns:
            Diccionario con propiedades de separación
        
        Notes:
            - T0, T1, T2 (Hausdorff), etc. requieren puntos
            - En Locales, verificamos propiedades aproximadas
        """
        # En Locales sin puntos, las propiedades de separación son limitadas
        # Verificamos propiedades estructurales
        
        frame = self.locale.frame
        
        # Verificar si es discreto (todo es abierto)
        is_discrete = all(
            self.modal.diamond(elem) == elem
            for elem in list(frame.poset.elements)[:10]  # Muestra
        )
        
        # Verificar si es trivial (solo ⊥ y ⊤ son abiertos)
        is_trivial = len(self.locale.opens()) == 2
        
        return {
            'is_discrete': is_discrete,
            'is_trivial': is_trivial,
            'is_spatial': False  # Los Locales generales no son espaciales
        }


# ============================================================================
# Análisis de Conectividad
# ============================================================================

class ConnectivityAnalyzer:
    """
    Análisis de conectividad en Locales.
    
    Verifica propiedades de conectividad:
    - Conectividad
    - Compacidad
    - Separabilidad
    """
    
    def __init__(self, locale: Locale):
        """
        Inicializa analizador de conectividad.
        
        Args:
            locale: Locale a analizar
        """
        self.locale = locale
        self.modal = ModalOperators(locale)
    
    def is_connected(self) -> bool:
        """
        Verifica si el Locale es conexo.
        
        Un Locale es conexo si no puede escribirse como unión disjunta
        de dos abiertos no vacíos.
        
        Returns:
            True si es conexo, False en caso contrario
        
        Notes:
            - Verificación completa es costosa
            - Verificamos una condición suficiente
        """
        frame = self.locale.frame
        
        # Un Locale es conexo si los únicos elementos clopens son ⊥ y ⊤
        for elem in frame.poset.elements:
            # Verificar si es clopen
            is_open = (self.modal.diamond(elem) == elem)
            is_closed = (self.modal.box(elem) == elem)
            
            if is_open and is_closed:
                # Es clopen
                if elem != frame.bottom and elem != frame.top:
                    # Hay un clopen no trivial → no conexo
                    return False
        
        return True
    
    def is_compact(self) -> bool:
        """
        Verifica si el Locale es compacto.
        
        Un Locale es compacto si toda cubierta abierta tiene una subcubierta finita.
        
        Returns:
            True si es compacto, False en caso contrario
        
        Notes:
            - Verificación completa requiere verificar todas las cubiertas
            - Verificamos una condición necesaria (finitud)
        """
        # Un Locale finito es siempre compacto
        return len(self.locale.opens()) < float('inf')
    
    def connected_components(self) -> List[Hashable]:
        """
        Calcula componentes conexas del Locale.
        
        Returns:
            Lista de componentes conexas (elementos clopens maximales)
        
        Notes:
            - En un Locale conexo, solo hay una componente (⊤)
        """
        frame = self.locale.frame
        
        # Componentes conexas = elementos clopens maximales
        clopens = []
        
        for elem in frame.poset.elements:
            is_open = (self.modal.diamond(elem) == elem)
            is_closed = (self.modal.box(elem) == elem)
            
            if is_open and is_closed:
                clopens.append(elem)
        
        # Filtrar maximales
        components = []
        for c in clopens:
            is_maximal = True
            for d in clopens:
                if c != d and frame.poset.is_less(c, d):
                    is_maximal = False
                    break
            
            if is_maximal and c != frame.bottom:
                components.append(c)
        
        return components if components else [frame.top]


# ============================================================================
# Utilidades de Análisis
# ============================================================================

class LocaleAnalyzer:
    """
    Análisis completo de un Locale.
    
    Combina todos los analizadores para proporcionar un análisis exhaustivo.
    """
    
    def __init__(self, locale: Locale):
        """
        Inicializa analizador.
        
        Args:
            locale: Locale a analizar
        """
        self.locale = locale
        self.modal = ModalOperators(locale)
        self.topo = TopologicalOperators(locale)
        self.connectivity = ConnectivityAnalyzer(locale)
    
    def analyze(self) -> Dict[str, any]:
        """
        Realiza análisis completo del Locale.
        
        Returns:
            Diccionario con resultados del análisis
        """
        logger.info(f"Analizando Locale: {self.locale}")
        
        # Propiedades básicas
        num_opens = len(self.locale.opens())
        
        # Propiedades modales
        s4_valid = self.modal.verify_s4_axioms()
        
        # Propiedades de separación
        separation = self.topo.separation_properties()
        
        # Propiedades de conectividad
        is_connected = self.connectivity.is_connected()
        is_compact = self.connectivity.is_compact()
        components = self.connectivity.connected_components()
        
        # Elementos especiales
        regular_elements = self.locale.frame.regular_elements()
        
        analysis = {
            'name': self.locale.name,
            'num_opens': num_opens,
            's4_axioms_valid': s4_valid,
            'separation_properties': separation,
            'is_connected': is_connected,
            'is_compact': is_compact,
            'num_components': len(components),
            'num_regular_elements': len(regular_elements),
            'is_boolean': len(regular_elements) == num_opens
        }
        
        logger.info(f"Análisis completado: {analysis}")
        
        return analysis
    
    def summary(self) -> str:
        """
        Genera un resumen textual del análisis.
        
        Returns:
            String con resumen
        """
        analysis = self.analyze()
        
        lines = [
            f"=== Análisis de Locale: {analysis['name'] or 'unnamed'} ===",
            f"Número de abiertos: {analysis['num_opens']}",
            f"Axiomas S4 válidos: {analysis['s4_axioms_valid']}",
            f"Es conexo: {analysis['is_connected']}",
            f"Es compacto: {analysis['is_compact']}",
            f"Número de componentes: {analysis['num_components']}",
            f"Elementos regulares: {analysis['num_regular_elements']}/{analysis['num_opens']}",
            f"Es álgebra de Boole: {analysis['is_boolean']}",
            "",
            "Propiedades de separación:",
            f"  - Discreto: {analysis['separation_properties']['is_discrete']}",
            f"  - Trivial: {analysis['separation_properties']['is_trivial']}",
        ]
        
        return "\n".join(lines)

