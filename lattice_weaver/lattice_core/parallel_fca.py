# lattice_weaver/lattice_core/parallel_fca.py

"""
FCA Paralelo

Implementa Análisis Formal de Conceptos paralelizado usando multiprocessing
para eludir el GIL de Python y acelerar el cálculo de conceptos formales.

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
Versión: 1.0
"""

from multiprocessing import Pool, cpu_count
from typing import Set, FrozenSet, List, Tuple, Dict, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class ParallelFCABuilder:
    """
    Constructor de retículos FCA paralelizado.
    
    Divide el espacio de búsqueda de conceptos entre múltiples procesos
    para acelerar el cálculo en problemas grandes.
    
    Attributes:
        num_workers: Número de procesos paralelos
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Inicializa el constructor paralelo.
        
        Args:
            num_workers: Número de procesos (default: CPU count)
        """
        self.num_workers = num_workers or cpu_count()
        logger.info(f"ParallelFCABuilder inicializado con {self.num_workers} workers")
    
    def build_lattice_parallel(self, context) -> Set[Tuple[FrozenSet, FrozenSet]]:
        """
        Construye el retículo de conceptos en paralelo.
        
        Estrategia:
        1. Convertir contexto a formato serializable
        2. Dividir objetos en chunks
        3. Cada proceso calcula conceptos para su chunk
        4. Combinar y calcular cierre
        
        Args:
            context: FormalContext con objetos, atributos e incidencias
        
        Returns:
            Conjunto de conceptos formales (extent, intent)
        """
        # Convertir contexto a formato serializable
        serializable_context = self._make_serializable_context(context)
        
        objects = list(context.objects)
        
        if len(objects) == 0:
            logger.warning("Contexto vacío, retornando conjunto vacío")
            return set()
        
        # Dividir objetos en chunks
        chunk_size = max(1, len(objects) // self.num_workers)
        chunks = [
            objects[i:i+chunk_size] 
            for i in range(0, len(objects), chunk_size)
        ]
        
        logger.info(f"Dividiendo {len(objects)} objetos en {len(chunks)} chunks")
        
        # Procesar chunks en paralelo
        with Pool(processes=self.num_workers) as pool:
            results = pool.starmap(
                _compute_concepts_for_chunk,
                [(chunk, serializable_context) for chunk in chunks]
            )
        
        # Combinar resultados
        all_concepts = set()
        for concepts in results:
            all_concepts.update(concepts)
        
        logger.info(f"Conceptos parciales encontrados: {len(all_concepts)}")
        
        # Calcular cierre (conceptos derivados de combinaciones)
        closed_concepts = self._compute_closure(all_concepts, serializable_context)
        
        logger.info(f"Conceptos totales después del cierre: {len(closed_concepts)}")
        
        return closed_concepts
    
    def _make_serializable_context(self, context) -> Dict:
        """
        Convierte FormalContext a formato serializable.
        
        Args:
            context: FormalContext original
        
        Returns:
            Diccionario serializable
        """
        return {
            'objects': frozenset(context.objects),
            'attributes': frozenset(context.attributes),
            'incidence': frozenset(context.incidence),
            # Precalcular mapeos para eficiencia
            'obj_to_attrs': {
                obj: frozenset(context.get_object_attributes(obj))
                for obj in context.objects
            },
            'attr_to_objs': {
                attr: frozenset(context.get_attribute_objects(attr))
                for attr in context.attributes
            }
        }
    
    def _compute_closure(self, concepts: Set[Tuple[FrozenSet, FrozenSet]], 
                        context: Dict) -> Set[Tuple[FrozenSet, FrozenSet]]:
        """
        Calcula el cierre del conjunto de conceptos.
        
        Genera conceptos adicionales mediante intersecciones de extents.
        
        Args:
            concepts: Conjunto inicial de conceptos
            context: Contexto serializable
        
        Returns:
            Conjunto cerrado de conceptos
        """
        closed = set(concepts)
        queue = list(concepts)
        
        while queue:
            c1_extent, c1_intent = queue.pop(0)
            
            for c2_extent, c2_intent in list(closed):
                # Calcular meet (intersección de extents)
                meet_extent = c1_extent.intersection(c2_extent)
                
                # Calcular intent correspondiente
                meet_intent = self._compute_intent(meet_extent, context)
                
                # Verificar si es un nuevo concepto
                new_concept = (meet_extent, meet_intent)
                if new_concept not in closed:
                    # Verificar que es un concepto formal válido
                    if self._is_formal_concept(meet_extent, meet_intent, context):
                        closed.add(new_concept)
                        queue.append(new_concept)
        
        return closed
    
    def _compute_intent(self, extent: FrozenSet, context: Dict) -> FrozenSet:
        """
        Calcula el intent (atributos comunes) de un extent.
        
        Args:
            extent: Conjunto de objetos
            context: Contexto serializable
        
        Returns:
            Conjunto de atributos comunes
        """
        if not extent:
            return frozenset(context['attributes'])
        
        # Intersección de atributos de todos los objetos
        common = None
        for obj in extent:
            obj_attrs = context['obj_to_attrs'].get(obj, frozenset())
            if common is None:
                common = obj_attrs
            else:
                common = common.intersection(obj_attrs)
        
        return common or frozenset()
    
    def _compute_extent(self, intent: FrozenSet, context: Dict) -> FrozenSet:
        """
        Calcula el extent (objetos) de un intent.
        
        Args:
            intent: Conjunto de atributos
            context: Contexto serializable
        
        Returns:
            Conjunto de objetos que tienen todos los atributos
        """
        if not intent:
            return frozenset(context['objects'])
        
        # Intersección de objetos que tienen todos los atributos
        common = None
        for attr in intent:
            attr_objs = context['attr_to_objs'].get(attr, frozenset())
            if common is None:
                common = attr_objs
            else:
                common = common.intersection(attr_objs)
        
        return common or frozenset()
    
    def _is_formal_concept(self, extent: FrozenSet, intent: FrozenSet, 
                          context: Dict) -> bool:
        """
        Verifica si (extent, intent) es un concepto formal.
        
        Un concepto formal cumple: extent' = intent y intent' = extent
        
        Args:
            extent: Conjunto de objetos
            intent: Conjunto de atributos
            context: Contexto serializable
        
        Returns:
            True si es un concepto formal
        """
        computed_intent = self._compute_intent(extent, context)
        computed_extent = self._compute_extent(intent, context)
        
        return computed_intent == intent and computed_extent == extent


def _compute_concepts_for_chunk(objects_chunk: List, context: Dict) -> Set[Tuple[FrozenSet, FrozenSet]]:
    """
    Función auxiliar para procesar un chunk de objetos.
    
    Esta función se ejecuta en un proceso separado.
    
    Args:
        objects_chunk: Subconjunto de objetos a procesar
        context: Contexto serializable
    
    Returns:
        Conjunto de conceptos encontrados
    """
    concepts = set()
    
    # Concepto trivial: conjunto vacío
    empty_extent = frozenset()
    empty_intent = frozenset(context['attributes'])
    concepts.add((empty_extent, empty_intent))
    
    # Concepto trivial: todos los objetos
    full_extent = frozenset(context['objects'])
    full_intent = frozenset()
    
    # Calcular intent de todos los objetos
    common_attrs = None
    for obj in full_extent:
        obj_attrs = context['obj_to_attrs'].get(obj, frozenset())
        if common_attrs is None:
            common_attrs = obj_attrs
        else:
            common_attrs = common_attrs.intersection(obj_attrs)
    
    if common_attrs:
        full_intent = common_attrs
    
    concepts.add((full_extent, full_intent))
    
    # Generar conceptos para subconjuntos del chunk
    # Limitamos a subconjuntos de tamaño razonable para evitar explosión combinatoria
    max_subset_size = min(len(objects_chunk), 10)
    
    for r in range(1, max_subset_size + 1):
        for obj_subset in combinations(objects_chunk, r):
            obj_set = frozenset(obj_subset)
            
            # Calcular intent (atributos comunes)
            intent = _compute_intent_helper(obj_set, context)
            
            # Calcular extent (objetos con esos atributos)
            extent = _compute_extent_helper(intent, context)
            
            # Verificar si es un concepto formal (cierre)
            # Un concepto es formal si extent' = intent
            recomputed_intent = _compute_intent_helper(extent, context)
            
            if recomputed_intent == intent:
                concepts.add((extent, intent))
    
    return concepts


def _compute_intent_helper(extent: FrozenSet, context: Dict) -> FrozenSet:
    """Helper para calcular intent en proceso separado."""
    if not extent:
        return frozenset(context['attributes'])
    
    common = None
    for obj in extent:
        obj_attrs = context['obj_to_attrs'].get(obj, frozenset())
        if common is None:
            common = obj_attrs
        else:
            common = common.intersection(obj_attrs)
    
    return common or frozenset()


def _compute_extent_helper(intent: FrozenSet, context: Dict) -> FrozenSet:
    """Helper para calcular extent en proceso separado."""
    if not intent:
        return frozenset(context['objects'])
    
    common = None
    for attr in intent:
        attr_objs = context['attr_to_objs'].get(attr, frozenset())
        if common is None:
            common = attr_objs
        else:
            common = common.intersection(attr_objs)
    
    return common or frozenset()

