"""
LatticeBuilder: Construcción de retículos de conceptos mediante FCA.

Este módulo implementa Formal Concept Analysis (FCA) para extraer la teoría
lógica de un problema CSP, construyendo el retículo de conceptos formales.

Autor: Manus AI
Fecha: 11 de Octubre de 2025
"""

from typing import Set, Tuple, List, Dict, FrozenSet, Optional
from .context import FormalContext
import itertools


class LatticeBuilder:
    """
    Constructor de retículos de conceptos mediante FCA.
    
    Implementa el algoritmo Close-by-One (CbO) para construir eficientemente
    el retículo de conceptos formales de un contexto.
    
    Attributes:
        context: Contexto formal del problema
        concepts: Lista de conceptos formales (extent, intent)
        lattice_graph: Grafo del retículo (relaciones de orden)
    """
    
    def __init__(self, context: FormalContext):
        """
        Inicializa el constructor de retículos.
        
        Args:
            context: Contexto formal a analizar
        """
        self.context = context
        self.concepts: List[Tuple[FrozenSet, FrozenSet]] = []
        self.lattice_graph = {}
    
    def build_lattice(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Construye el retículo de conceptos usando el algoritmo CbO.
        
        Returns:
            Lista de conceptos formales
        """
        self.concepts.clear()
        
        # Iniciar con el concepto top
        top_extent = frozenset()
        top_intent = frozenset(self.context.prime_objects(set()))
        
        self._cbo(set(), top_intent, -1)
        
        return self.concepts
    
    def _cbo(self, extent: Set, intent: FrozenSet, y: int):
        """
        Algoritmo Close-by-One recursivo.
        
        Args:
            extent: Extensión actual
            intent: Intensión actual
            y: Índice del último atributo añadido
        """
        # Añadir concepto
        extent_frozen = frozenset(extent)
        self.concepts.append((extent_frozen, intent))
        
        # Obtener lista de objetos ordenada
        objects_list = sorted(self.context.objects)
        
        # Explorar añadiendo cada objeto
        for i, obj in enumerate(objects_list):
            if i <= y or obj in extent:
                continue
            
            # Calcular nuevo concepto
            new_extent = extent | {obj}
            new_intent = frozenset(self.context.prime_objects(new_extent))
            
            # Verificar canonicidad (no hemos visto este intent antes)
            is_canonical = True
            for j in range(i):
                if objects_list[j] in self.context.prime_attributes(new_intent):
                    is_canonical = False
                    break
            
            if is_canonical:
                self._cbo(new_extent, new_intent, i)
    
    def build_lattice_with_library(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Construye el retículo usando la librería `concepts`.
        
        Returns:
            Lista de conceptos formales
        """
        try:
            from concepts import Context as ConceptsContext
            
            # Convertir a formato de concepts
            context_str = self.context.to_concepts_format()
            concepts_context = ConceptsContext.fromstring(context_str)
            
            # Construir retículo
            lattice = concepts_context.lattice
            
            # Convertir a nuestro formato
            self.concepts = [
                (frozenset(c.extent), frozenset(c.intent))
                for c in lattice
            ]
            
            return self.concepts
        
        except ImportError:
            print("Librería 'concepts' no disponible. Usando implementación propia.")
            return self.build_lattice()
    
    def get_implications(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Extrae las implicaciones lógicas del retículo.
        
        Una implicación A → B significa que todo objeto que tiene todos
        los atributos en A también tiene todos los atributos en B.
        
        Returns:
            Lista de implicaciones (premisa, conclusión)
        """
        if not self.concepts:
            self.build_lattice()
        
        implications = []
        
        # Para cada par de conceptos
        for i, (extent1, intent1) in enumerate(self.concepts):
            for j, (extent2, intent2) in enumerate(self.concepts):
                if i == j:
                    continue
                
                # Si extent1 ⊆ extent2, entonces intent2 ⊆ intent1
                # Lo que significa: intent1 → intent2
                if extent1.issubset(extent2) and not intent1.issubset(intent2):
                    # Implicación no trivial
                    premise = intent1 - intent2
                    conclusion = intent2 - intent1
                    
                    if premise and conclusion:
                        implications.append((premise, conclusion))
        
        return implications
    
    def find_maximal_concepts(self) -> List[Tuple[FrozenSet, FrozenSet]]:
        """
        Encuentra los conceptos maximales (más específicos).
        
        Returns:
            Lista de conceptos maximales
        """
        if not self.concepts:
            self.build_lattice()
        
        maximal = []
        
        for i, (extent1, intent1) in enumerate(self.concepts):
            is_maximal = True
            
            for j, (extent2, intent2) in enumerate(self.concepts):
                if i == j:
                    continue
                
                # Si extent1 ⊂ extent2 (subconjunto propio), no es maximal
                if extent1 < extent2:
                    is_maximal = False
                    break
            
            if is_maximal:
                maximal.append((extent1, intent1))
        
        return maximal
    
    def get_concept_by_extent(self, extent: Set) -> Optional[Tuple[FrozenSet, FrozenSet]]:
        """
        Busca un concepto por su extensión.
        
        Args:
            extent: Extensión a buscar
            
        Returns:
            Concepto correspondiente o None
        """
        extent_frozen = frozenset(extent)
        
        for concept_extent, concept_intent in self.concepts:
            if concept_extent == extent_frozen:
                return (concept_extent, concept_intent)
        
        return None
    
    def get_concept_by_intent(self, intent: Set) -> Optional[Tuple[FrozenSet, FrozenSet]]:
        """
        Busca un concepto por su intensión.
        
        Args:
            intent: Intensión a buscar
            
        Returns:
            Concepto correspondiente o None
        """
        intent_frozen = frozenset(intent)
        
        for concept_extent, concept_intent in self.concepts:
            if concept_intent == intent_frozen:
                return (concept_extent, concept_intent)
        
        return None
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del retículo.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.concepts:
            return {'num_concepts': 0}
        
        # Tamaños de extensiones e intensiones
        extent_sizes = [len(extent) for extent, _ in self.concepts]
        intent_sizes = [len(intent) for _, intent in self.concepts]
        
        return {
            'num_concepts': len(self.concepts),
            'avg_extent_size': round(sum(extent_sizes) / len(extent_sizes), 2) if extent_sizes else 0,
            'avg_intent_size': round(sum(intent_sizes) / len(intent_sizes), 2) if intent_sizes else 0,
            'max_extent_size': max(extent_sizes) if extent_sizes else 0,
            'max_intent_size': max(intent_sizes) if intent_sizes else 0,
            'num_maximal_concepts': len(self.find_maximal_concepts())
        }
    
    def export_lattice(self, filepath: str):
        """
        Exporta el retículo a un archivo JSON.
        
        Args:
            filepath: Ruta del archivo
        """
        import json
        from pathlib import Path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'concepts': [
                {
                    'extent': list(extent),
                    'intent': list(intent)
                }
                for extent, intent in self.concepts
            ],
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def from_arc_engine(cls, arc_engine) -> 'LatticeBuilder':
        """
        Crea un LatticeBuilder desde un ArcEngine.
        
        Args:
            arc_engine: Instancia de ArcEngine
            
        Returns:
            LatticeBuilder con el contexto construido
        """
        from .context import FormalContext
        
        context = FormalContext.from_arc_engine(arc_engine)
        return cls(context)

