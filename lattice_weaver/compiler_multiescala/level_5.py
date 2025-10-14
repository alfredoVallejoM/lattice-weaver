"""
Nivel L5: Meta-patrones

Este módulo implementa el quinto nivel de agregación del compilador multiescala,
que detecta patrones de conceptos de dominio de L4, formando meta-patrones.
"""

from typing import Dict, List, Set, FrozenSet, Any, Optional, Tuple
import math
from collections import defaultdict
from dataclasses import dataclass

from .base import AbstractionLevel
from .level_4 import Level4, DomainConcept, DomainConceptSignature


@dataclass(frozen=True)
class MetaPatternSignature:
    """
    Firma de un meta-patrón para identificar meta-patrones similares.
    
    Attributes:
        pattern_type: Tipo de meta-patrón (e.g., 'pipeline', 'star_schema').
        num_concepts: Número de conceptos de dominio que lo forman.
        concept_types: Tupla ordenada de tipos de conceptos (por firma).
        properties: Diccionario de propiedades clave del meta-patrón.
    """
    pattern_type: str
    num_concepts: int
    concept_types: Tuple[str, ...]
    properties: FrozenSet[Tuple[str, Any]]
    
    def __hash__(self):
        return hash((self.pattern_type, self.num_concepts, 
                    self.concept_types, self.properties))


class MetaPattern:
    """
    Representa un meta-patrón de alto nivel.
    
    Un meta-patrón agrupa conceptos de dominio de L4 que, en conjunto,
    forman una estructura recurrente a nivel de dominio.
    
    Attributes:
        meta_pattern_id: Identificador único del meta-patrón.
        signature: Firma del meta-patrón para identificación.
        concepts: Lista de IDs de conceptos de dominio que forman el meta-patrón.
        internal_constraints: Restricciones internas al meta-patrón.
        meta_pattern_properties: Propiedades específicas del meta-patrón.
    """
    
    def __init__(self, meta_pattern_id: int, signature: MetaPatternSignature,
                 concepts: List[int],
                 internal_constraints: List,
                 meta_pattern_properties: Dict[str, Any] = None):
        self.meta_pattern_id = meta_pattern_id
        self.signature = signature
        self.concepts = concepts
        self.internal_constraints = internal_constraints
        self.meta_pattern_properties = meta_pattern_properties or {}
    
    def __repr__(self) -> str:
        return f"MetaPattern(id={self.meta_pattern_id}, type={self.signature.pattern_type}, concepts={len(self.concepts)})"


class Level5(AbstractionLevel):
    """
    Nivel L5: Meta-patrones
    
    Detecta patrones de conceptos de dominio de L4, formando meta-patrones.
    
    Attributes:
        meta_patterns: Lista de meta-patrones detectados.
        isolated_concepts: Conceptos de dominio que no forman meta-patrones.
        inter_meta_pattern_constraints: Restricciones entre meta-patrones.
    """

    def __init__(self, meta_patterns: List[MetaPattern],
                 isolated_concepts: List[DomainConcept],
                 inter_meta_pattern_constraints: List,
                 config: dict = None):
        """
        Inicializa el nivel L5 con meta-patrones.
        
        Args:
            meta_patterns: Lista de meta-patrones detectados.
            isolated_concepts: Conceptos de dominio que no forman meta-patrones.
            inter_meta_pattern_constraints: Restricciones entre meta-patrones.
            config: Configuración opcional para el nivel.
        """
        super().__init__(level=5, config=config)
        self.meta_patterns = meta_patterns
        self.isolated_concepts = isolated_concepts
        self.inter_meta_pattern_constraints = inter_meta_pattern_constraints
        self.data = {
            'meta_patterns': meta_patterns,
            'isolated_concepts': isolated_concepts,
            'inter_meta_pattern_constraints': inter_meta_pattern_constraints
        }
        
        # Construir mapeos
        self.concept_to_meta_pattern = {}
        for meta_pattern in meta_patterns:
            for concept_id in meta_pattern.concepts:
                self.concept_to_meta_pattern[concept_id] = meta_pattern.meta_pattern_id

    def _compute_meta_pattern_signature(self, concepts: List[int],
                                        meta_pattern_properties: Dict[str, Any]) -> MetaPatternSignature:
        """
        Calcula la firma de un meta-patrón.
        
        Args:
            concepts: IDs de conceptos de dominio en el meta-patrón.
            meta_pattern_properties: Propiedades específicas del meta-patrón.
        
        Returns:
            La firma del meta-patrón.
        """
        # Extraer tipos de conceptos (simplificado)
        concept_types = tuple(sorted([str(c) for c in concepts]))
        
        # Convertir propiedades a frozenset para hashability
        frozen_properties = frozenset(sorted(meta_pattern_properties.items()))
        
        # Por ahora, un tipo de meta-patrón genérico
        pattern_type = "generic_meta_pattern"
        
        return MetaPatternSignature(
            pattern_type=pattern_type,
            num_concepts=len(concepts),
            concept_types=concept_types,
            properties=frozen_properties
        )

    def build_from_lower(self, lower_level: Level4):
        """
        Construye la representación de L5 a partir de L4.
        
        Este método detecta meta-patrones en los conceptos de dominio de L4
        y los agrupa para formar la representación de L5.
        
        Args:
            lower_level: El nivel L4 desde el cual construir L5.
        """
        if not isinstance(lower_level, Level4):
            raise TypeError("lower_level must be a Level4 instance")
        
        # Para simplificar, asumimos que cada concepto de dominio de L4 forma un meta-patrón
        # En una implementación real, se usarían heurísticas o reglas de dominio para agruparlos
        meta_patterns = []
        isolated_concepts = []
        meta_pattern_id = 0
        
        for concept in lower_level.concepts:
            # Asumimos que cada concepto es un meta-patrón por ahora
            meta_pattern_properties = {"size": len(concept.structures)}
            signature = self._compute_meta_pattern_signature([concept.concept_id], meta_pattern_properties)
            
            meta_pattern = MetaPattern(
                meta_pattern_id=meta_pattern_id,
                signature=signature,
                concepts=[concept.concept_id],
                internal_constraints=[], # Simplificado
                meta_pattern_properties=meta_pattern_properties
            )
            meta_patterns.append(meta_pattern)
            meta_pattern_id += 1
        
        # Los conceptos de L4 que no se agrupan en meta-patrones se convierten en conceptos aislados en L5
        # Por ahora, como cada concepto de L4 se convierte en un meta-patrón, no hay conceptos aislados de L4
        # que no formen parte de un meta-patrón en L5. Si la lógica de detección de meta-patrones
        # fuera más compleja, esta parte necesitaría ser ajustada.
        isolated_concepts.extend(lower_level.concepts)
        
        # Eliminar los conceptos que ya están en meta_patterns de isolated_concepts
        concepts_in_meta_patterns_ids = set()
        for mp in meta_patterns:
            concepts_in_meta_patterns_ids.update(mp.concepts)
        
        final_isolated_concepts = []
        for concept in isolated_concepts:
            if concept.concept_id not in concepts_in_meta_patterns_ids:
                final_isolated_concepts.append(concept)
        isolated_concepts = final_isolated_concepts
        
        # Las restricciones inter-meta-patrón son las mismas que las inter-concepto de L4
        inter_meta_pattern_constraints = lower_level.inter_concept_constraints
        
        # Actualizar el estado de L5
        self.meta_patterns = meta_patterns
        self.isolated_concepts = isolated_concepts
        self.inter_meta_pattern_constraints = inter_meta_pattern_constraints
        self.data = {
            'meta_patterns': meta_patterns,
            'isolated_concepts': isolated_concepts,
            'inter_meta_pattern_constraints': inter_meta_pattern_constraints
        }
        
        # Almacenar los conceptos originales de L4 para poder refinar correctamente
        self.config['original_concepts'] = lower_level.concepts
        
        # Almacenar referencia al nivel inferior para roundtrip completo
        self.lower_level = lower_level
        
        # Reconstruir mapeo de conceptos a meta-patrones
        self.concept_to_meta_pattern = {}
        for meta_pattern in meta_patterns:
            for concept_id in meta_pattern.concepts:
                self.concept_to_meta_pattern[concept_id] = meta_pattern.meta_pattern_id

    def refine_to_lower(self) -> Level4:
        """
        Refina la representación de L5 a L4.
        
        Este método desagrega los meta-patrones en sus conceptos de dominio
        constituyentes para reconstruir la representación de L4.
        
        Returns:
            Un nuevo Level4 con los conceptos de dominio reconstruidos.
        """
        # Si tenemos una referencia al nivel inferior, devolverla directamente
        if hasattr(self, 'lower_level') and self.lower_level is not None:
            return self.lower_level
        
        # Recolectar todos los conceptos
        all_concepts = []
        
        # Necesitamos acceso a los conceptos originales de L4
        if 'original_concepts' not in self.config:
            raise ValueError("Cannot refine to L4 without original concept information")
        
        original_concepts = self.config['original_concepts']
        concept_dict = {c.concept_id: c for c in original_concepts}
        
        for meta_pattern in self.meta_patterns:
            for concept_id in meta_pattern.concepts:
                if concept_id in concept_dict:
                    all_concepts.append(concept_dict[concept_id])
        
        # Añadir conceptos aislados
        all_concepts.extend(self.isolated_concepts)
        
        # Reconstruir L4
        # Es crucial pasar los isolated_structures originales de L4 para que el roundtrip sea completo
        original_isolated_structures = self.config.get('original_isolated_structures', [])
        
        return Level4(
            concepts=all_concepts,
            isolated_structures=original_isolated_structures,
            inter_concept_constraints=self.inter_meta_pattern_constraints,
            config=self.config
        )

    def renormalize(self, partitioner, k: int) -> 'Level5':
        """
        Aplica la renormalización en el nivel L5.
        
        La renormalización en L5 opera sobre los meta-patrones,
        agrupándolos en super-meta-patrones para reducir aún más la complejidad.
        
        Args:
            partitioner: Estrategia de particionamiento.
            k: Número de particiones deseadas.
        
        Returns:
            Un nuevo Level5 con meta-patrones renormalizados.
        """
        # Delegamos a L4 para renormalizar
        level4 = self.refine_to_lower()
        renormalized_level4 = level4.renormalize(partitioner, k)
        
        # Reconstruir L5 desde el L4 renormalizado
        new_level5 = Level5([], [], [], config=self.config)
        new_level5.build_from_lower(renormalized_level4)
        
        return new_level5

    def validate(self) -> bool:
        """
        Valida la coherencia interna de la representación de L5.
        
        Verifica:
        - Todos los meta-patrones tienen al menos un concepto.
        - Los conceptos no se solapan entre meta-patrones.
        - Los conceptos aislados no están en ningún meta-patrón.
        
        Returns:
            True si la representación es válida, False en caso contrario.
        """
        try:
            # Verificar que todos los meta-patrones tienen al menos un concepto
            for meta_pattern in self.meta_patterns:
                if not meta_pattern.concepts:
                    return False
            
            # Verificar que los conceptos no se solapan entre meta-patrones
            all_concepts_in_meta_patterns = set()
            for meta_pattern in self.meta_patterns:
                for concept_id in meta_pattern.concepts:
                    if concept_id in all_concepts_in_meta_patterns:
                        return False  # Solapamiento detectado
                    all_concepts_in_meta_patterns.add(concept_id)
            
            # Verificar que los conceptos aislados no están en meta-patrones
            isolated_concept_ids = {c.concept_id for c in self.isolated_concepts}
            if all_concepts_in_meta_patterns.intersection(isolated_concept_ids):
                return False
            
            return True
        except Exception:
            return False

    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad de la representación de L5.
        
        La complejidad se define como la suma de las complejidades de los meta-patrones
        (contando cada meta-patrón una vez) más la complejidad de los conceptos aislados.
        
        Returns:
            La complejidad total de L5.
        """
        if not self.meta_patterns and not self.isolated_concepts:
            return 0.0
        
        # Complejidad de los meta-patrones
        meta_pattern_complexity = 0.0
        for meta_pattern in self.meta_patterns:
            sig = meta_pattern.signature
            meta_pattern_complexity += (math.log(sig.num_concepts + 1) + 
                                        1.0)  # Factor de tipo de meta-patrón
        
        # Complejidad de los conceptos aislados
        isolated_concept_complexity = 0.0
        # Para calcular la complejidad de los conceptos aislados, necesitamos su firma
        # Esto es una simplificación, en una implementación real, se calcularía su complejidad real
        for concept in self.isolated_concepts:
            isolated_concept_complexity += (math.log(len(concept.structures) + 1) + 
                                             math.log(len(concept.domain_properties) + 1))
        
        # Complejidad de las interacciones
        inter_complexity = math.log(len(self.inter_meta_pattern_constraints) + 1)
        
        return (meta_pattern_complexity + isolated_concept_complexity + inter_complexity)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre la representación de L5.
        
        Returns:
            Un diccionario con estadísticas de L5.
        """
        total_concepts_in_meta_patterns = sum(len(mp.concepts) for mp in self.meta_patterns)
        total_components = (total_concepts_in_meta_patterns + len(self.isolated_concepts))
        
        # Análisis de tipos de meta-patrones
        meta_pattern_type_counts = defaultdict(int)
        for meta_pattern in self.meta_patterns:
            meta_pattern_type_counts[meta_pattern.signature.pattern_type] += 1
        
        return {
            'level': self.level,
            'num_meta_patterns': len(self.meta_patterns),
            'num_isolated_concepts': len(self.isolated_concepts),
            'total_concepts_in_meta_patterns': total_concepts_in_meta_patterns,
            'total_components': total_components,
            'avg_concepts_per_meta_pattern': total_concepts_in_meta_patterns / len(self.meta_patterns) if self.meta_patterns else 0,
            'meta_pattern_type_distribution': dict(meta_pattern_type_counts),
            'num_inter_meta_pattern_constraints': len(self.inter_meta_pattern_constraints),
            'complexity': self.complexity,
        }

    def __repr__(self) -> str:
        total_concepts = sum(len(mp.concepts) for mp in self.meta_patterns) + len(self.isolated_concepts)
        return f"Level5(meta_patterns={len(self.meta_patterns)}, isolated_concepts={len(self.isolated_concepts)}, total_concepts={total_concepts}, complexity={self.complexity:.2f})"

