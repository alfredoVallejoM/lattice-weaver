"""
Nivel L4: Abstracciones de Dominio

Este módulo implementa el cuarto nivel de agregación del compilador multiescala,
que mapea estructuras compuestas de L3 a conceptos de dominio de alto nivel.
"""

from typing import Dict, List, Set, FrozenSet, Any, Optional, Tuple
import math
from collections import defaultdict
from dataclasses import dataclass

from .base import AbstractionLevel
from .level_3 import Level3, CompositeStructure
from .level_2 import LocalPattern
from .level_1 import ConstraintBlock


@dataclass(frozen=True)
class DomainConceptSignature:
    """
    Firma de un concepto de dominio para identificar abstracciones similares.
    
    Attributes:
        concept_type: Tipo de concepto (e.g., 'scheduling_task', 'resource_allocation').
        num_structures: Número de estructuras compuestas que lo forman.
        structure_types: Tupla ordenada de tipos de estructuras (por firma).
        properties: Diccionario de propiedades clave del dominio.
    """
    concept_type: str
    num_structures: int
    structure_types: Tuple[str, ...]
    properties: FrozenSet[Tuple[str, Any]]
    
    def __hash__(self):
        return hash((self.concept_type, self.num_structures, 
                    self.structure_types, self.properties))


class DomainConcept:
    """
    Representa un concepto de dominio de alto nivel.
    
    Un concepto de dominio agrupa estructuras compuestas de L3 que, en conjunto,
    forman una abstracción semántica significativa en el dominio del problema.
    
    Attributes:
        concept_id: Identificador único del concepto.
        signature: Firma del concepto para identificación.
        structures: Lista de IDs de estructuras compuestas que forman el concepto.
        internal_constraints: Restricciones internas al concepto.
        domain_properties: Propiedades específicas del dominio.
    """
    
    def __init__(self, concept_id: int, signature: DomainConceptSignature,
                 structures: List[int],
                 internal_constraints: List,
                 domain_properties: Dict[str, Any] = None):
        self.concept_id = concept_id
        self.signature = signature
        self.structures = structures
        self.internal_constraints = internal_constraints
        self.domain_properties = domain_properties or {}
    
    def __repr__(self) -> str:
        return f"DomainConcept(id={self.concept_id}, type={self.signature.concept_type}, structures={len(self.structures)})"


class Level4(AbstractionLevel):
    """
    Nivel L4: Abstracciones de Dominio
    
    Mapea estructuras compuestas de L3 a conceptos de dominio de alto nivel.
    
    Attributes:
        concepts: Lista de conceptos de dominio detectados.
        isolated_structures: Estructuras compuestas que no forman conceptos.
        inter_concept_constraints: Restricciones entre conceptos.
    """

    def __init__(self, concepts: List[DomainConcept],
                 isolated_structures: List[CompositeStructure],
                 inter_concept_constraints: List,
                 config: dict = None):
        """
        Inicializa el nivel L4 con conceptos de dominio.
        
        Args:
            concepts: Lista de conceptos de dominio detectados.
            isolated_structures: Estructuras compuestas que no forman conceptos.
            inter_concept_constraints: Restricciones entre conceptos.
            config: Configuración opcional para el nivel.
        """
        super().__init__(level=4, config=config)
        self.concepts = concepts
        self.isolated_structures = isolated_structures
        self.inter_concept_constraints = inter_concept_constraints
        self.data = {
            'concepts': concepts,
            'isolated_structures': isolated_structures,
            'inter_concept_constraints': inter_concept_constraints
        }
        
        # Construir mapeos
        self.structure_to_concept = {}
        for concept in concepts:
            for structure_id in concept.structures:
                self.structure_to_concept[structure_id] = concept.concept_id

    def _compute_concept_signature(self, structures: List[int],
                                   domain_properties: Dict[str, Any]) -> DomainConceptSignature:
        """
        Calcula la firma de un concepto de dominio.
        
        Args:
            structures: IDs de estructuras compuestas en el concepto.
            domain_properties: Propiedades específicas del dominio.
        
        Returns:
            La firma del concepto.
        """
        # Extraer tipos de estructuras (simplificado)
        structure_types = tuple(sorted([str(s) for s in structures]))
        
        # Convertir propiedades a frozenset para hashability
        frozen_properties = frozenset(sorted(domain_properties.items()))
        
        # Por ahora, un tipo de concepto genérico
        concept_type = "generic_domain_concept"
        
        return DomainConceptSignature(
            concept_type=concept_type,
            num_structures=len(structures),
            structure_types=structure_types,
            properties=frozen_properties
        )

    def build_from_lower(self, lower_level: Level3):
        """
        Construye la representación de L4 a partir de L3.
        
        Este método detecta conceptos de dominio en las estructuras compuestas de L3
        y las agrupa para formar la representación de L4.
        
        Args:
            lower_level: El nivel L3 desde el cual construir L4.
        """
        if not isinstance(lower_level, Level3):
            raise TypeError("lower_level must be a Level3 instance")
        
        # Para simplificar, asumimos que cada estructura compuesta de L3 forma un concepto de dominio
        # En una implementación real, se usarían heurísticas o reglas de dominio para agruparlas
        concepts = []
        isolated_structures = []
        concept_id = 0
        
        for structure in lower_level.structures:
            # Asumimos que cada estructura es un concepto por ahora
            domain_properties = {"size": len(structure.patterns) + len(structure.unique_blocks)}
            signature = self._compute_concept_signature([structure.structure_id], domain_properties)
            
            concept = DomainConcept(
                concept_id=concept_id,
                signature=signature,
                structures=[structure.structure_id],
                internal_constraints=[], # Simplificado
                domain_properties=domain_properties
            )
            concepts.append(concept)
            concept_id += 1
        
        # Las estructuras aisladas de L3 (que pueden ser patrones o bloques aislados) se convierten en estructuras aisladas en L4
        # Para la simplificación actual, tratamos los patrones y bloques aislados de L3 como estructuras aisladas en L4.
        # En una implementación más sofisticada, se podría intentar agruparlos en conceptos de dominio.
        isolated_structures.extend(lower_level.isolated_patterns)
        isolated_structures.extend(lower_level.isolated_blocks)
        
        # Las restricciones inter-concepto son las mismas que las inter-estructura de L3
        inter_concept_constraints = lower_level.inter_structure_constraints
        
        # Actualizar el estado de L4
        self.concepts = concepts
        self.isolated_structures = isolated_structures
        self.inter_concept_constraints = inter_concept_constraints
        self.data = {
            'concepts': concepts,
            'isolated_structures': isolated_structures,
            'inter_concept_constraints': inter_concept_constraints
        }
        
        # Reconstruir mapeo de estructuras a conceptos
        self.structure_to_concept = {}
        for concept in concepts:
            for structure_id in concept.structures:
                self.structure_to_concept[structure_id] = concept.concept_id
        
        # Guardar información original para refinamiento desde L5
        if 'original_concepts' not in self.config:
            self.config['original_concepts'] = concepts
        if 'original_isolated_structures' not in self.config:
            self.config['original_isolated_structures'] = isolated_structures

    def refine_to_lower(self) -> Level3:
        """
        Refina la representación de L4 a L3.
        
        Este método desagrega los conceptos de dominio en sus estructuras compuestas
        constituyentes para reconstruir la representación de L3.
        
        Returns:
            Un nuevo Level3 con las estructuras compuestas reconstruidas.
        """
        # Recolectar todas las estructuras
        all_structures = []
        
        # Necesitamos acceso a las estructuras originales de L3
        if 'original_structures' not in self.config:
            raise ValueError("Cannot refine to L3 without original structure information")
        
        original_structures = self.config['original_structures']
        structure_dict = {s.structure_id: s for s in original_structures}
        
        for concept in self.concepts:
            for structure_id in concept.structures:
                if structure_id in structure_dict:
                    all_structures.append(structure_dict[structure_id])
        
        # Añadir estructuras aisladas
        # Necesitamos separar los patrones aislados y los bloques aislados de L4
        # para reconstruir correctamente Level3.
        reconstructed_isolated_patterns = []
        reconstructed_isolated_blocks = []
        for item in self.isolated_structures:
            if isinstance(item, LocalPattern):
                reconstructed_isolated_patterns.append(item)
            elif isinstance(item, ConstraintBlock):
                reconstructed_isolated_blocks.append(item)
            elif isinstance(item, CompositeStructure):
                all_structures.append(item) # Si una estructura compuesta de L3 fue aislada en L4, la devolvemos como estructura
            else:
                # Manejar otros tipos si es necesario, o lanzar un error
                pass
        
        # Reconstruir L3
        return Level3(
            structures=all_structures,
            isolated_patterns=reconstructed_isolated_patterns,
            isolated_blocks=reconstructed_isolated_blocks,
            inter_structure_constraints=self.inter_concept_constraints,
            config=self.config
        )

    def renormalize(self, partitioner, k: int) -> 'Level4':
        """
        Aplica la renormalización en el nivel L4.
        
        La renormalización en L4 opera sobre los conceptos de dominio,
        agrupándolos en super-conceptos para reducir aún más la complejidad.
        
        Args:
            partitioner: Estrategia de particionamiento.
            k: Número de particiones deseadas.
        
        Returns:
            Un nuevo Level4 con conceptos renormalizados.
        """
        # Delegamos a L3 para renormalizar
        level3 = self.refine_to_lower()
        renormalized_level3 = level3.renormalize(partitioner, k)
        
        # Reconstruir L4 desde el L3 renormalizado
        new_level4 = Level4([], [], [], config=self.config)
        new_level4.build_from_lower(renormalized_level3)
        
        return new_level4

    def validate(self) -> bool:
        """
        Valida la coherencia interna de la representación de L4.
        
        Verifica:
        - Todos los conceptos tienen al menos una estructura.
        - Las estructuras no se solapan entre conceptos.
        - Las estructuras aisladas no están en ningún concepto.
        
        Returns:
            True si la representación es válida, False en caso contrario.
        """
        try:
            # Verificar que todos los conceptos tienen al menos una estructura
            for concept in self.concepts:
                if not concept.structures:
                    return False
            
            # Verificar que las estructuras no se solapan entre conceptos
            all_structures_in_concepts = set()
            for concept in self.concepts:
                for structure_id in concept.structures:
                    if structure_id in all_structures_in_concepts:
                        return False  # Solapamiento detectado
                    all_structures_in_concepts.add(structure_id)
            
            # Verificar que las estructuras aisladas no están en conceptos
            isolated_structure_ids = set()
            for item in self.isolated_structures:
                if isinstance(item, CompositeStructure):
                    isolated_structure_ids.add(item.structure_id)
                elif isinstance(item, LocalPattern):
                    isolated_structure_ids.add(item.pattern_id)
                elif isinstance(item, ConstraintBlock):
                    isolated_structure_ids.add(item.block_id)
            
            if all_structures_in_concepts.intersection(isolated_structure_ids):
                return False
            
            return True
        except Exception:
            return False

    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad de la representación de L4.
        
        La complejidad se define como la suma de las complejidades de los conceptos
        (contando cada concepto una vez) más la complejidad de las estructuras aisladas.
        
        Returns:
            La complejidad total de L4.
        """
        if not self.concepts and not self.isolated_structures:
            return 0.0
        
        # Complejidad de los conceptos
        concept_complexity = 0.0
        for concept in self.concepts:
            sig = concept.signature
            concept_complexity += (math.log(sig.num_structures + 1) + 
                                   1.0)  # Factor de tipo de concepto
        
        # Complejidad de las estructuras aisladas
        isolated_structure_complexity = 0.0
        # Para calcular la complejidad de las estructuras aisladas, necesitamos su firma
        # Esto es una simplificación, en una implementación real, se calcularía su complejidad real
        for structure in self.isolated_structures:
            isolated_structure_complexity += (math.log(len(structure.patterns) + 1) + 
                                              math.log(len(structure.unique_blocks) + 1))
        
        # Complejidad de las interacciones
        inter_complexity = math.log(len(self.inter_concept_constraints) + 1)
        
        return (concept_complexity + isolated_structure_complexity + inter_complexity)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre la representación de L4.
        
        Returns:
            Un diccionario con estadísticas de L4.
        """
        total_structures_in_concepts = sum(len(c.structures) for c in self.concepts)
        total_components = (total_structures_in_concepts + len(self.isolated_structures))
        
        # Análisis de tipos de conceptos
        concept_type_counts = defaultdict(int)
        for concept in self.concepts:
            concept_type_counts[concept.signature.concept_type] += 1
        
        return {
            'level': self.level,
            'num_concepts': len(self.concepts),
            'num_isolated_structures': len(self.isolated_structures),
            'total_structures_in_concepts': total_structures_in_concepts,
            'total_components': total_components,
            'avg_structures_per_concept': total_structures_in_concepts / len(self.concepts) if self.concepts else 0,
            'concept_type_distribution': dict(concept_type_counts),
            'num_inter_concept_constraints': len(self.inter_concept_constraints),
            'complexity': self.complexity,
        }

    def __repr__(self) -> str:
        total_structures = sum(len(c.structures) for c in self.concepts) + len(self.isolated_structures)
        return f"Level4(concepts={len(self.concepts)}, isolated_structures={len(self.isolated_structures)}, total_structures={total_structures}, complexity={self.complexity:.2f})"

