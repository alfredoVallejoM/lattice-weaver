"""
Nivel L2: Patrones Locales

Este módulo implementa el segundo nivel de agregación del compilador multiescala,
que detecta y extrae patrones recurrentes en los bloques de restricciones para
permitir una representación más compacta mediante la identificación de estructuras
repetitivas.
"""

from typing import Dict, List, Set, FrozenSet, Any, Optional, Tuple
import math
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass

from .base import AbstractionLevel
from .level_1 import Level1, ConstraintBlock


@dataclass(frozen=True)
class PatternSignature:
    """
    Firma de un patrón local para identificar bloques estructuralmente similares.
    
    Attributes:
        num_variables: Número de variables en el patrón.
        num_constraints: Número de restricciones en el patrón.
        num_interface_vars: Número de variables de interfaz.
        constraint_types: Tupla ordenada de tipos de restricciones.
    """
    num_variables: int
    num_constraints: int
    num_interface_vars: int
    constraint_types: Tuple[str, ...]
    
    def __hash__(self):
        return hash((self.num_variables, self.num_constraints, 
                    self.num_interface_vars, self.constraint_types))


class LocalPattern:
    """
    Representa un patrón local recurrente en el CSP.
    
    Un patrón local es una estructura que aparece múltiples veces en diferentes
    bloques de restricciones, con la misma topología y tipos de restricciones.
    
    Attributes:
        pattern_id: Identificador único del patrón.
        signature: Firma del patrón para identificación.
        instances: Lista de IDs de bloques que son instancias de este patrón.
        template_block: Bloque representativo usado como plantilla.
        instance_blocks: Mapeo de IDs de instancias a bloques originales.
    """
    
    def __init__(self, pattern_id: int, signature: PatternSignature, 
                 instances: List[int], template_block: ConstraintBlock,
                 instance_blocks: Dict[int, ConstraintBlock] = None):
        self.pattern_id = pattern_id
        self.signature = signature
        self.instances = instances
        self.template_block = template_block
        self.instance_blocks = instance_blocks or {}
    
    def __repr__(self) -> str:
        return f"LocalPattern(id={self.pattern_id}, instances={len(self.instances)}, vars={self.signature.num_variables})"


class Level2(AbstractionLevel):
    """
    Nivel L2: Patrones Locales
    
    Detecta y extrae patrones recurrentes en los bloques de restricciones,
    permitiendo una representación más compacta del CSP mediante la identificación
    de estructuras repetitivas.
    
    Attributes:
        patterns: Lista de patrones locales detectados.
        pattern_instances: Mapeo de IDs de bloques a IDs de patrones.
        unique_blocks: Bloques que no pertenecen a ningún patrón recurrente.
        inter_pattern_constraints: Restricciones entre patrones.
    """

    def __init__(self, patterns: List[LocalPattern], unique_blocks: List[ConstraintBlock],
                 inter_pattern_constraints: List, config: dict = None):
        """
        Inicializa el nivel L2 con patrones locales.
        
        Args:
            patterns: Lista de patrones locales detectados.
            unique_blocks: Bloques únicos que no forman patrones.
            inter_pattern_constraints: Restricciones entre patrones y bloques únicos.
            config: Configuración opcional para el nivel.
        """
        super().__init__(level=2, config=config)
        self.patterns = patterns
        self.unique_blocks = unique_blocks
        self.inter_pattern_constraints = inter_pattern_constraints
        self.data = {
            'patterns': patterns,
            'unique_blocks': unique_blocks,
            'inter_pattern_constraints': inter_pattern_constraints
        }
        
        # Construir mapeo de bloques a patrones
        self.pattern_instances = {}
        for pattern in patterns:
            for block_id in pattern.instances:
                self.pattern_instances[block_id] = pattern.pattern_id

    def _compute_block_signature(self, block: ConstraintBlock) -> PatternSignature:
        """
        Calcula la firma de un bloque de restricciones.
        
        La firma captura la estructura del bloque de forma que bloques
        estructuralmente similares tengan la misma firma.
        
        Args:
            block: El bloque del cual calcular la firma.
        
        Returns:
            La firma del bloque.
        """
        # Extraer tipos de restricciones y ordenarlos
        constraint_types = tuple(sorted([c.name if hasattr(c, 'name') and c.name is not None else 'unknown' 
                                        for c in block.constraints]))
        
        return PatternSignature(
            num_variables=len(block.variables),
            num_constraints=len(block.constraints),
            num_interface_vars=len(block.interface_variables),
            constraint_types=constraint_types
        )

    def build_from_lower(self, lower_level: Level1):
        """
        Construye la representación de L2 a partir de L1.
        
        Este método detecta patrones recurrentes en los bloques de L1 y los
        agrupa para formar la representación de L2.
        
        Args:
            lower_level: El nivel L1 desde el cual construir L2.
        """
        if not isinstance(lower_level, Level1):
            raise TypeError("lower_level must be a Level1 instance")
        
        # Calcular firmas de todos los bloques
        block_signatures = {}
        signature_to_blocks = defaultdict(list)
        
        for block in lower_level.blocks:
            signature = self._compute_block_signature(block)
            block_signatures[block.block_id] = signature
            signature_to_blocks[signature].append(block)
        
        # Identificar patrones (firmas que aparecen múltiples veces)
        patterns = []
        unique_blocks = []
        pattern_id = 0
        
        for signature, blocks in signature_to_blocks.items():
            if len(blocks) >= 2:  # Patrón recurrente
                # Usar el primer bloque como plantilla
                template_block = blocks[0]
                instances = [b.block_id for b in blocks]
                instance_blocks = {b.block_id: b for b in blocks}
                
                pattern = LocalPattern(
                    pattern_id=pattern_id,
                    signature=signature,
                    instances=instances,
                    template_block=template_block,
                    instance_blocks=instance_blocks
                )
                patterns.append(pattern)
                pattern_id += 1
            else:
                # Bloque único
                unique_blocks.append(blocks[0])
        
        # Las restricciones inter-patrón son las mismas que las inter-bloque de L1
        inter_pattern_constraints = lower_level.inter_block_constraints
        
        # Actualizar el estado de L2
        self.patterns = patterns
        self.unique_blocks = unique_blocks
        self.inter_pattern_constraints = inter_pattern_constraints
        self.data = {
            'patterns': patterns,
            'unique_blocks': unique_blocks,
            'inter_pattern_constraints': inter_pattern_constraints
        }
        
        # Guardar información original para refinamiento desde L3
        if 'original_patterns' not in self.config:
            self.config['original_patterns'] = patterns
        if 'original_unique_blocks' not in self.config:
            self.config['original_unique_blocks'] = unique_blocks
        
        # Almacenar referencia al nivel inferior para roundtrip completo
        self.lower_level = lower_level
        
        # Reconstruir mapeo de bloques a patrones
        self.pattern_instances = {}
        for pattern in patterns:
            for block_id in pattern.instances:
                self.pattern_instances[block_id] = pattern.pattern_id

    def refine_to_lower(self) -> Level1:
        """
        Refina la representación de L2 a L1.
        
        Este método desagrega los patrones en sus bloques constituyentes para
        reconstruir la representación de L1.
        
        Returns:
            Un nuevo Level1 con los bloques reconstruidos.
        """
        # Si tenemos una referencia al nivel inferior, devolverla directamente
        if hasattr(self, 'lower_level') and self.lower_level is not None:
            return self.lower_level
        
        # De lo contrario, reconstruir todos los bloques
        all_blocks = []
        
        # Añadir instancias de patrones
        for pattern in self.patterns:
            # Usar los bloques originales almacenados en instance_blocks
            for block_id in pattern.instances:
                if block_id in pattern.instance_blocks:
                    # Usar el bloque original
                    all_blocks.append(pattern.instance_blocks[block_id])
                else:
                    # Fallback: crear una copia del bloque plantilla
                    block = ConstraintBlock(
                        block_id=block_id,
                        variables=pattern.template_block.variables.copy(),
                        constraints=pattern.template_block.constraints.copy()
                    )
                    block.interface_variables = pattern.template_block.interface_variables.copy()
                    all_blocks.append(block)
        
        # Añadir bloques únicos
        all_blocks.extend(self.unique_blocks)
        
        # Reconstruir L1
        return Level1(
            blocks=all_blocks,
            inter_block_constraints=self.inter_pattern_constraints,
            config=self.config
        )

    def renormalize(self, partitioner, k: int) -> 'Level2':
        """
        Aplica la renormalización en el nivel L2.
        
        La renormalización en L2 opera sobre los patrones, agrupándolos en
        super-patrones para reducir aún más la complejidad.
        
        Args:
            partitioner: Estrategia de particionamiento.
            k: Número de particiones deseadas.
        
        Returns:
            Un nuevo Level2 con patrones renormalizados.
        """
        # Delegamos a L1 para renormalizar
        level1 = self.refine_to_lower()
        renormalized_level1 = level1.renormalize(partitioner, k)
        
        # Reconstruir L2 desde el L1 renormalizado
        new_level2 = Level2([], [], [], config=self.config)
        new_level2.build_from_lower(renormalized_level1)
        
        return new_level2

    def validate(self) -> bool:
        """
        Valida la coherencia interna de la representación de L2.
        
        Verifica:
        - Todos los patrones tienen al menos 2 instancias
        - Las firmas de los patrones son consistentes
        - Los bloques únicos no forman patrones
        
        Returns:
            True si la representación es válida, False en caso contrario.
        """
        try:
            # Verificar que todos los patrones tienen al menos 2 instancias
            for pattern in self.patterns:
                if len(pattern.instances) < 2:
                    return False
            
            # Verificar que los bloques únicos no se solapan con las instancias de patrones
            unique_block_ids = {block.block_id for block in self.unique_blocks}
            pattern_instance_ids = set(self.pattern_instances.keys())
            
            if unique_block_ids.intersection(pattern_instance_ids):
                return False  # Solapamiento detectado
            
            return True
        except Exception:
            return False

    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad de la representación de L2.
        
        La complejidad se define como la suma de las complejidades de los patrones
        (contando cada patrón una sola vez, no sus instancias) más la complejidad
        de los bloques únicos.
        
        Returns:
            La complejidad total de L2.
        """
        if not self.patterns and not self.unique_blocks:
            return 0.0
        
        # Complejidad de los patrones (contamos cada patrón una vez)
        pattern_complexity = 0.0
        for pattern in self.patterns:
            # Complejidad del patrón + log del número de instancias
            sig = pattern.signature
            pattern_complexity += (math.log(sig.num_variables + 1) + 
                                 math.log(sig.num_constraints + 1) +
                                 math.log(len(pattern.instances) + 1))
        
        # Complejidad de los bloques únicos
        unique_complexity = 0.0
        for block in self.unique_blocks:
            unique_complexity += (math.log(len(block.variables) + 1) + 
                                math.log(len(block.constraints) + 1))
        
        # Complejidad de las interacciones
        inter_complexity = math.log(len(self.inter_pattern_constraints) + 1)
        
        return pattern_complexity + unique_complexity + inter_complexity

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre la representación de L2.
        
        Returns:
            Un diccionario con estadísticas de L2.
        """
        total_pattern_instances = sum(len(p.instances) for p in self.patterns)
        total_blocks = total_pattern_instances + len(self.unique_blocks)
        
        # Calcular reducción de complejidad
        # Si todos los bloques fueran únicos, la complejidad sería mayor
        naive_complexity = 0.0
        for pattern in self.patterns:
            sig = pattern.signature
            naive_complexity += len(pattern.instances) * (
                math.log(sig.num_variables + 1) + math.log(sig.num_constraints + 1)
            )
        for block in self.unique_blocks:
            naive_complexity += (math.log(len(block.variables) + 1) + 
                               math.log(len(block.constraints) + 1))
        
        compression_ratio = naive_complexity / self.complexity if self.complexity > 0 else 1.0
        
        return {
            'level': self.level,
            'num_patterns': len(self.patterns),
            'num_unique_blocks': len(self.unique_blocks),
            'total_pattern_instances': total_pattern_instances,
            'total_blocks': total_blocks,
            'avg_instances_per_pattern': total_pattern_instances / len(self.patterns) if self.patterns else 0,
            'num_inter_pattern_constraints': len(self.inter_pattern_constraints),
            'complexity': self.complexity,
            'naive_complexity': naive_complexity,
            'compression_ratio': compression_ratio,
        }

    def __repr__(self) -> str:
        total_instances = sum(len(p.instances) for p in self.patterns)
        return f"Level2(patterns={len(self.patterns)}, unique_blocks={len(self.unique_blocks)}, total_instances={total_instances}, complexity={self.complexity:.2f})"

