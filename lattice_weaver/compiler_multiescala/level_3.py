"""
Nivel L3: Estructuras Compuestas

Este módulo implementa el tercer nivel de agregación del compilador multiescala,
que compone patrones de L2 en estructuras de orden superior, permitiendo representar
relaciones jerárquicas y composiciones complejas entre patrones.
"""

from typing import Dict, List, Set, FrozenSet, Any, Optional, Tuple
import math
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass

from .base import AbstractionLevel
from .level_2 import Level2, LocalPattern


@dataclass(frozen=True)
class CompositeSignature:
    """
    Firma de una estructura compuesta para identificar composiciones similares.
    
    Attributes:
        num_patterns: Número de patrones en la composición.
        num_unique_blocks: Número de bloques únicos en la composición.
        pattern_types: Tupla ordenada de tipos de patrones (por firma).
        topology: Descriptor de la topología de conexión.
    """
    num_patterns: int
    num_unique_blocks: int
    pattern_types: Tuple[str, ...]
    topology: str  # 'linear', 'tree', 'dag', 'cyclic'
    
    def __hash__(self):
        return hash((self.num_patterns, self.num_unique_blocks, 
                    self.pattern_types, self.topology))


class CompositeStructure:
    """
    Representa una estructura compuesta de patrones y bloques.
    
    Una estructura compuesta agrupa patrones y bloques únicos que están
    fuertemente conectados y forman una unidad funcional de orden superior.
    
    Attributes:
        structure_id: Identificador único de la estructura.
        signature: Firma de la estructura para identificación.
        patterns: Lista de IDs de patrones que forman parte de la estructura.
        unique_blocks: Lista de IDs de bloques únicos en la estructura.
        internal_constraints: Restricciones internas a la estructura.
        interface_patterns: IDs de patrones que conectan con otras estructuras.
    """
    
    def __init__(self, structure_id: int, signature: CompositeSignature,
                 patterns: List[int], unique_blocks: List[int],
                 internal_constraints: List, interface_patterns: Set[int] = None):
        self.structure_id = structure_id
        self.signature = signature
        self.patterns = patterns
        self.unique_blocks = unique_blocks
        self.internal_constraints = internal_constraints
        self.interface_patterns = interface_patterns or set()
    
    def __repr__(self) -> str:
        return f"CompositeStructure(id={self.structure_id}, patterns={len(self.patterns)}, unique_blocks={len(self.unique_blocks)})"


class Level3(AbstractionLevel):
    """
    Nivel L3: Estructuras Compuestas
    
    Compone patrones de L2 en estructuras de orden superior, permitiendo
    representar relaciones jerárquicas y composiciones complejas entre patrones.
    
    Attributes:
        structures: Lista de estructuras compuestas detectadas.
        isolated_patterns: Patrones que no pertenecen a ninguna estructura.
        isolated_blocks: Bloques únicos que no pertenecen a ninguna estructura.
        inter_structure_constraints: Restricciones entre estructuras.
    """

    def __init__(self, structures: List[CompositeStructure],
                 isolated_patterns: List[LocalPattern],
                 isolated_blocks: List,
                 inter_structure_constraints: List,
                 config: dict = None):
        """
        Inicializa el nivel L3 con estructuras compuestas.
        
        Args:
            structures: Lista de estructuras compuestas detectadas.
            isolated_patterns: Patrones que no forman estructuras.
            isolated_blocks: Bloques únicos que no forman estructuras.
            inter_structure_constraints: Restricciones entre estructuras.
            config: Configuración opcional para el nivel.
        """
        super().__init__(level=3, config=config)
        self.structures = structures
        self.isolated_patterns = isolated_patterns
        self.isolated_blocks = isolated_blocks
        self.inter_structure_constraints = inter_structure_constraints
        self.data = {
            'structures': structures,
            'isolated_patterns': isolated_patterns,
            'isolated_blocks': isolated_blocks,
            'inter_structure_constraints': inter_structure_constraints
        }
        
        # Construir mapeos
        self.pattern_to_structure = {}
        for structure in structures:
            for pattern_id in structure.patterns:
                self.pattern_to_structure[pattern_id] = structure.structure_id

    def _analyze_topology(self, graph: nx.Graph) -> str:
        """
        Analiza la topología de un grafo de conexiones.
        
        Args:
            graph: Grafo de conexiones entre patrones/bloques.
        
        Returns:
            Descriptor de topología: 'linear', 'tree', 'dag', 'cyclic'.
        """
        if len(graph.nodes()) == 0:
            return 'empty'
        
        if len(graph.nodes()) == 1:
            return 'singleton'
        
        # Verificar si es un árbol
        if nx.is_tree(graph):
            # Verificar si es lineal (todos los nodos tienen grado <= 2)
            if all(degree <= 2 for _, degree in graph.degree()):
                return 'linear'
            return 'tree'
        
        # Verificar si es un DAG
        if nx.is_directed_acyclic_graph(graph.to_directed()):
            return 'dag'
        
        # Si tiene ciclos
        return 'cyclic'

    def _compute_structure_signature(self, patterns: List[int], 
                                     unique_blocks: List[int],
                                     graph: nx.Graph) -> CompositeSignature:
        """
        Calcula la firma de una estructura compuesta.
        
        Args:
            patterns: IDs de patrones en la estructura.
            unique_blocks: IDs de bloques únicos en la estructura.
            graph: Grafo de conexiones de la estructura.
        
        Returns:
            La firma de la estructura.
        """
        # Extraer tipos de patrones (simplificado)
        pattern_types = tuple(sorted([str(p) for p in patterns]))
        
        # Analizar topología
        topology = self._analyze_topology(graph)
        
        return CompositeSignature(
            num_patterns=len(patterns),
            num_unique_blocks=len(unique_blocks),
            pattern_types=pattern_types,
            topology=topology
        )

    def build_from_lower(self, lower_level: Level2):
        """
        Construye la representación de L3 a partir de L2.
        
        Este método detecta estructuras compuestas en los patrones y bloques de L2
        y las agrupa para formar la representación de L3.
        
        Args:
            lower_level: El nivel L2 desde el cual construir L3.
        """
        if not isinstance(lower_level, Level2):
            raise TypeError("lower_level must be a Level2 instance")
        
        # Construir grafo de conexiones entre patrones y bloques
        connection_graph = nx.Graph()
        
        # Añadir nodos para patrones
        for pattern in lower_level.patterns:
            connection_graph.add_node(f"pattern_{pattern.pattern_id}", 
                                     type='pattern', 
                                     obj=pattern)
        
        # Añadir nodos para bloques únicos
        for block in lower_level.unique_blocks:
            connection_graph.add_node(f"block_{block.block_id}", 
                                     type='block', 
                                     obj=block)
        
        # Añadir aristas basadas en restricciones inter-patrón
        for constraint in lower_level.inter_pattern_constraints:
            # Determinar qué patrones/bloques están conectados por esta restricción
            nodes_involved = []
            
            for var in constraint.scope:
                # Buscar en qué patrón o bloque está esta variable
                found = False
                
                # Buscar en patrones
                for pattern in lower_level.patterns:
                    for block_id in pattern.instances:
                        if block_id in pattern.instance_blocks:
                            block = pattern.instance_blocks[block_id]
                            if var in block.variables:
                                nodes_involved.append(f"pattern_{pattern.pattern_id}")
                                found = True
                                break
                    if found:
                        break
                
                # Buscar en bloques únicos
                if not found:
                    for block in lower_level.unique_blocks:
                        if var in block.variables:
                            nodes_involved.append(f"block_{block.block_id}")
                            break
            
            # Añadir arista si conecta dos nodos diferentes
            if len(nodes_involved) >= 2:
                for i in range(len(nodes_involved) - 1):
                    connection_graph.add_edge(nodes_involved[i], nodes_involved[i + 1])
        
        # Detectar componentes conexas como estructuras compuestas
        structures = []
        isolated_patterns = []
        isolated_blocks = []
        structure_id = 0
        
        for component in nx.connected_components(connection_graph):
            if len(component) >= 2:  # Estructura compuesta
                # Extraer patrones y bloques de la componente
                patterns_in_structure = []
                blocks_in_structure = []
                
                for node in component:
                    node_data = connection_graph.nodes[node]
                    if node_data['type'] == 'pattern':
                        patterns_in_structure.append(node_data['obj'].pattern_id)
                    elif node_data['type'] == 'block':
                        blocks_in_structure.append(node_data['obj'].block_id)
                
                # Crear subgrafo para analizar topología
                subgraph = connection_graph.subgraph(component)
                
                # Calcular firma
                signature = self._compute_structure_signature(
                    patterns_in_structure, 
                    blocks_in_structure,
                    subgraph
                )
                
                # Identificar patrones de interfaz (conectados a otras estructuras)
                interface_patterns = set()
                # Por simplicidad, asumimos que todos los patrones pueden ser de interfaz
                
                # Crear estructura
                structure = CompositeStructure(
                    structure_id=structure_id,
                    signature=signature,
                    patterns=patterns_in_structure,
                    unique_blocks=blocks_in_structure,
                    internal_constraints=[],  # Simplificado
                    interface_patterns=interface_patterns
                )
                structures.append(structure)
                structure_id += 1
            else:
                # Nodo aislado
                node = list(component)[0]
                node_data = connection_graph.nodes[node]
                if node_data['type'] == 'pattern':
                    isolated_patterns.append(node_data['obj'])
                elif node_data['type'] == 'block':
                    isolated_blocks.append(node_data['obj'])
        
        # Las restricciones inter-estructura son las mismas que las inter-patrón de L2
        inter_structure_constraints = lower_level.inter_pattern_constraints
        
        # Actualizar el estado de L3
        self.structures = structures
        self.isolated_patterns = isolated_patterns
        self.isolated_blocks = isolated_blocks
        self.inter_structure_constraints = inter_structure_constraints
        self.data = {
            'structures': structures,
            'isolated_patterns': isolated_patterns,
            'isolated_blocks': isolated_blocks,
            'inter_structure_constraints': inter_structure_constraints
        }
        
        # Reconstruir mapeo de patrones a estructuras
        self.pattern_to_structure = {}
        for structure in structures:
            for pattern_id in structure.patterns:
                self.pattern_to_structure[pattern_id] = structure.structure_id
        
        # Guardar información original para refinamiento desde L4
        if 'original_structures' not in self.config:
            self.config['original_structures'] = structures
        if 'original_isolated_patterns' not in self.config:
            self.config['original_isolated_patterns'] = isolated_patterns
        if 'original_isolated_blocks' not in self.config:
            self.config['original_isolated_blocks'] = isolated_blocks

    def refine_to_lower(self) -> Level2:
        """
        Refina la representación de L3 a L2.
        
        Este método desagrega las estructuras compuestas en sus patrones y bloques
        constituyentes para reconstruir la representación de L2.
        
        Returns:
            Un nuevo Level2 con los patrones y bloques reconstruidos.
        """
        # Recolectar todos los patrones
        all_patterns = []
        
        # Añadir patrones de estructuras
        # Necesitamos acceso a los patrones originales de L2
        # Por ahora, asumimos que están disponibles en el config
        if 'original_patterns' not in self.config:
            raise ValueError("Cannot refine to L2 without original pattern information")
        
        original_patterns = self.config['original_patterns']
        pattern_dict = {p.pattern_id: p for p in original_patterns}
        
        for structure in self.structures:
            for pattern_id in structure.patterns:
                if pattern_id in pattern_dict:
                    all_patterns.append(pattern_dict[pattern_id])
        
        # Añadir patrones aislados
        all_patterns.extend(self.isolated_patterns)
        
        # Recolectar todos los bloques únicos
        all_unique_blocks = []
        
        # Añadir bloques de estructuras
        if 'original_unique_blocks' not in self.config:
            raise ValueError("Cannot refine to L2 without original unique block information")
        
        original_unique_blocks = self.config['original_unique_blocks']
        block_dict = {b.block_id: b for b in original_unique_blocks}
        
        for structure in self.structures:
            for block_id in structure.unique_blocks:
                if block_id in block_dict:
                    all_unique_blocks.append(block_dict[block_id])
        
        # Añadir bloques aislados
        all_unique_blocks.extend(self.isolated_blocks)
        
        # Reconstruir L2
        return Level2(
            patterns=all_patterns,
            unique_blocks=all_unique_blocks,
            inter_pattern_constraints=self.inter_structure_constraints,
            config=self.config
        )

    def renormalize(self, partitioner, k: int) -> 'Level3':
        """
        Aplica la renormalización en el nivel L3.
        
        La renormalización en L3 opera sobre las estructuras compuestas,
        agrupándolas en super-estructuras para reducir aún más la complejidad.
        
        Args:
            partitioner: Estrategia de particionamiento.
            k: Número de particiones deseadas.
        
        Returns:
            Un nuevo Level3 con estructuras renormalizadas.
        """
        # Delegamos a L2 para renormalizar
        level2 = self.refine_to_lower()
        renormalized_level2 = level2.renormalize(partitioner, k)
        
        # Reconstruir L3 desde el L2 renormalizado
        new_level3 = Level3([], [], [], [], config=self.config)
        new_level3.build_from_lower(renormalized_level2)
        
        return new_level3

    def validate(self) -> bool:
        """
        Valida la coherencia interna de la representación de L3.
        
        Verifica:
        - Todas las estructuras tienen al menos 2 componentes (patrones o bloques)
        - Los patrones y bloques no se solapan entre estructuras
        - Los patrones aislados no están en ninguna estructura
        
        Returns:
            True si la representación es válida, False en caso contrario.
        """
        try:
            # Verificar que todas las estructuras tienen al menos 2 componentes
            for structure in self.structures:
                if len(structure.patterns) + len(structure.unique_blocks) < 2:
                    return False
            
            # Verificar que los patrones no se solapan entre estructuras
            all_patterns_in_structures = set()
            for structure in self.structures:
                for pattern_id in structure.patterns:
                    if pattern_id in all_patterns_in_structures:
                        return False  # Solapamiento detectado
                    all_patterns_in_structures.add(pattern_id)
            
            # Verificar que los patrones aislados no están en estructuras
            isolated_pattern_ids = {p.pattern_id for p in self.isolated_patterns}
            if all_patterns_in_structures.intersection(isolated_pattern_ids):
                return False
            
            return True
        except Exception:
            return False

    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad de la representación de L3.
        
        La complejidad se define como la suma de las complejidades de las estructuras
        (contando cada estructura una vez) más la complejidad de los patrones y
        bloques aislados.
        
        Returns:
            La complejidad total de L3.
        """
        if not self.structures and not self.isolated_patterns and not self.isolated_blocks:
            return 0.0
        
        # Complejidad de las estructuras
        structure_complexity = 0.0
        for structure in self.structures:
            sig = structure.signature
            structure_complexity += (math.log(sig.num_patterns + 1) + 
                                   math.log(sig.num_unique_blocks + 1) +
                                   1.0)  # Factor de topología
        
        # Complejidad de los patrones aislados
        isolated_pattern_complexity = 0.0
        for pattern in self.isolated_patterns:
            sig = pattern.signature
            isolated_pattern_complexity += (math.log(sig.num_variables + 1) + 
                                          math.log(sig.num_constraints + 1))
        
        # Complejidad de los bloques aislados
        isolated_block_complexity = 0.0
        for block in self.isolated_blocks:
            isolated_block_complexity += (math.log(len(block.variables) + 1) + 
                                        math.log(len(block.constraints) + 1))
        
        # Complejidad de las interacciones
        inter_complexity = math.log(len(self.inter_structure_constraints) + 1)
        
        return (structure_complexity + isolated_pattern_complexity + 
                isolated_block_complexity + inter_complexity)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre la representación de L3.
        
        Returns:
            Un diccionario con estadísticas de L3.
        """
        total_patterns_in_structures = sum(len(s.patterns) for s in self.structures)
        total_blocks_in_structures = sum(len(s.unique_blocks) for s in self.structures)
        total_components = (total_patterns_in_structures + total_blocks_in_structures + 
                          len(self.isolated_patterns) + len(self.isolated_blocks))
        
        # Análisis de topologías
        topology_counts = defaultdict(int)
        for structure in self.structures:
            topology_counts[structure.signature.topology] += 1
        
        return {
            'level': self.level,
            'num_structures': len(self.structures),
            'num_isolated_patterns': len(self.isolated_patterns),
            'num_isolated_blocks': len(self.isolated_blocks),
            'total_patterns_in_structures': total_patterns_in_structures,
            'total_blocks_in_structures': total_blocks_in_structures,
            'total_components': total_components,
            'avg_components_per_structure': (total_patterns_in_structures + total_blocks_in_structures) / len(self.structures) if self.structures else 0,
            'topology_distribution': dict(topology_counts),
            'num_inter_structure_constraints': len(self.inter_structure_constraints),
            'complexity': self.complexity,
        }

    def __repr__(self) -> str:
        total_patterns = sum(len(s.patterns) for s in self.structures) + len(self.isolated_patterns)
        return f"Level3(structures={len(self.structures)}, isolated_patterns={len(self.isolated_patterns)}, isolated_blocks={len(self.isolated_blocks)}, total_patterns={total_patterns}, complexity={self.complexity:.2f})"

