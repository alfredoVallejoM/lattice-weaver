"""
Nivel L1: Bloques de Restricciones

Este módulo implementa el primer nivel de agregación del compilador multiescala,
que agrupa restricciones fuertemente acopladas en "bloques" para reducir la
granularidad del grafo de restricciones.
"""

from typing import Dict, List, Set, FrozenSet, Any, Optional, Tuple
import math
from collections import defaultdict
import networkx as nx

from .base import AbstractionLevel
from .level_0 import Level0
from ..core.csp_problem import CSP, Constraint


class ConstraintBlock:
    """
    Representa un bloque de restricciones fuertemente acopladas.
    
    Un bloque agrupa un conjunto de variables y las restricciones que las conectan,
    formando una unidad cohesiva que puede ser tratada como una sola entidad en
    niveles superiores de abstracción.
    
    Attributes:
        block_id: Identificador único del bloque.
        variables: Conjunto de variables en el bloque.
        constraints: Lista de restricciones que involucran solo variables del bloque.
        interface_variables: Variables que tienen restricciones con variables fuera del bloque.
    """
    
    def __init__(self, block_id: int, variables: Set[str], constraints: List[Constraint]):
        self.block_id = block_id
        self.variables = variables
        self.constraints = constraints
        self.interface_variables = set()  # Se calculará después
    
    def __repr__(self) -> str:
        return f"ConstraintBlock(id={self.block_id}, vars={len(self.variables)}, constraints={len(self.constraints)})"


class Level1(AbstractionLevel):
    """
    Nivel L1: Bloques de Restricciones
    
    Agrupa restricciones fuertemente acopladas en bloques, reduciendo la granularidad
    del grafo de restricciones. Este nivel proporciona una vista más compacta del CSP
    al abstraer grupos de variables y restricciones relacionadas.
    
    Attributes:
        blocks: Lista de bloques de restricciones.
        block_graph: Grafo donde los nodos son bloques y las aristas representan
                     restricciones entre bloques.
        variable_to_block: Mapeo de variables a sus bloques.
        inter_block_constraints: Restricciones que cruzan entre bloques.
    """

    def __init__(self, blocks: List[ConstraintBlock], inter_block_constraints: List[Constraint], config: dict = None):
        """
        Inicializa el nivel L1 con bloques de restricciones.
        
        Args:
            blocks: Lista de bloques de restricciones.
            inter_block_constraints: Restricciones que conectan diferentes bloques.
            config: Configuración opcional para el nivel.
        """
        super().__init__(level=1, config=config)
        self.blocks = blocks
        self.inter_block_constraints = inter_block_constraints
        self.data = {'blocks': blocks, 'inter_block_constraints': inter_block_constraints}
        
        # Construir mapeo de variables a bloques
        self.variable_to_block = {}
        for block in blocks:
            for var in block.variables:
                self.variable_to_block[var] = block.block_id
        
        # Identificar variables de interfaz
        self._identify_interface_variables()
        
        # Construir grafo de bloques
        self.block_graph = self._build_block_graph()

    def _identify_interface_variables(self):
        """
        Identifica las variables de interfaz de cada bloque.
        
        Las variables de interfaz son aquellas que tienen restricciones con
        variables de otros bloques.
        """
        for constraint in self.inter_block_constraints:
            # Agrupar variables por bloque
            blocks_involved = defaultdict(set)
            for var in constraint.scope:
                if var in self.variable_to_block:
                    block_id = self.variable_to_block[var]
                    blocks_involved[block_id].add(var)
            
            # Marcar variables de interfaz
            if len(blocks_involved) > 1:
                for block_id, vars_in_block in blocks_involved.items():
                    block = self.blocks[block_id]
                    block.interface_variables.update(vars_in_block)

    def _build_block_graph(self) -> nx.Graph:
        """
        Construye un grafo de bloques donde los nodos son bloques y las aristas
        representan restricciones entre bloques.
        
        Returns:
            Un grafo NetworkX con los bloques como nodos.
        """
        G = nx.Graph()
        
        # Añadir nodos para cada bloque
        for block in self.blocks:
            G.add_node(block.block_id, block=block, num_vars=len(block.variables))
        
        # Añadir aristas para restricciones inter-bloque
        for constraint in self.inter_block_constraints:
            # Identificar bloques involucrados
            blocks_involved = set()
            for var in constraint.scope:
                if var in self.variable_to_block:
                    blocks_involved.add(self.variable_to_block[var])
            
            # Añadir aristas entre todos los pares de bloques involucrados
            blocks_list = list(blocks_involved)
            for i in range(len(blocks_list)):
                for j in range(i + 1, len(blocks_list)):
                    b1, b2 = blocks_list[i], blocks_list[j]
                    if G.has_edge(b1, b2):
                        G[b1][b2]['weight'] += 1
                        G[b1][b2]['constraints'].append(constraint)
                    else:
                        G.add_edge(b1, b2, weight=1, constraints=[constraint])
        
        return G

    def build_from_lower(self, lower_level: Level0):
        """
        Construye la representación de L1 a partir de L0.
        
        Este método detecta bloques de restricciones en el CSP de L0 y los
        agrupa para formar la representación de L1.
        
        Args:
            lower_level: El nivel L0 desde el cual construir L1.
        """
        if not isinstance(lower_level, Level0):
            raise TypeError("lower_level must be a Level0 instance")
        
        # Detectar bloques usando el método de L0
        block_variables_list = lower_level.detect_constraint_blocks()
        
        # Crear bloques de restricciones
        blocks = []
        for block_id, block_vars in enumerate(block_variables_list):
            # Encontrar restricciones internas al bloque
            internal_constraints = []
            for constraint in lower_level.csp.constraints:
                if constraint.scope.issubset(block_vars):
                    internal_constraints.append(constraint)
            
            block = ConstraintBlock(block_id, block_vars, internal_constraints)
            blocks.append(block)
        
        # Encontrar restricciones inter-bloque
        inter_block_constraints = []
        for constraint in lower_level.csp.constraints:
            # Si la restricción involucra variables de múltiples bloques
            blocks_involved = set()
            for var in constraint.scope:
                for block_id, block_vars in enumerate(block_variables_list):
                    if var in block_vars:
                        blocks_involved.add(block_id)
                        break
            
            if len(blocks_involved) > 1:
                inter_block_constraints.append(constraint)
        
        # Actualizar el estado de L1
        self.blocks = blocks
        self.inter_block_constraints = inter_block_constraints
        self.data = {'blocks': blocks, 'inter_block_constraints': inter_block_constraints}
        
        # Reconstruir estructuras auxiliares
        self.variable_to_block = {}
        for block in blocks:
            for var in block.variables:
                self.variable_to_block[var] = block.block_id
        
        self._identify_interface_variables()
        self.block_graph = self._build_block_graph()

    def refine_to_lower(self) -> Level0:
        """
        Refina la representación de L1 a L0.
        
        Este método desagrega los bloques de restricciones en sus variables y
        restricciones primitivas para reconstruir el CSP de L0.
        
        Returns:
            Un nuevo Level0 con el CSP reconstruido.
        """
        # Recolectar todas las variables
        all_variables = set()
        for block in self.blocks:
            all_variables.update(block.variables)
        
        # Recolectar todos los dominios (necesitamos acceso al CSP original)
        # Por ahora, asumimos que los dominios están disponibles en el config
        if 'original_domains' not in self.config:
            raise ValueError("Cannot refine to L0 without original domain information")
        
        domains = self.config['original_domains']
        
        # Recolectar todas las restricciones
        all_constraints = []
        for block in self.blocks:
            all_constraints.extend(block.constraints)
        all_constraints.extend(self.inter_block_constraints)
        
        # Crear el CSP de L0
        csp = CSP(variables=all_variables, domains=domains, constraints=all_constraints)
        
        return Level0(csp, config=self.config)

    def renormalize(self, partitioner, k: int) -> 'Level1':
        """
        Aplica la renormalización en el nivel L1.
        
        La renormalización en L1 opera sobre los bloques, agrupándolos en
        super-bloques para reducir aún más la complejidad.
        
        Args:
            partitioner: Estrategia de particionamiento.
            k: Número de particiones deseadas.
        
        Returns:
            Un nuevo Level1 con bloques renormalizados.
        """
        # Por ahora, delegamos a L0 para renormalizar
        # En una implementación más sofisticada, podríamos renormalizar directamente sobre bloques
        level0 = self.refine_to_lower()
        renormalized_level0 = level0.renormalize(partitioner, k)
        
        # Reconstruir L1 desde el L0 renormalizado
        new_level1 = Level1([], [], config=self.config)
        new_level1.build_from_lower(renormalized_level0)
        
        return new_level1

    def validate(self) -> bool:
        """
        Valida la coherencia interna de la representación de L1.
        
        Verifica:
        - Todos los bloques tienen variables
        - Las variables no se solapan entre bloques
        - Las restricciones inter-bloque referencian variables de múltiples bloques
        - El grafo de bloques es consistente
        
        Returns:
            True si la representación es válida, False en caso contrario.
        """
        try:
            # Verificar que todos los bloques tienen variables
            for block in self.blocks:
                if len(block.variables) == 0:
                    return False
            
            # Verificar que las variables no se solapan entre bloques
            all_vars = set()
            for block in self.blocks:
                if all_vars.intersection(block.variables):
                    return False  # Solapamiento detectado
                all_vars.update(block.variables)
            
            # Verificar restricciones inter-bloque
            for constraint in self.inter_block_constraints:
                blocks_involved = set()
                for var in constraint.scope:
                    if var in self.variable_to_block:
                        blocks_involved.add(self.variable_to_block[var])
                
                if len(blocks_involved) <= 1:
                    return False  # No es una restricción inter-bloque
            
            return True
        except Exception:
            return False

    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad de la representación de L1.
        
        La complejidad se define como la suma de las complejidades de los bloques
        individuales más la complejidad de las interacciones entre bloques.
        
        Returns:
            La complejidad total de L1.
        """
        if not self.blocks:
            return 0.0
        
        # Complejidad de los bloques individuales
        block_complexity = 0.0
        for block in self.blocks:
            # Estimar complejidad del bloque (número de variables y restricciones)
            block_complexity += math.log(len(block.variables) + 1) + math.log(len(block.constraints) + 1)
        
        # Complejidad de las interacciones inter-bloque
        inter_block_complexity = math.log(len(self.inter_block_constraints) + 1)
        
        return block_complexity + inter_block_complexity

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre la representación de L1.
        
        Returns:
            Un diccionario con estadísticas de L1.
        """
        total_variables = sum(len(block.variables) for block in self.blocks)
        total_internal_constraints = sum(len(block.constraints) for block in self.blocks)
        
        # Manejar grafos vacíos
        graph_diameter = None
        num_connected_components = 0
        
        if len(self.block_graph.nodes()) > 0:
            try:
                if nx.is_connected(self.block_graph):
                    graph_diameter = nx.diameter(self.block_graph)
            except nx.NetworkXError:
                pass
            num_connected_components = nx.number_connected_components(self.block_graph)
        
        return {
            'level': self.level,
            'num_blocks': len(self.blocks),
            'num_variables': total_variables,
            'num_internal_constraints': total_internal_constraints,
            'num_inter_block_constraints': len(self.inter_block_constraints),
            'avg_block_size': total_variables / len(self.blocks) if self.blocks else 0,
            'graph_density': nx.density(self.block_graph) if len(self.block_graph.nodes()) > 0 else 0,
            'graph_diameter': graph_diameter,
            'num_connected_components': num_connected_components,
            'complexity': self.complexity,
        }

    def __repr__(self) -> str:
        return f"Level1(blocks={len(self.blocks)}, inter_block_constraints={len(self.inter_block_constraints)}, complexity={self.complexity:.2f})"

