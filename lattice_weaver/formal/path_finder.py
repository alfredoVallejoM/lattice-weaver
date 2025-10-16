"""
PathFinder: Búsqueda de Caminos entre Soluciones CSP

Este módulo implementa la búsqueda de caminos (paths) entre soluciones de CSP
en el espacio de tipos cúbicos, permitiendo:

1. Encontrar caminos entre dos soluciones
2. Verificar equivalencia de soluciones
3. Analizar la estructura topológica del espacio de soluciones
4. Construir pruebas formales de equivalencia

Los caminos representan transformaciones continuas entre soluciones,
fundamentales para el análisis homotópico de problemas CSP.

Autor: LatticeWeaver Team (Track: CSP-Cubical Integration)
Fecha: 12 de Octubre de 2025
Versión: 1.0
"""

from typing import Dict, List, Set, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import logging

from .cubical_csp_type import CubicalCSPType
from .cubical_syntax import Term, PathType, PathAbs, Var
from .csp_cubical_bridge import CSPToCubicalBridge

logger = logging.getLogger(__name__)


# ============================================================================
# Representación de Caminos
# ============================================================================

@dataclass(frozen=True)
class SolutionPath:
    """
    Camino entre dos soluciones CSP.
    
    Representa una secuencia de transformaciones que conectan dos soluciones,
    junto con la prueba formal de que el camino es válido.
    
    Attributes:
        start: Solución inicial
        end: Solución final
        steps: Pasos intermedios del camino
        distance: Distancia de Hamming entre soluciones
        proof_term: Término cúbico que prueba la equivalencia (opcional)
    """
    start: Dict[str, Any]
    end: Dict[str, Any]
    steps: Tuple[Dict[str, Any], ...]
    distance: int
    proof_term: Optional[Term] = None
    
    def __len__(self) -> int:
        """Longitud del camino (número de pasos)."""
        return len(self.steps)
    
    def is_direct(self) -> bool:
        """Verifica si el camino es directo (sin pasos intermedios)."""
        return len(self.steps) == 0
    
    def __str__(self) -> str:
        """Representación en string del camino."""
        if self.is_direct():
            return f"Path({self.start} → {self.end}, direct)"
        return f"Path({self.start} → ... ({len(self.steps)} steps) → {self.end})"


# ============================================================================
# PathFinder - Motor de Búsqueda de Caminos
# ============================================================================

@dataclass
class PathFinder:
    """
    Motor de búsqueda de caminos entre soluciones CSP.
    
    Utiliza búsqueda en anchura (BFS) para encontrar caminos mínimos
    entre soluciones en el espacio de configuraciones.
    
    Attributes:
        bridge: Bridge CSP-Cubical
        max_search_depth: Profundidad máxima de búsqueda
        
    Examples:
        >>> finder = PathFinder(bridge)
        >>> path = finder.find_path(solution1, solution2)
        >>> if path:
        ...     print(f"Camino encontrado: {path}")
    """
    
    bridge: CSPToCubicalBridge
    max_search_depth: int = 10
    
    # Caché de caminos encontrados
    _path_cache: Dict[Tuple[frozenset, frozenset], Optional[SolutionPath]] = field(
        default_factory=dict, repr=False
    )
    
    def find_path(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        use_cache: bool = True
    ) -> Optional[SolutionPath]:
        """
        Encuentra un camino entre dos soluciones.
        
        Usa BFS para encontrar el camino más corto que conecta las dos
        soluciones a través de soluciones intermedias válidas.
        
        Args:
            start: Solución inicial
            end: Solución final
            use_cache: Si True, usa caché de caminos
            
        Returns:
            Camino encontrado, o None si no existe
            
        Examples:
            >>> path = finder.find_path({'X': 1, 'Y': 2}, {'X': 2, 'Y': 3})
        """
        # Verificar que ambas soluciones sean válidas
        if not self.bridge.verify_solution(start):
            logger.warning(f"Solución inicial inválida: {start}")
            return None
        
        if not self.bridge.verify_solution(end):
            logger.warning(f"Solución final inválida: {end}")
            return None
        
        # Verificar caché
        if use_cache:
            cache_key = (frozenset(start.items()), frozenset(end.items()))
            if cache_key in self._path_cache:
                logger.debug(f"Usando camino cacheado")
                return self._path_cache[cache_key]
        
        # Caso especial: soluciones idénticas
        if start == end:
            path = SolutionPath(
                start=start,
                end=end,
                steps=tuple(),
                distance=0
            )
            if use_cache:
                self._path_cache[cache_key] = path
            return path
        
        # BFS para encontrar camino
        path = self._bfs_search(start, end)
        
        # Cachear resultado
        if use_cache:
            cache_key = (frozenset(start.items()), frozenset(end.items()))
            self._path_cache[cache_key] = path
        
        return path
    
    def _bfs_search(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any]
    ) -> Optional[SolutionPath]:
        """
        Búsqueda en anchura (BFS) para encontrar camino.
        
        Args:
            start: Solución inicial
            end: Solución final
            
        Returns:
            Camino encontrado, o None
        """
        # Cola de búsqueda: (solución_actual, camino_hasta_ahora)
        queue = deque([(start, [])])
        visited = {self._solution_hash(start)}
        
        while queue:
            current, path = queue.popleft()
            
            # Verificar profundidad máxima
            if len(path) >= self.max_search_depth:
                continue
            
            # Generar vecinos (soluciones a distancia 1)
            for neighbor in self._get_neighbors(current):
                # Verificar si es válida
                if not self.bridge.verify_solution(neighbor):
                    continue
                
                # Verificar si ya visitada
                neighbor_hash = self._solution_hash(neighbor)
                if neighbor_hash in visited:
                    continue
                
                visited.add(neighbor_hash)
                new_path = path + [neighbor]
                
                # Verificar si llegamos al destino
                if neighbor == end:
                    return SolutionPath(
                        start=start,
                        end=end,
                        steps=tuple(path),  # Pasos intermedios (sin incluir start y end)
                        distance=len(new_path)
                    )
                
                # Añadir a la cola
                queue.append((neighbor, new_path))
        
        # No se encontró camino
        logger.debug(f"No se encontró camino entre {start} y {end}")
        return None
    
    def _get_neighbors(self, solution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera soluciones vecinas (a distancia de Hamming 1).
        
        Args:
            solution: Solución actual
            
        Returns:
            Lista de soluciones vecinas
        """
        neighbors = []
        cubical_type = self.bridge.cubical_type
        
        if cubical_type is None:
            return neighbors
        
        # Para cada variable, probar cambiar su valor
        for var in solution.keys():
            if var not in cubical_type.domain_types:
                continue
            
            # Acceder a los valores del dominio directamente desde el objeto FiniteType
            domain_values = cubical_type.domain_types[var].values
            current_value = solution[var]
            
            # Probar cada valor del dominio
            for new_value in domain_values:
                if new_value == current_value:
                    continue
                
                # Crear solución vecina
                neighbor = solution.copy()
                neighbor[var] = new_value
                neighbors.append(neighbor)
        
        return neighbors
    
    def _solution_hash(self, solution: Dict[str, Any]) -> int:
        """
        Calcula hash de una solución.
        
        Args:
            solution: Solución a hashear
            
        Returns:
            Hash de la solución
        """
        return hash(frozenset(solution.items()))
    
    def hamming_distance(
        self,
        solution1: Dict[str, Any],
        solution2: Dict[str, Any]
    ) -> int:
        """
        Calcula la distancia de Hamming entre dos soluciones.
        
        La distancia de Hamming es el número de variables que tienen
        valores diferentes.
        
        Args:
            solution1: Primera solución
            solution2: Segunda solución
            
        Returns:
            Distancia de Hamming
            
        Examples:
            >>> dist = finder.hamming_distance({'X': 1, 'Y': 2}, {'X': 1, 'Y': 3})
            >>> print(dist)
            1
        """
        if set(solution1.keys()) != set(solution2.keys()):
            raise ValueError("Las soluciones deben tener las mismas variables")
        
        distance = 0
        for var in solution1.keys():
            if solution1[var] != solution2[var]:
                distance += 1
        
        return distance
    
    def are_equivalent(
        self,
        solution1: Dict[str, Any],
        solution2: Dict[str, Any]
    ) -> bool:
        """
        Verifica si dos soluciones son equivalentes (conectadas por un camino).
        
        Args:
            solution1: Primera solución
            solution2: Segunda solución
            
        Returns:
            True si existe un camino entre las soluciones
            
        Examples:
            >>> equiv = finder.are_equivalent(sol1, sol2)
        """
        path = self.find_path(solution1, solution2)
        return path is not None
    
    def find_all_paths(
        self,
        start: Dict[str, Any],
        end: Dict[str, Any],
        max_paths: int = 10
    ) -> List[SolutionPath]:
        """
        Encuentra múltiples caminos entre dos soluciones.
        
        Args:
            start: Solución inicial
            end: Solución final
            max_paths: Número máximo de caminos a encontrar
            
        Returns:
            Lista de caminos encontrados
        """
        paths = []
        
        # Implementación simple: encontrar camino mínimo
        # TODO: Implementar búsqueda de múltiples caminos
        min_path = self.find_path(start, end)
        if min_path:
            paths.append(min_path)
        
        return paths
    
    def get_solution_neighbors(
        self,
        solution: Dict[str, Any],
        valid_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Obtiene todas las soluciones vecinas de una solución dada.
        
        Args:
            solution: Solución de referencia
            valid_only: Si True, solo retorna vecinos válidos
            
        Returns:
            Lista de soluciones vecinas
        """
        neighbors = self._get_neighbors(solution)
        
        if valid_only:
            neighbors = [
                n for n in neighbors
                if self.bridge.verify_solution(n)
            ]
        
        return neighbors
    
    def clear_cache(self):
        """Limpia la caché de caminos."""
        self._path_cache.clear()
        logger.debug("Caché de caminos limpiada")
    
    def __str__(self) -> str:
        """Representación en string del PathFinder."""
        return f"PathFinder(max_depth={self.max_search_depth})"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        return (
            f"PathFinder("
            f"max_depth={self.max_search_depth}, "
            f"cached_paths={len(self._path_cache)})"
        )


# ============================================================================
# Funciones de Utilidad
# ============================================================================

def create_path_finder(bridge: CSPToCubicalBridge, max_depth: int = 10) -> PathFinder:
    """
    Crea un PathFinder desde un bridge.
    
    Args:
        bridge: Bridge CSP-Cubical
        max_depth: Profundidad máxima de búsqueda
    
    Returns:
        PathFinder configurado
    """
    return PathFinder(bridge, max_search_depth=max_depth)


# ============================================================================
# Ejemplo de Uso
# ============================================================================

def example_usage():
    """
    Ejemplo de uso de PathFinder.
    """
    from .csp_cubical_bridge import create_simple_csp_bridge
    
    logger.info("=== Ejemplo de PathFinder ===")
    
    # Crear bridge con CSP simple
    bridge = create_simple_csp_bridge(
        variables=['X', 'Y', 'Z'],
        domains={'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}},
        constraints=[
            ('X', 'Y', lambda x, y: x != y),
            ('Y', 'Z', lambda y, z: y != z)
        ]
    )
    
    # Crear PathFinder
    finder = PathFinder(bridge)
    
    logger.info(f"PathFinder: {finder}")
    
    # Definir soluciones
    solution1 = {'X': 1, 'Y': 2, 'Z': 1}
    solution2 = {'X': 2, 'Y': 3, 'Z': 1}
    
    logger.info(f"Solución 1: {solution1}")
    logger.info(f"Solución 2: {solution2}")
    
    # Calcular distancia de Hamming
    distance = finder.hamming_distance(solution1, solution2)
    logger.info(f"Distancia de Hamming: {distance}")
    
    # Buscar camino
    path = finder.find_path(solution1, solution2)
    
    if path:
        logger.info(f"Camino encontrado: {path}")
        logger.info(f"  Longitud: {len(path)}")
        logger.info(f"  Distancia: {path.distance}")
        if path.steps:
            logger.info(f"  Pasos intermedios:")
            for i, step in enumerate(path.steps, 1):
                logger.info(f"    {i}. {step}")
    else:
        logger.info("No se encontró camino")
    
    # Verificar equivalencia
    equiv = finder.are_equivalent(solution1, solution2)
    logger.info(f"Soluciones equivalentes: {equiv}")
    
    # Obtener vecinos
    neighbors = finder.get_solution_neighbors(solution1)
    logger.info(f"Vecinos válidos de {solution1}: {len(neighbors)}")
    for i, neighbor in enumerate(neighbors[:5], 1):
        logger.info(f"  {i}. {neighbor}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    example_usage()

