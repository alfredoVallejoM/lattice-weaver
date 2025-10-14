"""
Generador de problemas de coloración de mapas (Map Coloring).

Este módulo implementa la familia de problemas de coloración de mapas,
una aplicación clásica de CSP donde se deben colorear regiones de un mapa
de manera que regiones adyacentes tengan colores diferentes.

El problema de coloración de mapas es equivalente a coloración de grafos,
donde cada región es un nodo y las adyacencias son aristas.

Características:
- Mapas predefinidos (USA, Europa, Australia, América del Sur)
- Generación de mapas planares aleatorios
- Integración con Graph Coloring
- Validador de soluciones
- Metadatos detallados

Referencias:
- Four Color Theorem: Appel, K., & Haken, W. (1977)
- Planar Graph Coloring: West, D. B. (2001). Introduction to Graph Theory
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import random

from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.problems.base import ProblemFamily
from lattice_weaver.problems.catalog import register_family
from lattice_weaver.problems.utils.validators import validate_graph_coloring_solution

logger = logging.getLogger(__name__)


# Mapas predefinidos: diccionario de adyacencias
PREDEFINED_MAPS = {
    'usa': {
        # Estados contiguos de USA (48 estados)
        'WA': ['OR', 'ID'],
        'OR': ['WA', 'ID', 'NV', 'CA'],
        'CA': ['OR', 'NV', 'AZ'],
        'ID': ['WA', 'OR', 'NV', 'UT', 'WY', 'MT'],
        'NV': ['OR', 'CA', 'AZ', 'UT', 'ID'],
        'UT': ['ID', 'NV', 'AZ', 'NM', 'CO', 'WY'],
        'AZ': ['CA', 'NV', 'UT', 'NM'],
        'MT': ['ID', 'WY', 'SD', 'ND'],
        'WY': ['MT', 'ID', 'UT', 'CO', 'NE', 'SD'],
        'CO': ['WY', 'UT', 'NM', 'OK', 'KS', 'NE'],
        'NM': ['AZ', 'UT', 'CO', 'OK', 'TX'],
        'ND': ['MT', 'SD', 'MN'],
        'SD': ['ND', 'MT', 'WY', 'NE', 'IA', 'MN'],
        'NE': ['SD', 'WY', 'CO', 'KS', 'MO', 'IA'],
        'KS': ['NE', 'CO', 'OK', 'MO'],
        'OK': ['KS', 'CO', 'NM', 'TX', 'AR', 'MO'],
        'TX': ['NM', 'OK', 'AR', 'LA'],
        'MN': ['ND', 'SD', 'IA', 'WI'],
        'IA': ['MN', 'SD', 'NE', 'MO', 'IL', 'WI'],
        'MO': ['IA', 'NE', 'KS', 'OK', 'AR', 'TN', 'KY', 'IL'],
        'AR': ['MO', 'OK', 'TX', 'LA', 'MS', 'TN'],
        'LA': ['TX', 'AR', 'MS'],
        'WI': ['MN', 'IA', 'IL', 'MI'],
        'IL': ['WI', 'IA', 'MO', 'KY', 'IN'],
        'MI': ['WI', 'IN', 'OH'],
        'IN': ['MI', 'IL', 'KY', 'OH'],
        'OH': ['MI', 'IN', 'KY', 'WV', 'PA'],
        'KY': ['IL', 'MO', 'TN', 'VA', 'WV', 'OH', 'IN'],
        'TN': ['KY', 'MO', 'AR', 'MS', 'AL', 'GA', 'NC', 'VA'],
        'MS': ['LA', 'AR', 'TN', 'AL'],
        'AL': ['MS', 'TN', 'GA', 'FL'],
        'WV': ['OH', 'KY', 'VA', 'MD', 'PA'],
        'VA': ['WV', 'KY', 'TN', 'NC', 'MD'],
        'NC': ['VA', 'TN', 'GA', 'SC'],
        'SC': ['NC', 'GA'],
        'GA': ['NC', 'TN', 'AL', 'FL', 'SC'],
        'FL': ['AL', 'GA'],
        'PA': ['OH', 'WV', 'MD', 'DE', 'NJ', 'NY'],
        'MD': ['PA', 'WV', 'VA', 'DE'],
        'DE': ['PA', 'MD', 'NJ'],
        'NJ': ['PA', 'DE', 'NY'],
        'NY': ['PA', 'NJ', 'CT', 'MA', 'VT'],
        'CT': ['NY', 'MA', 'RI'],
        'RI': ['CT', 'MA'],
        'MA': ['NY', 'CT', 'RI', 'NH', 'VT'],
        'VT': ['NY', 'MA', 'NH'],
        'NH': ['VT', 'MA', 'ME'],
        'ME': ['NH'],
    },
    
    'australia': {
        # Estados y territorios de Australia
        'WA': ['NT', 'SA'],
        'NT': ['WA', 'SA', 'QLD'],
        'SA': ['WA', 'NT', 'QLD', 'NSW', 'VIC'],
        'QLD': ['NT', 'SA', 'NSW'],
        'NSW': ['QLD', 'SA', 'VIC'],
        'VIC': ['SA', 'NSW'],
        'TAS': [],  # Tasmania es una isla
    },
    
    'europe_simple': {
        # Versión simplificada de Europa (países principales)
        'PT': ['ES'],
        'ES': ['PT', 'FR'],
        'FR': ['ES', 'BE', 'LU', 'DE', 'CH', 'IT'],
        'BE': ['FR', 'NL', 'LU', 'DE'],
        'NL': ['BE', 'DE'],
        'LU': ['FR', 'BE', 'DE'],
        'DE': ['NL', 'BE', 'LU', 'FR', 'CH', 'AT', 'CZ', 'PL', 'DK'],
        'CH': ['FR', 'DE', 'AT', 'IT'],
        'AT': ['DE', 'CH', 'IT', 'SI', 'HU', 'SK', 'CZ'],
        'IT': ['FR', 'CH', 'AT', 'SI'],
        'SI': ['IT', 'AT', 'HU', 'HR'],
        'HR': ['SI', 'HU', 'RS', 'BA'],
        'BA': ['HR', 'RS', 'ME'],
        'RS': ['HR', 'HU', 'RO', 'BG', 'ME', 'BA'],
        'ME': ['BA', 'RS', 'AL'],
        'AL': ['ME', 'GR'],
        'GR': ['AL', 'BG'],
        'BG': ['GR', 'RS', 'RO'],
        'RO': ['BG', 'RS', 'HU', 'UA'],
        'HU': ['AT', 'SI', 'HR', 'RS', 'RO', 'UA', 'SK'],
        'SK': ['AT', 'HU', 'UA', 'PL', 'CZ'],
        'CZ': ['DE', 'AT', 'SK', 'PL'],
        'PL': ['DE', 'CZ', 'SK', 'UA'],
        'UA': ['PL', 'SK', 'HU', 'RO'],
        'DK': ['DE'],
    },
    
    'south_america': {
        # América del Sur
        'CO': ['VE', 'BR', 'PE', 'EC'],
        'VE': ['CO', 'BR', 'GY'],
        'GY': ['VE', 'SR', 'BR'],
        'SR': ['GY', 'BR', 'GF'],
        'GF': ['SR', 'BR'],
        'EC': ['CO', 'PE'],
        'PE': ['EC', 'CO', 'BR', 'BO', 'CL'],
        'BR': ['VE', 'GY', 'SR', 'GF', 'CO', 'PE', 'BO', 'PY', 'AR', 'UY'],
        'BO': ['PE', 'BR', 'PY', 'AR', 'CL'],
        'PY': ['BO', 'BR', 'AR'],
        'CL': ['PE', 'BO', 'AR'],
        'AR': ['CL', 'BO', 'PY', 'BR', 'UY'],
        'UY': ['BR', 'AR'],
    },
}


class MapColoringProblem(ProblemFamily):
    """
    Familia de problemas de coloración de mapas.
    
    El problema consiste en asignar colores a regiones de un mapa de manera
    que regiones adyacentes tengan colores diferentes. Es equivalente al
    problema de coloración de grafos planares.
    
    Parámetros:
        map_name (str): Nombre del mapa predefinido o 'random' para generar aleatorio
        n_colors (int): Número de colores disponibles (default: 4)
        n_regions (int): Número de regiones (solo para mapas aleatorios)
        seed (int): Semilla para generación aleatoria (opcional)
    
    Mapas predefinidos:
        - 'usa': Estados contiguos de USA (48 regiones)
        - 'australia': Estados y territorios de Australia (7 regiones)
        - 'europe_simple': Países principales de Europa (25 regiones)
        - 'south_america': Países de América del Sur (13 regiones)
        - 'random': Mapa planar aleatorio
    
    Ejemplo:
        >>> from lattice_weaver.problems import get_catalog
        >>> catalog = get_catalog()
        >>> engine = catalog.generate_problem('map_coloring', map_name='australia', n_colors=3)
        >>> solution = {'WA': 0, 'NT': 1, 'SA': 2, 'QLD': 0, 'NSW': 1, 'VIC': 0, 'TAS': 0}
        >>> is_valid = catalog.validate_solution('map_coloring', solution, map_name='australia', n_colors=3)
    """
    
    def __init__(self):
        super().__init__(
            name='map_coloring',
            description='Coloración de mapas políticos - asignar colores a regiones sin que regiones adyacentes compartan color'
        )
        logger.info(f"Inicializada familia de problemas: {self.name}")
    
    def get_default_params(self) -> Dict[str, Any]:
        """Retorna parámetros por defecto."""
        return {
            'map_name': 'australia',
            'n_colors': 4
        }
    
    def get_param_schema(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna el esquema de parámetros para validación.
        
        Returns:
            Diccionario con especificación de parámetros
        """
        return {
            'map_name': {
                'type': str,
                'required': True,
                'choices': list(PREDEFINED_MAPS.keys()) + ['random'],
                'description': 'Nombre del mapa predefinido o "random" para generar aleatorio'
            },
            'n_colors': {
                'type': int,
                'required': True,
                'min': 2,
                'max': 10,
                'description': 'Número de colores disponibles'
            },
            'n_regions': {
                'type': int,
                'required': False,
                'min': 4,
                'max': 100,
                'description': 'Número de regiones (solo para mapas aleatorios)'
            },
            'seed': {
                'type': int,
                'required': False,
                'description': 'Semilla para generación aleatoria'
            }
        }
    
    def generate(self, **params) -> CSP:
        """
        Genera un problema de coloración de mapas.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            ArcEngine con el problema configurado
        """
        # Validar parámetros
        self.validate_params(**params)
        
        map_name = params['map_name']
        n_colors = params['n_colors']
        seed = params.get('seed', None)
        
        if seed is not None:
            random.seed(seed)
        
        logger.info(f"Generando problema Map Coloring: map={map_name}, colors={n_colors}")
        
        # Obtener mapa (predefinido o aleatorio)
        if map_name == 'random':
            n_regions = params.get('n_regions', 20)
            adjacency = self._generate_random_planar_map(n_regions, seed)
        else:
            adjacency = PREDEFINED_MAPS[map_name]
        
        # Crear ArcEngine
        csp_problem = CSP(variables=set(), domains={}, constraints=[], name=f"MapColoring_{map_name}")
        
        # Añadir variables (una por región)
        regions = sorted(adjacency.keys())
        for region in regions:
            domain = list(range(n_colors))
            csp_problem.add_variable(region, domain)
        
        logger.debug(f"Añadidas {len(regions)} variables (regiones)")
        
        # Añadir restricciones (regiones adyacentes deben tener colores diferentes)
        constraint_count = 0
        for region, neighbors in adjacency.items():
            for neighbor in neighbors:
                # Evitar duplicados (solo añadir una vez por par)
                if region < neighbor:
                    def different_colors(c1, c2):
                        return c1 != c2
                    
                    different_colors.__name__ = f'diff_{region}_{neighbor}'
                    cid = f'adj_{region}_{neighbor}'
                    csp_problem.add_constraint(Constraint(scope=frozenset({region, neighbor}), relation=different_colors, name=cid))
                    constraint_count += 1
        
        logger.info(f"Problema Map Coloring generado: {len(regions)} regiones, {constraint_count} restricciones")
        
        return csp_problem
    
    def _generate_random_planar_map(self, n_regions: int, seed: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Genera un mapa planar aleatorio.
        
        Usa un algoritmo simple que garantiza planaridad:
        1. Crear regiones en un grid
        2. Conectar regiones adyacentes en el grid
        3. Añadir algunas aristas aleatorias respetando planaridad
        
        Args:
            n_regions: Número de regiones
            seed: Semilla para reproducibilidad
        
        Returns:
            Diccionario de adyacencias
        """
        if seed is not None:
            random.seed(seed)
        
        # Crear grid aproximadamente cuadrado
        rows = int(n_regions ** 0.5)
        cols = (n_regions + rows - 1) // rows
        
        adjacency = {}
        regions = [f'R{i}' for i in range(n_regions)]
        
        # Inicializar diccionario
        for region in regions:
            adjacency[region] = []
        
        # Conectar regiones en grid
        for i in range(n_regions):
            row = i // cols
            col = i % cols
            region = regions[i]
            
            # Vecino derecha
            if col < cols - 1 and i + 1 < n_regions:
                neighbor = regions[i + 1]
                adjacency[region].append(neighbor)
                adjacency[neighbor].append(region)
            
            # Vecino abajo
            if row < rows - 1 and i + cols < n_regions:
                neighbor = regions[i + cols]
                adjacency[region].append(neighbor)
                adjacency[neighbor].append(region)
        
        # Eliminar duplicados
        for region in regions:
            adjacency[region] = list(set(adjacency[region]))
        
        return adjacency
    
    def validate_solution(self, solution: Dict[str, int], **params) -> bool:
        """
        Valida una solución del problema de coloración de mapas.
        
        Args:
            solution: Diccionario {región: color}
            **params: Parámetros del problema
        
        Returns:
            True si la solución es válida, False en caso contrario
        """
        map_name = params['map_name']
        n_colors = params['n_colors']
        seed = params.get('seed', None)
        
        # Obtener mapa
        if map_name == 'random':
            n_regions = params.get('n_regions', 20)
            adjacency = self._generate_random_planar_map(n_regions, seed)
        else:
            adjacency = PREDEFINED_MAPS[map_name]
        
        # Verificar que todas las regiones estén coloreadas
        regions = set(adjacency.keys())
        if set(solution.keys()) != regions:
            logger.debug(f"Regiones faltantes o extra en la solución")
            return False
        
        # Verificar que los colores sean válidos
        for region, color in solution.items():
            if not isinstance(color, int) or not 0 <= color < n_colors:
                logger.debug(f"Color inválido para {region}: {color}")
                return False
        
        # Verificar que regiones adyacentes tengan colores diferentes
        for region, neighbors in adjacency.items():
            region_color = solution[region]
            for neighbor in neighbors:
                if neighbor in solution:  # Verificar que el vecino exista
                    neighbor_color = solution[neighbor]
                    if region_color == neighbor_color:
                        logger.debug(f"Regiones adyacentes con mismo color: {region} y {neighbor}")
                        return False
        
        return True
    
    def get_metadata(self, **params) -> Dict[str, Any]:
        """
        Retorna metadatos del problema.
        
        Args:
            **params: Parámetros del problema
        
        Returns:
            Diccionario con metadatos
        """
        map_name = params['map_name']
        n_colors = params['n_colors']
        seed = params.get('seed', None)
        
        # Obtener mapa
        if map_name == 'random':
            n_regions = params.get('n_regions', 20)
            adjacency = self._generate_random_planar_map(n_regions, seed)
        else:
            adjacency = PREDEFINED_MAPS[map_name]
        
        # Calcular número de aristas
        n_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        
        # Calcular grado promedio
        avg_degree = (2 * n_edges) / len(adjacency) if len(adjacency) > 0 else 0
        
        # Estimar dificultad
        if len(adjacency) <= 10:
            difficulty = 'easy'
        elif len(adjacency) <= 30:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
        
        return {
            'family': self.name,
            'map_name': map_name,
            'n_regions': len(adjacency),
            'n_colors': n_colors,
            'n_variables': len(adjacency),
            'n_edges': n_edges,
            'n_constraints': n_edges,
            'domain_size': n_colors,
            'avg_degree': round(avg_degree, 2),
            'complexity': 'O(|E|)',
            'problem_type': 'graph_coloring',
            'is_planar': True,
            'chromatic_number_upper_bound': 4,  # Four Color Theorem
            'difficulty': difficulty
        }


# Auto-registrar la familia en el catálogo global
        if not get_catalog().is_registered('map_coloring'):
            register_family(MapColoringProblem())

logger.info("Familia MapColoringProblem registrada en el catálogo global")

