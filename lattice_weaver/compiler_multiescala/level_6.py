"""
Nivel L6: Problema Completo

Este nivel representa el CSP completo como una unidad de abstracción máxima.
Integra todos los niveles anteriores (L0-L5) y proporciona una vista global del problema.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .base import AbstractionLevel
from .level_5 import Level5, MetaPattern


@dataclass
class ProblemDescription:
    """Descripción del problema CSP completo."""
    name: str
    description: str
    domain: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.name, self.description, self.domain))


class Level6(AbstractionLevel):
    """
    Nivel 6: Problema Completo
    
    Representa el CSP completo como una unidad de abstracción máxima.
    Integra todos los niveles anteriores y proporciona una vista global del problema.
    """
    
    def __init__(self, problem: ProblemDescription, level5: Optional[Level5] = None):
        """
        Inicializa el Nivel L6.
        
        Args:
            problem: Descripción del problema CSP completo
            level5: Representación de L5 (opcional)
        """
        self.problem = problem
        self.level5 = level5
    
    def build_from_lower(self, lower_level: Level5, problem_name: str = "CSP Problem", 
                        problem_description: str = "Generic CSP Problem",
                        problem_domain: str = "General"):
        """
        Construye L6 desde L5.
        
        Args:
            lower_level: Instancia de Level5
            problem_name: Nombre del problema
            problem_description: Descripción del problema
            problem_domain: Dominio del problema
            
        Returns:
            Instancia de Level6
        """
        # Inferir propiedades del problema desde L5
        stats = lower_level.get_statistics()
        
        properties = {
            'num_meta_patterns': stats['num_meta_patterns'],
            'num_isolated_concepts': stats['num_isolated_concepts'],
            'total_concepts': stats['num_meta_patterns'] + stats['num_isolated_concepts'],
            'has_meta_patterns': stats['num_meta_patterns'] > 0,
        }
        
        problem = ProblemDescription(
            name=problem_name,
            description=problem_description,
            domain=problem_domain,
            properties=properties
        )
        
        self.problem = problem
        self.level5 = lower_level
    
    def refine_to_lower(self) -> Level5:
        """
        Refina L6 a L5.
        
        Returns:
            Instancia de Level5
        """
        if self.level5 is None:
            raise ValueError("Cannot refine L6 without L5 information")
        
        return self.level5
    
    def renormalize(self, **kwargs) -> 'Level6':
        """
        Renormaliza L6.
        
        Delega la renormalización a L5 y reconstruye L6.
        
        Returns:
            Nueva instancia de Level6 renormalizada
        """
        if self.level5 is None:
            return self
        
        # Renormalizar L5
        renormalized_l5 = self.level5.renormalize(**kwargs)
        
        # Reconstruir L6 con L5 renormalizado
        new_l6 = Level6(
            problem=ProblemDescription(
                name=self.problem.name,
                description=self.problem.description,
                domain=self.problem.domain
            )
        )
        new_l6.build_from_lower(
            renormalized_l5,
            problem_name=self.problem.name,
            problem_description=self.problem.description,
            problem_domain=self.problem.domain
        )
        return new_l6
    
    def validate(self) -> bool:
        """
        Valida L6.
        
        Returns:
            True si L6 es válido, False en caso contrario
        """
        # Validar que el problema tenga un nombre y descripción
        if not self.problem.name or not self.problem.description:
            return False
        
        # Validar que el problema tenga un dominio
        if not self.problem.domain:
            return False
        
        # Validar L5 si existe
        if self.level5 is not None:
            if not self.level5.validate():
                return False
        
        return True
    
    @property
    def complexity(self) -> float:
        """
        Calcula la complejidad de L6.
        
        La complejidad de L6 es la complejidad de L5 si existe,
        o 0 si no hay L5.
        
        Returns:
            Complejidad de L6
        """
        if self.level5 is None:
            return 0.0
        
        return self.level5.complexity
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de L6.
        
        Returns:
            Diccionario con estadísticas de L6
        """
        stats = {
            'problem_name': self.problem.name,
            'problem_description': self.problem.description,
            'problem_domain': self.problem.domain,
            'problem_properties': self.problem.properties,
            'complexity': self.complexity,
        }
        
        # Agregar estadísticas de L5 si existe
        if self.level5 is not None:
            l5_stats = self.level5.get_statistics()
            stats['level5_stats'] = l5_stats
        
        return stats

