"""
Catálogo centralizado de familias de problemas CSP.

Este módulo proporciona un registro global donde se pueden registrar,
buscar y listar todas las familias de problemas disponibles.
"""

from typing import Dict, List, Optional, Any
import logging

from .base import ProblemFamily

logger = logging.getLogger(__name__)


class ProblemCatalog:
    """
    Catálogo centralizado de familias de problemas CSP.
    
    Permite registrar, buscar y listar familias de problemas.
    Implementa el patrón Singleton para mantener un registro global.
    
    Attributes:
        _families: Diccionario de familias registradas (nombre -> ProblemFamily)
    """
    
    def __init__(self):
        """Inicializa el catálogo vacío."""
        self._families: Dict[str, ProblemFamily] = {}
        logger.debug("Inicializado ProblemCatalog")
    
    def register(self, family: ProblemFamily) -> None:
        """
        Registra una familia de problemas en el catálogo.
        
        Args:
            family: Instancia de ProblemFamily a registrar
            
        Raises:
            ValueError: Si ya existe una familia con ese nombre
            TypeError: Si family no es una instancia de ProblemFamily
        """
        if not isinstance(family, ProblemFamily):
            raise TypeError(
                f"Se esperaba una instancia de ProblemFamily, "
                f"recibido {type(family).__name__}"
            )
        
        if family.name in self._families:
            raise ValueError(
                f"Familia '{family.name}' ya está registrada. "
                f"Use unregister() primero si desea reemplazarla."
            )
        
        self._families[family.name] = family
        logger.info(f"Registrada familia: {family.name}")
    
    def unregister(self, name: str) -> None:
        """
        Elimina una familia del catálogo.
        
        Args:
            name: Nombre de la familia a eliminar
            
        Raises:
            KeyError: Si la familia no existe
        """
        if name not in self._families:
            raise KeyError(f"Familia '{name}' no encontrada en el catálogo")
        
        del self._families[name]
        logger.info(f"Eliminada familia: {name}")
    
    def get(self, name: str) -> Optional[ProblemFamily]:
        """
        Obtiene una familia por nombre.
        
        Args:
            name: Nombre de la familia
            
        Returns:
            ProblemFamily si existe, None en caso contrario
        """
        return self._families.get(name)
    
    def has(self, name: str) -> bool:
        """
        Verifica si una familia está registrada.
        
        Args:
            name: Nombre de la familia
            
        Returns:
            bool: True si la familia existe
        """
        return name in self._families
    
    def list_families(self) -> List[str]:
        """
        Lista todas las familias registradas.
        
        Returns:
            Lista de nombres de familias (ordenada alfabéticamente)
        """
        return sorted(self._families.keys())
    
    def get_all_families(self) -> Dict[str, ProblemFamily]:
        """
        Obtiene todas las familias registradas.
        
        Returns:
            Dict nombre -> ProblemFamily
        """
        return dict(self._families)
    
    def generate_problem(self, family_name: str, **params):
        """
        Genera un problema de una familia específica.
        
        Método de conveniencia que busca la familia y llama a su
        método generate().
        
        Args:
            family_name: Nombre de la familia
            **params: Parámetros del problema
            
        Returns:
            ArcEngine configurado con el problema
            
        Raises:
            ValueError: Si la familia no existe
        """
        family = self.get(family_name)
        if family is None:
            available = ', '.join(self.list_families())
            raise ValueError(
                f"Familia desconocida: '{family_name}'. "
                f"Familias disponibles: {available}"
            )
        
        logger.info(f"Generando problema de familia '{family_name}' con params: {params}")
        return family.generate(**params)
    
    def validate_solution(self, family_name: str, solution: Dict[str, Any], **params) -> bool:
        """
        Valida una solución para un problema de una familia específica.
        
        Args:
            family_name: Nombre de la familia
            solution: Solución a validar
            **params: Parámetros del problema
            
        Returns:
            bool: True si la solución es válida
            
        Raises:
            ValueError: Si la familia no existe
        """
        family = self.get(family_name)
        if family is None:
            raise ValueError(f"Familia desconocida: '{family_name}'")
        
        return family.validate_solution(solution, **params)
    
    def get_metadata(self, family_name: str, **params) -> Dict[str, Any]:
        """
        Obtiene metadatos de un problema de una familia específica.
        
        Args:
            family_name: Nombre de la familia
            **params: Parámetros del problema
            
        Returns:
            Dict con metadatos
            
        Raises:
            ValueError: Si la familia no existe
        """
        family = self.get(family_name)
        if family is None:
            raise ValueError(f"Familia desconocida: '{family_name}'")
        
        return family.get_metadata(**params)
    
    def print_catalog(self) -> None:
        """
        Imprime un resumen del catálogo en formato legible.
        """
        families = self.list_families()
        if not families:
            print("Catálogo vacío - no hay familias registradas")
            return
        
        print(f"\n{'='*70}")
        print(f"CATÁLOGO DE FAMILIAS DE PROBLEMAS CSP ({len(families)} familias)")
        print(f"{'='*70}\n")
        
        for name in families:
            family = self._families[name]
            print(f"  • {name}")
            print(f"    {family.description}")
            
            # Mostrar parámetros si están disponibles
            schema = family.get_param_schema()
            if schema:
                print(f"    Parámetros:")
                for param_name, param_info in schema.items():
                    required = " (requerido)" if param_info.get('required') else ""
                    print(f"      - {param_name}{required}: {param_info.get('description', 'N/A')}")
            print()
        
        print(f"{'='*70}\n")
    
    def clear(self) -> None:
        """
        Elimina todas las familias del catálogo.
        
        Útil para testing.
        """
        self._families.clear()
        logger.info("Catálogo limpiado")
    
    def __len__(self) -> int:
        """Retorna el número de familias registradas."""
        return len(self._families)
    
    def __contains__(self, name: str) -> bool:
        """Permite usar 'in' para verificar si una familia existe."""
        return name in self._families
    
    def __repr__(self) -> str:
        """Representación string del catálogo."""
        return f"<ProblemCatalog: {len(self._families)} familias>"


# Instancia global del catálogo (Singleton)
_global_catalog = ProblemCatalog()


def get_catalog() -> ProblemCatalog:
    """
    Retorna la instancia global del catálogo.
    
    Returns:
        ProblemCatalog: Instancia singleton del catálogo
    """
    return _global_catalog


def register_family(family: ProblemFamily) -> None:
    """
    Registra una familia en el catálogo global.
    
    Función de conveniencia equivalente a get_catalog().register(family).
    
    Args:
        family: Familia a registrar
    """
    _global_catalog.register(family)


def get_family(name: str) -> Optional[ProblemFamily]:
    """
    Obtiene una familia del catálogo global.
    
    Función de conveniencia equivalente a get_catalog().get(name).
    
    Args:
        name: Nombre de la familia
        
    Returns:
        ProblemFamily si existe, None en caso contrario
    """
    return _global_catalog.get(name)

