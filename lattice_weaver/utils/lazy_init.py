"""
Lazy Initialization - Sistema de Inicialización Perezosa

Utilidades para inicializar estructuras solo cuando se necesiten.

Autor: Agente Autónomo - Lattice Weaver
Fecha: 15 de Octubre, 2025
"""

from typing import TypeVar, Generic, Callable, Optional
from functools import wraps


T = TypeVar('T')


class LazyProperty(Generic[T]):
    """
    Descriptor para propiedades lazy.
    
    La propiedad se inicializa solo cuando se accede por primera vez.
    """
    
    def __init__(self, initializer: Callable[[any], T]):
        """
        Inicializa LazyProperty.
        
        Args:
            initializer: Función que inicializa la propiedad
        """
        self.initializer = initializer
        self.attr_name = None
    
    def __set_name__(self, owner, name):
        """Guarda el nombre del atributo."""
        self.attr_name = f"_lazy_{name}"
    
    def __get__(self, instance, owner):
        """Obtiene el valor, inicializando si es necesario."""
        if instance is None:
            return self
        
        # Verificar si ya está inicializado
        if not hasattr(instance, self.attr_name):
            # Inicializar
            value = self.initializer(instance)
            setattr(instance, self.attr_name, value)
        
        return getattr(instance, self.attr_name)
    
    def __set__(self, instance, value):
        """Establece el valor directamente."""
        setattr(instance, self.attr_name, value)


class LazyObject(Generic[T]):
    """
    Wrapper para objetos lazy.
    
    El objeto se crea solo cuando se accede por primera vez.
    """
    
    def __init__(self, factory: Callable[[], T]):
        """
        Inicializa LazyObject.
        
        Args:
            factory: Función que crea el objeto
        """
        self._factory = factory
        self._value: Optional[T] = None
        self._initialized = False
    
    def get(self) -> T:
        """
        Obtiene el objeto, creándolo si es necesario.
        
        Returns:
            Objeto inicializado
        """
        if not self._initialized:
            self._value = self._factory()
            self._initialized = True
        return self._value
    
    def is_initialized(self) -> bool:
        """
        Verifica si el objeto está inicializado.
        
        Returns:
            True si está inicializado
        """
        return self._initialized
    
    def reset(self):
        """Resetea el objeto lazy."""
        self._value = None
        self._initialized = False


def lazy_init(factory: Callable[[], T]) -> LazyObject[T]:
    """
    Crea un objeto lazy.
    
    Args:
        factory: Función que crea el objeto
    
    Returns:
        LazyObject wrapper
    """
    return LazyObject(factory)


def lazy_method(method):
    """
    Decorator para métodos lazy.
    
    El resultado del método se cachea después de la primera llamada.
    """
    attr_name = f"_lazy_method_{method.__name__}"
    
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Solo funciona sin argumentos
        if args or kwargs:
            return method(self, *args, **kwargs)
        
        # Verificar si ya está cacheado
        if not hasattr(self, attr_name):
            result = method(self)
            setattr(self, attr_name, result)
        
        return getattr(self, attr_name)
    
    return wrapper


class ConditionalInit:
    """
    Inicialización condicional basada en flags.
    
    Permite activar/desactivar componentes dinámicamente.
    """
    
    def __init__(self):
        """Inicializa ConditionalInit."""
        self._components: dict = {}
        self._enabled: dict = {}
    
    def register(self, name: str, factory: Callable[[], T], enabled: bool = True):
        """
        Registra un componente.
        
        Args:
            name: Nombre del componente
            factory: Función que crea el componente
            enabled: Si el componente está habilitado
        """
        self._components[name] = LazyObject(factory)
        self._enabled[name] = enabled
    
    def get(self, name: str) -> Optional[T]:
        """
        Obtiene un componente.
        
        Args:
            name: Nombre del componente
        
        Returns:
            Componente si está habilitado, None si no
        """
        if name not in self._components:
            return None
        
        if not self._enabled[name]:
            return None
        
        return self._components[name].get()
    
    def enable(self, name: str):
        """Habilita un componente."""
        if name in self._enabled:
            self._enabled[name] = True
    
    def disable(self, name: str):
        """Deshabilita un componente."""
        if name in self._enabled:
            self._enabled[name] = False
    
    def is_enabled(self, name: str) -> bool:
        """Verifica si un componente está habilitado."""
        return self._enabled.get(name, False)
    
    def is_initialized(self, name: str) -> bool:
        """Verifica si un componente está inicializado."""
        if name not in self._components:
            return False
        return self._components[name].is_initialized()


# Ejemplo de uso:
#
# class MyClass:
#     @LazyProperty
#     def expensive_property(self):
#         # Se inicializa solo cuando se accede
#         return compute_expensive_value()
#
#     @lazy_method
#     def expensive_method(self):
#         # Se cachea después de primera llamada
#         return compute_expensive_result()

