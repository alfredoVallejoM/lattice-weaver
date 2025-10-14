from abc import ABC, abstractmethod

class AbstractionLevel(ABC):
    """Interfaz para un nivel de abstracción en el compilador multiescala."""

    def __init__(self, level: int, config: dict = None):
        self.level = level
        self.config = config or {}
        self.data = None  # Estructura de datos específica del nivel

    @abstractmethod
    def build_from_lower(self, lower_level: 'AbstractionLevel'):
        """Construye la representación de este nivel a partir de un nivel inferior."""
        pass

    @abstractmethod
    def refine_to_lower(self) -> 'AbstractionLevel':
        """Refina la representación de este nivel a un nivel inferior."""
        pass

    @abstractmethod
    def renormalize(self, partitioner, k: int) -> 'AbstractionLevel':
        """Aplica la renormalización en este nivel de abstracción."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Valida la coherencia interna de la representación de este nivel."""
        pass

    @property
    @abstractmethod
    def complexity(self) -> float:
        """Calcula una métrica de complejidad para este nivel."""
        pass

