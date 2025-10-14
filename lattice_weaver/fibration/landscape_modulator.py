from typing import Dict, Callable, Optional
from .energy_landscape_optimized import EnergyLandscapeOptimized
from .constraint_hierarchy import ConstraintLevel

class ModulationStrategy:
    """Estrategia base para la modulación del paisaje."""
    
    def __init__(self, name: str):
        self.name = name
        
    def compute_modulation(self, landscape: EnergyLandscapeOptimized, 
                          context: Dict) -> Dict[ConstraintLevel, float]:
        """
        Calcula los factores de modulación para cada nivel de restricción.
        
        Args:
            landscape: Paisaje de energía a modular.
            context: Contexto actual (estado de búsqueda, estadísticas, etc.).
            
        Returns:
            Diccionario {nivel: factor_multiplicativo}.
        """
        raise NotImplementedError

class FocusOnLocalStrategy(ModulationStrategy):
    """Estrategia que enfatiza restricciones locales."""
    
    def __init__(self):
        super().__init__("focus_on_local")
        
    def compute_modulation(self, landscape: EnergyLandscapeOptimized, 
                          context: Dict) -> Dict[ConstraintLevel, float]:
        return {
            ConstraintLevel.LOCAL: 2.0,    # Duplicar peso de restricciones locales
            ConstraintLevel.PATTERN: 1.0,
            ConstraintLevel.GLOBAL: 0.5    # Reducir peso de restricciones globales
        }

class FocusOnGlobalStrategy(ModulationStrategy):
    """Estrategia que enfatiza restricciones globales."""
    
    def __init__(self):
        super().__init__("focus_on_global")
        
    def compute_modulation(self, landscape: EnergyLandscapeOptimized, 
                          context: Dict) -> Dict[ConstraintLevel, float]:
        return {
            ConstraintLevel.LOCAL: 0.5,
            ConstraintLevel.PATTERN: 1.0,
            ConstraintLevel.GLOBAL: 2.0    # Duplicar peso de restricciones globales
        }

class AdaptiveStrategy(ModulationStrategy):
    """Estrategia adaptativa basada en el progreso de la búsqueda."""
    
    def __init__(self):
        super().__init__("adaptive")
        
    def compute_modulation(self, landscape: EnergyLandscapeOptimized, 
                          context: Dict) -> Dict[ConstraintLevel, float]:
        # Extraer métricas del contexto
        progress = context.get("progress", 0.0)  # 0.0 = inicio, 1.0 = casi completo
        local_violations = context.get("local_violations", 0)
        global_violations = context.get("global_violations", 0)
        
        # Estrategia: al inicio, enfocarse en restricciones locales
        # A medida que avanza, enfocarse en restricciones globales
        local_weight = 2.0 - progress  # 2.0 -> 1.0
        global_weight = 1.0 + progress  # 1.0 -> 2.0
        
        # Ajustar según violaciones
        if local_violations > global_violations:
            local_weight *= 1.5
        elif global_violations > local_violations:
            global_weight *= 1.5
            
        return {
            ConstraintLevel.LOCAL: local_weight,
            ConstraintLevel.PATTERN: 1.0,
            ConstraintLevel.GLOBAL: global_weight
        }

class LandscapeModulator:
    """
    Modula dinámicamente el paisaje de energía según el contexto.
    """
    
    def __init__(self, landscape: EnergyLandscapeOptimized):
        self.landscape = landscape
        self.current_strategy: Optional[ModulationStrategy] = None
        self.base_weights = landscape.level_weights.copy()
        
    def set_strategy(self, strategy: ModulationStrategy):
        """Establece la estrategia de modulación."""
        self.current_strategy = strategy
        
    def apply_modulation(self, context: Dict):
        """
        Aplica la modulación al paisaje según el contexto actual.
        """
        if self.current_strategy is None:
            return
        
        # Calcular factores de modulación
        modulation_factors = self.current_strategy.compute_modulation(
            self.landscape, context
        )
        
        # Aplicar modulación
        for level, factor in modulation_factors.items():
            self.landscape.level_weights[level] = self.base_weights[level] * factor
            
        # Limpiar cache (los pesos han cambiado)
        self.landscape.clear_cache()
        
    def reset_modulation(self):
        """
        Restaura los pesos base del paisaje de energía.
        """
        self.landscape.level_weights = self.base_weights.copy()
        self.landscape.clear_cache()

    def get_statistics(self) -> Dict:
        """Devuelve estadísticas del modulador."""
        return {
            "current_strategy": self.current_strategy.name if self.current_strategy else "None",
            "base_weights": {level.name: weight for level, weight in self.base_weights.items()},
            "current_weights": {level.name: weight for level, weight in self.landscape.level_weights.items()}
        }

