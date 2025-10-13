from typing import List, Dict, Any
from .simplicial_complex import SimplicialComplex
from .homology_engine import HomologyEngine

class CascadeUpdater:
    """
    Implementa un mecanismo de actualización en cascada para detectar estructuras emergentes
    en un SimplicialComplex cuando se añade un nuevo elemento.
    """
    def __init__(self, simplicial_complex: SimplicialComplex, homology_engine: HomologyEngine):
        self.simplicial_complex = simplicial_complex
        self.homology_engine = homology_engine
        # La homología inicial de un complejo vacío es 0 en todas las dimensiones
        self.previous_homology = self.homology_engine.compute_homology(self.simplicial_complex)

    def add_element_and_update(self, simplex: List[int], filtration: float = 0.0) -> Dict[str, Any]:
        """
        Añade un nuevo simplex al complejo y desencadena una actualización en cascada.
        Detecta cambios en la homología y posibles estructuras emergentes.

        Args:
            simplex: El nuevo simplex a añadir.
            filtration: El valor de filtración del nuevo simplex.

        Returns:
            Un diccionario con los resultados de la actualización, incluyendo la homología actual
            y las estructuras emergentes detectadas.
        """
        # 1. Añadir el nuevo simplex al complejo
        self.simplicial_complex.add_simplex(simplex, filtration)

        # 2. Recalcular la homología del complejo actualizado
        current_homology = self.homology_engine.compute_homology(self.simplicial_complex)

        # 3. Detectar estructuras emergentes (cambios en los números de Betti)
        emergent_structures = self._detect_emergent_structures(self.previous_homology, current_homology)

        # 4. Actualizar la homología previa para la próxima iteración
        self.previous_homology = current_homology

        return {
            "current_homology": current_homology,
            "emergent_structures": emergent_structures
        }

    def _detect_emergent_structures(self, old_homology: Dict[str, int], new_homology: Dict[str, int]) -> Dict[str, Any]:
        """
        Compara la homología antigua y nueva para detectar cambios significativos.
        """
        emergent = {}
        # Asegurarse de que todas las dimensiones de interés estén presentes en ambos diccionarios
        all_dimensions = sorted(list(set(old_homology.keys()).union(new_homology.keys())))

        for dim_key in all_dimensions:
            old_beta = old_homology.get(dim_key, 0)
            new_beta = new_homology.get(dim_key, 0)
            
            dim_num = int(dim_key.split('_')[1])

            if new_beta > old_beta:
                emergent[dim_key] = f"Aumento en {new_beta - old_beta} componentes/ciclos/cavidades de dimensión {dim_num}"
            elif new_beta < old_beta:
                emergent[dim_key] = f"Disminución en {old_beta - new_beta} componentes/ciclos/cavidades de dimensión {dim_num}"
        return emergent

