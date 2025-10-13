import networkx as nx
from typing import Dict

# Asumiendo que CubicalComplex está disponible en el mismo módulo o se importa
# from .cubical_complex import CubicalComplex # Esto se ajustará si es necesario

class HomologyEngine:
    def __init__(self):
        pass

    def compute_homology(self, cubical_complex) -> Dict[str, int]:
        """
        Calcula la homología cúbica de un complejo cúbico dado.

        Args:
            cubical_complex: Una instancia de CubicalComplex.

        Returns:
            Diccionario con los números de Betti (beta_0, beta_1, beta_2).
        """
        if not hasattr(cubical_complex, 'graph') or not hasattr(cubical_complex, 'cubes'):
            raise ValueError("El objeto cubical_complex debe tener atributos 'graph' y 'cubes'.")

        # Asegurarse de que el complejo esté construido
        if not getattr(cubical_complex, '_built', False):
            cubical_complex.build_complex()

        beta_0 = self._compute_beta0_cubical(cubical_complex)
        beta_1 = self._compute_beta1_cubical(cubical_complex)
        beta_2 = self._compute_beta2_cubical(cubical_complex)

        return {
            'beta_0': beta_0,
            'beta_1': beta_1,
            'beta_2': beta_2
        }

    def _compute_beta0_cubical(self, cubical_complex) -> int:
        """
        Calcula β₀: Número de componentes conexas.
        """
        return nx.number_connected_components(cubical_complex.graph)

    def _compute_beta1_cubical(self, cubical_complex) -> int:
        """
        Calcula β₁: Número de ciclos independientes.
        Usa la fórmula de Euler: β₁ = |E| - |V| + |C|
        donde E = aristas, V = vértices, C = componentes
        """
        num_edges = len(cubical_complex.cubes[1])
        num_vertices = len(cubical_complex.cubes[0])
        num_components = self._compute_beta0_cubical(cubical_complex)

        beta_1 = num_edges - num_vertices + num_components
        return max(0, beta_1)

    def _compute_beta2_cubical(self, cubical_complex) -> int:
        """
        Calcula β₂: Número de cavidades 2D.
        Aproximación: contar cuadrados que no son caras de cubos.
        """
        num_squares = len(cubical_complex.cubes[2])
        num_cubes = len(cubical_complex.cubes[3])

        # Cada cubo tiene 6 caras cuadradas
        # Los cuadrados que no son caras de cubos contribuyen a β₂
        beta_2 = max(0, num_squares - 6 * num_cubes)
        return beta_2

