import gudhi
from typing import List, Tuple, Set, Dict

class SimplicialComplex:
    """
    Representa un complejo simplicial utilizando la estructura de datos SimplexTree de GUDHI.
    """
    def __init__(self):
        self.simplex_tree = gudhi.SimplexTree()
        self._built = False

    def add_simplex(self, simplex: List[int], filtration: float = 0.0):
        """
        Añade un simplex al complejo simplicial.

        Args:
            simplex: Una lista de enteros que representan los vértices del simplex.
            filtration: El valor de filtración del simplex.
        """
        if not all(isinstance(v, int) for v in simplex):
            raise ValueError("Los vértices del simplex deben ser enteros.")
        # GUDHI SimplexTree automáticamente añade las caras de un simplex.
        # Para asegurar que los vértices se consideren, basta con añadir el simplex.
        # La filtración por defecto 0.0 es adecuada para la mayoría de los casos.
        self.simplex_tree.insert(simplex, filtration=filtration)
        self._built = True

    def get_simplices(self, dimension: int = -1) -> List[Tuple[List[int], float]]:
        """
        Obtiene todos los simplices de una dimensión específica o todos los simplices si dimension es -1.

        Args:
            dimension: La dimensión de los simplices a obtener. -1 para todos.

        Returns:
            Una lista de tuplas, donde cada tupla contiene el simplex (lista de vértices) y su valor de filtración.
        """
        if dimension == -1:
            return list(self.simplex_tree.get_simplices())
        else:
            # Filtrar los simplices para obtener solo los de la dimensión exacta
            exact_dim_simplices = []
            for simplex, filtration in self.simplex_tree.get_skeleton(dimension):
                if len(simplex) - 1 == dimension:
                    exact_dim_simplices.append((simplex, filtration))
            return exact_dim_simplices

    def num_simplices(self) -> int:
        """
        Retorna el número total de simplices en el complejo.
        """
        return self.simplex_tree.num_simplices()

    def num_vertices(self) -> int:
        """
        Retorna el número de vértices (0-simplices) en el complejo.
        """
        return self.simplex_tree.num_vertices()

    def get_max_dimension(self) -> int:
        """
        Retorna la dimensión máxima de los simplices en el complejo.
        """
        return self.simplex_tree.dimension()

    def __str__(self):
        simplices_str = []
        for simplex, filtration in self.simplex_tree.get_simplices():
            simplices_str.append(f"({simplex}, {filtration})")
        return f"SimplicialComplex(num_vertices={self.num_vertices()}, num_simplices={self.num_simplices()}, max_dim={self.get_max_dimension()})\nSimplices: {', '.join(simplices_str)}"

    def __repr__(self):
        return self.__str__()

    def is_built(self) -> bool:
        """
        Indica si el complejo simplicial ha sido construido (es decir, si se han añadido simplices).
        """
        return self._built

