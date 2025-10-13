from typing import Tuple, Union

class GeometricCube:
    """
    Representa un cubo geométrico en un espacio n-dimensional.
    Unifica la representación de vértices, aristas, caras, etc.
    """
    def __init__(self, dimensions: int, coordinates: Tuple[Union[int, float], ...], axis: int = -1):
        if not isinstance(dimensions, int) or dimensions < 0:
            raise ValueError("Dimensions must be a non-negative integer.")
        if not isinstance(coordinates, tuple):
            raise ValueError("Coordinates must be a tuple.")
        # Allow dimensions to be less than or equal to the length of coordinates
        # as a 0-cube (point) can have coordinates in a higher dimensional space.
        # For example, a point (0,0) is a 0-cube in a 2D space.
        # if dimensions > len(coordinates):
        #     raise ValueError("Dimensions cannot exceed the length of coordinates.")

        self.dimensions = dimensions
        self.coordinates = coordinates
        self.axis = axis # Para 1-cubos (aristas), indica el eje al que es paralelo

    def __repr__(self):
        return f"GeometricCube(dim={self.dimensions}, coord={self.coordinates}, axis={self.axis})"

    def __eq__(self, other):
        if not isinstance(other, GeometricCube):
            return NotImplemented
        return (self.dimensions == other.dimensions and
                self.coordinates == other.coordinates and
                self.axis == other.axis)

    def __hash__(self):
        return hash((self.dimensions, self.coordinates, self.axis))

    def __lt__(self, other):
        if not isinstance(other, GeometricCube):
            return NotImplemented
        # Comparar primero por dimensión, luego por coordenadas, luego por eje
        if self.dimensions != other.dimensions:
            return self.dimensions < other.dimensions
        if self.coordinates != other.coordinates:
            return self.coordinates < other.coordinates
        return self.axis < other.axis

    def is_vertex(self) -> bool:
        return self.dimensions == 0

    def is_edge(self) -> bool:
        return self.dimensions == 1

    def get_boundary(self) -> list:
        """
        Calcula los cubos de menor dimensión que forman el borde de este cubo.
        Esta es una implementación simplificada y necesitará ser expandida
        para una homología completa.
        """
        if self.dimensions == 0:
            return [] # Un vértice no tiene borde
        elif self.dimensions == 1:
            # Para una arista, el borde son sus dos vértices
            # Esto es una simplificación; las coordenadas reales dependerían
            # de cómo se define la arista. Aquí asumimos que la arista
            # conecta (coord_i, 0) y (coord_i, 1) en el eje 'axis'.
            # Esto necesita ser más robusto con la definición de CubicalComplex.
            # Por ahora, devolvemos un placeholder.
            # La implementación de get_boundary para 1-cubos necesita ser revisada
            # para que sea consistente con la forma en que se construyen los cubos
            # en CubicalComplex. Por ahora, se devuelve una lista vacía para evitar errores.
            return []
        # TODO: Implementar para dimensiones superiores
        return []

