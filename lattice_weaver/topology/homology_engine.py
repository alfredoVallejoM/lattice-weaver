import gudhi
from typing import Dict, Union
from .cubical_complex import CubicalComplex
from .simplicial_complex import SimplicialComplex

class HomologyEngine:
    def __init__(self):
        pass

    def compute_homology(self, complex_obj: Union[CubicalComplex, SimplicialComplex]) -> Dict[str, int]:
        if isinstance(complex_obj, CubicalComplex):
            if not hasattr(complex_obj, 'graph') or not hasattr(complex_obj, 'cubes'):
                raise ValueError("El objeto CubicalComplex debe tener atributos 'graph' y 'cubes'.")
            if not getattr(complex_obj, '_built', False):
                complex_obj.build_complex()

            beta_0 = self._compute_beta0_cubical(complex_obj)
            beta_1 = self._compute_beta1_cubical(complex_obj)
            beta_2 = self._compute_beta2_cubical(complex_obj)

            return {
                'beta_0': beta_0,
                'beta_1': beta_1,
                'beta_2': beta_2
            }
        elif isinstance(complex_obj, SimplicialComplex):
            if not complex_obj.is_built() or complex_obj.num_simplices() == 0:
                return {"beta_0": 0, "beta_1": 0, "beta_2": 0}

            if complex_obj.num_simplices() == complex_obj.num_vertices() and complex_obj.num_vertices() > 0:
                return {"beta_0": complex_obj.num_vertices(), "beta_1": 0, "beta_2": 0}

            st_copy = complex_obj.simplex_tree.copy()
            st_copy.compute_persistence()
            betti_numbers_list = st_copy.betti_numbers()

            homology_dict = {f'beta_{i}': beta for i, beta in enumerate(betti_numbers_list)}
            for i in range(3):
                if f'beta_{i}' not in homology_dict:
                    homology_dict[f'beta_{i}'] = 0
            return homology_dict
        else:
            raise TypeError("Tipo de complejo no soportado. Debe ser CubicalComplex o SimplicialComplex.")

    def _compute_beta0_cubical(self, cubical_complex: CubicalComplex) -> int:
        import networkx as nx
        return nx.number_connected_components(cubical_complex.graph)

    def _compute_beta1_cubical(self, cubical_complex: CubicalComplex) -> int:
        num_edges = len(cubical_complex.cubes[1])
        num_vertices = len(cubical_complex.cubes[0])
        num_components = self._compute_beta0_cubical(cubical_complex)

        beta_1 = num_edges - num_vertices + num_components
        return max(0, beta_1)

    def _compute_beta2_cubical(self, cubical_complex: CubicalComplex) -> int:
        num_squares = len(cubical_complex.cubes[2])
        num_cubes = len(cubical_complex.cubes[3])
        beta_2 = max(0, num_squares - 6 * num_cubes)
        return beta_2

