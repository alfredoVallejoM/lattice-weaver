"""
Utilidades para generación y validación de problemas CSP.
"""

from .graph_generators import (
    generate_random_graph,
    generate_complete_graph,
    generate_bipartite_graph,
    generate_grid_graph,
    generate_cycle_graph,
    generate_path_graph,
    generate_star_graph,
    generate_wheel_graph,
    edges_to_adjacency_dict,
    get_graph_chromatic_number_lower_bound,
)

from .validators import (
    validate_all_different,
    validate_binary_constraints,
    validate_nqueens_solution,
    validate_graph_coloring_solution,
    validate_sudoku_solution,
    validate_latin_square_solution,
    validate_magic_square_solution,
)

__all__ = [
    # Graph generators
    'generate_random_graph',
    'generate_complete_graph',
    'generate_bipartite_graph',
    'generate_grid_graph',
    'generate_cycle_graph',
    'generate_path_graph',
    'generate_star_graph',
    'generate_wheel_graph',
    'edges_to_adjacency_dict',
    'get_graph_chromatic_number_lower_bound',
    # Validators
    'validate_all_different',
    'validate_binary_constraints',
    'validate_nqueens_solution',
    'validate_graph_coloring_solution',
    'validate_sudoku_solution',
    'validate_latin_square_solution',
    'validate_magic_square_solution',
]

