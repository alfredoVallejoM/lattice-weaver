# lattice_weaver/arc_engine/ac31.py

from typing import TYPE_CHECKING, Any, Callable, Tuple, List

if TYPE_CHECKING:
    from .core import ArcEngine

def revise_with_last_support(engine: 'ArcEngine', xi: str, xj: str, cid: str) -> Tuple[bool, List[Any]]:
    """
    Revises the domain of variable Xi against Xj using the last support optimization.
    This is the core of the AC-3.1 algorithm.

    :param engine: The ArcEngine instance.
    :param xi: The variable whose domain is to be revised.
    :param xj: The constraining variable.
    :param cid: The ID of the constraint between xi and xj.
    :return: A tuple (revised, removed_values).
    """
    revised = False
    removed_values = []
    di = engine.variables[xi]
    dj = engine.variables[xj]
    relation = engine.constraints[cid].relation

    for v in list(di.get_values()):
        key = (xi, xj, v)
        last_sup = engine.last_support.get(key)

        # Check if the last support is still valid
        if last_sup is not None and last_sup in dj and relation(v, last_sup):
            continue  # Support is still valid, move to next value

        # Last support is invalid, find a new one
        new_support_found = False
        for w in dj.get_values():
            if relation(v, w):
                engine.last_support[key] = w
                new_support_found = True
                break
        
        if not new_support_found:
            di.remove(v)
            removed_values.append(v)
            revised = True
            # Clean up last_support entry if the value is removed
            if key in engine.last_support:
                del engine.last_support[key]

    return revised, removed_values

