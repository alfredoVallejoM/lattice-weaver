# lattice_weaver/arc_engine/ac31.py

import logging
from typing import TYPE_CHECKING, Any, Callable, Tuple, List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

if TYPE_CHECKING:
    from .core import ArcEngine

def revise_with_last_support(engine: 'ArcEngine', xi: str, xj: str, cid: str, relation_func: Callable[[Any, Any, Dict[str, Any]], bool], metadata: Dict[str, Any]) -> Tuple[bool, List[Any]]:
    """
    Revises the domain of variable Xi against Xj using the last support optimization.
    This is the core of the AC-3.1 algorithm.

    :param engine: The ArcEngine instance.
    :param xi: The variable whose domain is to be revised.
    :param xj: The constraining variable.
    :param cid: The ID of the constraint between xi and xj.
    :return: A tuple (revised, removed_values).
    """
    logger.debug(f"Revisando dominio de {xi} contra {xj} para restricción {cid}")
    revised = False
    removed_values = []
    di = engine.variables[xi]
    dj = engine.variables[xj]


    initial_di_values = list(di.get_values())
    for v in initial_di_values:
        logger.debug(f"  Considerando valor {v} para {xi}")
        key = (xi, xj, v)
        last_sup = engine.last_support.get(key)

        # Check if the last support is still valid
        if last_sup is not None and last_sup in dj and relation_func(v, last_sup, metadata):
            logger.debug(f"    Soporte válido encontrado para {v} en {xj} con {last_sup}")
            continue  # Support is still valid, move to next value

        # Last support is invalid, find a new one
        new_support_found = False
        for w in dj.get_values():
            # logger.debug(f"      Comprobando relación para {xi}={v} y {xj}={w}. Resultado: {relation_func(v, w, metadata)}")
            if relation_func(v, w, metadata):
                engine.last_support[key] = w
                new_support_found = True
                # logger.debug(f"    Nuevo soporte encontrado para {v} en {xj} con {w}")
                break
        
        if not new_support_found:
            logger.debug(f"    No se encontró soporte para {v} en {xi}. Eliminando {v}.")
            di.remove(v)
            removed_values.append(v)
            revised = True
            # Clean up last_support entry if the value is removed
            if key in engine.last_support:
                del engine.last_support[key]
    
    logger.debug(f"Revisión de {xi} contra {xj} finalizada. Revisado: {revised}. Valores eliminados: {removed_values}")

    return revised, removed_values

