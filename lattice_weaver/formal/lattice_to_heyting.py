"""
Adaptador: Retículo de Conceptos → Álgebra de Heyting

Este módulo proporciona funciones para convertir el retículo de conceptos
construido por el LatticeBuilder (Capa 1) en un álgebra de Heyting,
dotando al sistema de una semántica lógica intuicionista.

La conversión permite:
1. Interpretar conceptos como proposiciones lógicas
2. Realizar operaciones lógicas sobre conceptos
3. Razonar de manera constructiva sobre el espacio de soluciones

Autor: LatticeWeaver Team
Fecha: 11 de Octubre de 2025
"""

from typing import List, Tuple, FrozenSet, Dict
from .heyting_algebra import HeytingAlgebra, HeytingElement
from lattice_weaver.lattice_core.builder import LatticeBuilder
import logging

logger = logging.getLogger(__name__)


def lattice_to_heyting(lattice_builder: LatticeBuilder, name: str = "H_Lattice") -> HeytingAlgebra:
    """
    Convierte un retículo de conceptos en un álgebra de Heyting.
    
    Cada concepto formal (extent, intent) se convierte en un elemento
    del álgebra de Heyting. El orden parcial del retículo se preserva.
    
    Args:
        lattice_builder: Constructor de retículos con conceptos construidos
        name: Nombre del álgebra resultante
    
    Returns:
        Álgebra de Heyting construida desde el retículo
    """
    if not lattice_builder.concepts:
        lattice_builder.build_lattice()
    
    algebra = HeytingAlgebra(name)
    
    # Mapeo de conceptos a elementos de Heyting
    concept_to_element: Dict[Tuple[FrozenSet, FrozenSet], HeytingElement] = {}
    
    # Crear elementos del álgebra
    for i, (extent, intent) in enumerate(lattice_builder.concepts):
        # Crear nombre descriptivo
        if len(extent) == 0:
            element_name = "⊤"  # Concepto top (extensión vacía)
        elif len(extent) == len(lattice_builder.context.objects):
            element_name = "⊥"  # Concepto bottom (todos los objetos)
        else:
            # Usar la intensión como nombre (más informativo)
            if len(intent) <= 3:
                intent_str = ",".join(sorted(str(a) for a in intent))
                element_name = f"C({intent_str})"
            else:
                element_name = f"C{i}"
        
        element = HeytingElement(element_name, extent)
        algebra.add_element(element)
        concept_to_element[(extent, intent)] = element
    
    # Identificar ⊥ y ⊤
    # En FCA: ⊤ tiene extensión vacía, ⊥ tiene intensión vacía
    for (extent, intent), element in concept_to_element.items():
        if len(extent) == 0:
            algebra.set_top(element)
        if len(intent) == 0:
            algebra.set_bottom(element)
    
    # Si no se encontró ⊥ o ⊤, usar los extremos
    if algebra.top is None:
        # El concepto con menor extensión
        min_concept = min(lattice_builder.concepts, key=lambda c: len(c[0]))
        algebra.set_top(concept_to_element[min_concept])
    
    if algebra.bottom is None:
        # El concepto con mayor extensión
        max_concept = max(lattice_builder.concepts, key=lambda c: len(c[0]))
        algebra.set_bottom(concept_to_element[max_concept])
    
    # Establecer orden parcial
    # En FCA: (E1, I1) ≤ (E2, I2) ⟺ E1 ⊆ E2 ⟺ I2 ⊆ I1
    for (extent1, intent1), elem1 in concept_to_element.items():
        for (extent2, intent2), elem2 in concept_to_element.items():
            if extent1.issubset(extent2):
                algebra.add_order(elem1, elem2)
    
    logger.info(f"Convertido retículo de {len(lattice_builder.concepts)} conceptos a álgebra de Heyting")
    
    return algebra


def concept_to_proposition(extent: FrozenSet, intent: FrozenSet) -> str:
    """
    Convierte un concepto formal en una proposición lógica legible.
    
    Args:
        extent: Extensión del concepto (objetos)
        intent: Intensión del concepto (atributos)
    
    Returns:
        Proposición lógica en forma de string
    """
    if len(intent) == 0:
        return "⊥ (contradicción)"
    
    if len(extent) == 0:
        return "⊤ (tautología)"
    
    # Construir proposición desde la intensión
    attrs = sorted(str(a) for a in intent)
    
    if len(attrs) == 1:
        return attrs[0]
    elif len(attrs) == 2:
        return f"{attrs[0]} ∧ {attrs[1]}"
    else:
        return " ∧ ".join(attrs[:3]) + (f" ∧ ... ({len(attrs)} atributos)" if len(attrs) > 3 else "")


def heyting_to_logic_table(algebra: HeytingAlgebra) -> str:
    """
    Genera una tabla de verdad de las operaciones lógicas del álgebra.
    
    Args:
        algebra: Álgebra de Heyting
    
    Returns:
        Tabla de verdad en formato string
    """
    lines = []
    lines.append("Tabla de Operaciones Lógicas")
    lines.append("=" * 70)
    
    # Seleccionar algunos elementos para la tabla (no todos si hay muchos)
    elements = sorted(algebra.elements, key=lambda e: e.name)[:6]
    
    # Tabla de conjunción (∧)
    lines.append("\nConjunción (∧):")
    lines.append("-" * 70)
    header = "  ∧  |" + "|".join(f"{e.name:^8}" for e in elements)
    lines.append(header)
    lines.append("-" * len(header))
    
    for a in elements:
        row = f"{a.name:^5}|"
        for b in elements:
            try:
                result = algebra.meet(a, b)
                row += f"{result.name:^8}|"
            except:
                row += f"{'?':^8}|"
        lines.append(row)
    
    # Tabla de disyunción (∨)
    lines.append("\nDisyunción (∨):")
    lines.append("-" * 70)
    header = "  ∨  |" + "|".join(f"{e.name:^8}" for e in elements)
    lines.append(header)
    lines.append("-" * len(header))
    
    for a in elements:
        row = f"{a.name:^5}|"
        for b in elements:
            try:
                result = algebra.join(a, b)
                row += f"{result.name:^8}|"
            except:
                row += f"{'?':^8}|"
        lines.append(row)
    
    # Tabla de implicación (→)
    lines.append("\nImplicación (→):")
    lines.append("-" * 70)
    header = "  →  |" + "|".join(f"{e.name:^8}" for e in elements)
    lines.append(header)
    lines.append("-" * len(header))
    
    for a in elements:
        row = f"{a.name:^5}|"
        for b in elements:
            try:
                result = algebra.implies(a, b)
                row += f"{result.name:^8}|"
            except:
                row += f"{'?':^8}|"
        lines.append(row)
    
    # Tabla de negación (¬)
    lines.append("\nNegación (¬):")
    lines.append("-" * 70)
    lines.append(f"{'a':^10}|{'¬a':^10}")
    lines.append("-" * 22)
    
    for a in elements:
        try:
            result = algebra.neg(a)
            lines.append(f"{a.name:^10}|{result.name:^10}")
        except:
            lines.append(f"{a.name:^10}|{'?':^10}")
    
    return "\n".join(lines)

