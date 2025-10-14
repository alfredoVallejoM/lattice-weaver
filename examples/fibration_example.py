"""
Ejemplo de Uso del Módulo de Fibración

Este ejemplo demuestra cómo usar el módulo de Flujo de Fibración
para resolver un problema de coloración de grafos con coherencia multinivel.

Problema: Colorear un grafo de 4 nodos con 3 colores (0, 1, 2)
tal que nodos adyacentes tengan colores diferentes.

Grafo:
    A --- B
    |     |
    |     |
    C --- D

Restricciones:
- Nivel LOCAL: Nodos adyacentes deben tener colores diferentes
- Nivel PATTERN: Preferir distribución balanceada de colores
- Nivel GLOBAL: Minimizar el número total de colores usados
"""

from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    EnergyLandscape,
    EnergyComponents
)


def create_graph_coloring_problem():
    """
    Crea un problema de coloración de grafos con jerarquía de restricciones.
    
    Returns:
        Tupla (hierarchy, domains)
    """
    hierarchy = ConstraintHierarchy()
    
    # Variables: A, B, C, D
    # Dominios: {0, 1, 2} (tres colores)
    domains = {
        "A": [0, 1, 2],
        "B": [0, 1, 2],
        "C": [0, 1, 2],
        "D": [0, 1, 2]
    }
    
    # NIVEL 1: Restricciones locales (aristas del grafo)
    # A-B, A-C, B-D, C-D
    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
    
    for v1, v2 in edges:
        hierarchy.add_local_constraint(
            v1, v2,
            lambda a, v1=v1, v2=v2: a[v1] != a[v2],
            weight=1.0,
            hardness=Hardness.HARD,
            metadata={"edge": f"{v1}-{v2}"}
        )
    
    print(f"✓ Añadidas {len(edges)} restricciones locales (aristas)")
    
    # NIVEL 2: Restricción de patrón (distribución balanceada)
    def balanced_color_distribution(assignment):
        """
        Preferir que los colores estén distribuidos uniformemente.
        Devuelve el grado de desbalance (0.0 = perfectamente balanceado).
        """
        if len(assignment) < 4:
            return 0.0  # No evaluar hasta tener asignación completa
        
        colors = list(assignment.values())
        color_counts = {0: 0, 1: 0, 2: 0}
        
        for color in colors:
            color_counts[color] += 1
        
        # Calcular desviación estándar de las cuentas
        counts = list(color_counts.values())
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std_dev = variance ** 0.5
        
        # Normalizar: máxima desviación posible es cuando todos tienen el mismo color
        max_std_dev = ((4 - 4/3) ** 2 + (0 - 4/3) ** 2 + (0 - 4/3) ** 2) / 3
        max_std_dev = max_std_dev ** 0.5
        
        return std_dev / max_std_dev if max_std_dev > 0 else 0.0
    
    hierarchy.add_pattern_constraint(
        ["A", "B", "C", "D"],
        balanced_color_distribution,
        pattern_type="balanced_distribution",
        weight=0.5,
        hardness=Hardness.SOFT
    )
    
    print("✓ Añadida restricción de patrón (distribución balanceada)")
    
    # NIVEL 3: Restricción global (minimizar número de colores)
    def minimize_color_count(assignment):
        """
        Preferir usar el menor número de colores posible.
        Devuelve el número de colores únicos usados (normalizado).
        """
        if len(assignment) < 4:
            return 0.0  # No evaluar hasta tener asignación completa
        
        unique_colors = len(set(assignment.values()))
        # Normalizar: 1 color = 0.0, 3 colores = 1.0
        return (unique_colors - 1) / 2.0
    
    hierarchy.add_global_constraint(
        ["A", "B", "C", "D"],
        minimize_color_count,
        objective="minimize",
        weight=0.3,
        hardness=Hardness.SOFT
    )
    
    print("✓ Añadida restricción global (minimizar colores)")
    
    return hierarchy, domains


def solve_with_energy_landscape(hierarchy, domains):
    """
    Resuelve el problema usando el paisaje de energía.
    
    Implementa una búsqueda greedy guiada por el gradiente de energía.
    """
    landscape = EnergyLandscape(hierarchy)
    
    # Variables en orden
    variables = ["A", "B", "C", "D"]
    
    # Asignación inicial vacía
    assignment = {}
    
    print("\n" + "="*60)
    print("BÚSQUEDA GUIADA POR ENERGÍA")
    print("="*60)
    
    for var in variables:
        print(f"\n→ Asignando variable {var}...")
        
        # Calcular gradiente de energía
        gradient = landscape.compute_energy_gradient(assignment, var, domains[var])
        
        print(f"  Gradiente de energía:")
        for value, energy in sorted(gradient.items()):
            print(f"    {var}={value} → E={energy:.3f}")
        
        # Seleccionar valor de mínima energía
        best_value = min(gradient, key=gradient.get)
        best_energy = gradient[best_value]
        
        assignment[var] = best_value
        
        print(f"  ✓ Seleccionado {var}={best_value} (E={best_energy:.3f})")
        
        # Mostrar energía actual
        energy = landscape.compute_energy(assignment)
        print(f"  Energía actual: {energy}")
    
    return assignment


def analyze_solution(hierarchy, landscape, assignment):
    """
    Analiza la solución encontrada.
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE LA SOLUCIÓN")
    print("="*60)
    
    print(f"\nAsignación: {assignment}")
    
    # Calcular energía total
    energy = landscape.compute_energy(assignment)
    print(f"\nEnergía total: {energy}")
    
    # Verificar restricciones por nivel
    print("\nVerificación por nivel:")
    
    for level in ConstraintLevel:
        constraints = hierarchy.get_constraints_at_level(level)
        violated = []
        
        for i, constraint in enumerate(constraints):
            satisfied, violation = constraint.evaluate(assignment)
            if not satisfied or violation > 0:
                violated.append((i, constraint, violation))
        
        status = "✓ SATISFECHO" if len(violated) == 0 else f"✗ {len(violated)} violadas"
        print(f"  {level.name:8s}: {len(constraints)} restricciones → {status}")
        
        for i, constraint, violation in violated:
            print(f"    - Restricción {i}: violación={violation:.3f}")
    
    # Visualizar solución
    print("\nVisualización del grafo coloreado:")
    print(f"    {assignment['A']} --- {assignment['B']}")
    print(f"    |       |")
    print(f"    |       |")
    print(f"    {assignment['C']} --- {assignment['D']}")
    
    # Estadísticas de colores
    colors_used = set(assignment.values())
    print(f"\nColores usados: {sorted(colors_used)} ({len(colors_used)} colores)")


def main():
    """Función principal del ejemplo."""
    print("="*60)
    print("EJEMPLO: COLORACIÓN DE GRAFOS CON FLUJO DE FIBRACIÓN")
    print("="*60)
    
    # Crear problema
    print("\n1. Creando jerarquía de restricciones...")
    hierarchy, domains = create_graph_coloring_problem()
    
    # Mostrar estadísticas
    stats = hierarchy.get_statistics()
    print(f"\nEstadísticas de la jerarquía:")
    print(f"  Total de restricciones: {stats['total_constraints']}")
    print(f"  Por nivel: {stats['by_level']}")
    print(f"  Por dureza: {stats['by_hardness']}")
    
    # Crear paisaje de energía
    print("\n2. Creando paisaje de energía...")
    landscape = EnergyLandscape(hierarchy)
    print(f"  Pesos por nivel: {landscape.level_weights}")
    
    # Resolver
    print("\n3. Resolviendo problema...")
    assignment = solve_with_energy_landscape(hierarchy, domains)
    
    # Analizar solución
    print("\n4. Analizando solución...")
    analyze_solution(hierarchy, landscape, assignment)
    
    # Estadísticas del cache
    print("\n5. Estadísticas del cache:")
    cache_stats = landscape.get_cache_statistics()
    print(f"  Tamaño del cache: {cache_stats['cache_size']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Cache misses: {cache_stats['cache_misses']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
    print("\n" + "="*60)
    print("EJEMPLO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()

