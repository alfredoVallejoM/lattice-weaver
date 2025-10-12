#!/usr/bin/env python3
"""
Ejemplo de Demostración - LatticeWeaver v4 Fase 1
Reglas de Homotopía Precomputadas
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lattice_weaver import ArcEngineExtended

def main():
    print("=" * 70)
    print("LatticeWeaver v4 - Demostración de Reglas de Homotopía")
    print("=" * 70)
    
    # Crear problema: Coloración de un grafo triangular
    print("\n📊 Problema: Colorear un grafo triangular con 3 colores")
    print("   Nodos: A, B, C")
    print("   Aristas: A-B, B-C, A-C (triángulo)")
    print("   Colores disponibles: Rojo, Verde, Azul")
    
    engine = ArcEngineExtended(use_homotopy_rules=True)
    
    # Variables
    for node in ["A", "B", "C"]:
        engine.add_variable(node, ["Rojo", "Verde", "Azul"])
    
    # Restricciones (nodos adyacentes deben tener colores diferentes)
    def different_colors(c1, c2):
        return c1 != c2
    
    engine.add_constraint("A", "B", different_colors, "arista_AB")
    engine.add_constraint("B", "C", different_colors, "arista_BC")
    engine.add_constraint("A", "C", different_colors, "arista_AC")
    
    print(f"\n🔧 Motor creado: {engine}")
    
    # Ejecutar consistencia de arcos
    print("\n⚙️  Ejecutando consistencia de arcos con reglas de homotopía...")
    is_consistent = engine.enforce_arc_consistency()
    
    print(f"\n✅ Resultado: {'CONSISTENTE' if is_consistent else 'INCONSISTENTE'}")
    
    # Mostrar estadísticas
    print("\n📈 Estadísticas de Homotopía:")
    stats = engine.get_homotopy_statistics()
    print(f"   • Total de restricciones: {stats['total_constraints']}")
    print(f"   • Pares conmutativos: {stats['commutative_pairs']}")
    print(f"   • Grupos independientes: {stats['independent_groups']}")
    print(f"   • Grafo tiene ciclos: {'Sí' if stats['has_cycles'] else 'No'}")
    print(f"   • Densidad del grafo: {stats['graph_density']:.2f}")
    
    # Mostrar grupos independientes
    print("\n🔀 Grupos Independientes (pueden procesarse en paralelo):")
    groups = engine.get_independent_groups()
    for i, group in enumerate(groups, 1):
        print(f"   Grupo {i}: {group}")
    
    # Mostrar dominios finales
    print("\n🎨 Dominios finales (colores posibles para cada nodo):")
    for var_name in sorted(engine.variables.keys()):
        domain = engine.variables[var_name]
        colors = list(domain.get_values())
        print(f"   {var_name}: {colors}")
    
    print("\n" + "=" * 70)
    print("✨ Demostración completada exitosamente")
    print("=" * 70)

if __name__ == "__main__":
    main()
