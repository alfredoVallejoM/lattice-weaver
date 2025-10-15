import pytest
from lattice_weaver.fibration import (
    ConstraintHierarchy,
    ConstraintLevel,
    Hardness,
    SimpleMultiscaleCompiler
)

class TestSimpleMultiscaleCompiler:
    """Tests para la clase SimpleMultiscaleCompiler."""

    def test_compile_decompile_simple_problem(self):
        """Test: Compilar y descompilar un problema simple."""
        # 1. Definir el problema original
        original_hierarchy = ConstraintHierarchy()
        original_domains = {
            'x': [1, 2],
            'y': [1, 2],
            'z': [1, 2]
        }
        original_hierarchy.add_local_constraint('x', 'y', lambda a: a['x'] != a['y'], hardness=Hardness.HARD)
        original_hierarchy.add_local_constraint('y', 'z', lambda a: a['y'] != a['z'], hardness=Hardness.HARD)
        original_hierarchy.add_unary_constraint('x', lambda a: a['x'] == 1, weight=1.0, hardness=Hardness.SOFT)

        # 2. Compilar el problema
        compiler = SimpleMultiscaleCompiler()
        optimized_hierarchy, optimized_domains, metadata = compiler.compile_problem(original_hierarchy, original_domains)

        # 3. Verificar la compilación
        # x, y, z deberían estar en un solo grupo
        assert len(optimized_domains) == 1
        group_name = list(optimized_domains.keys())[0]
        assert group_name.startswith("Group_")
        assert set(metadata["group_definitions"][group_name]) == {'x', 'y', 'z'}

        # El dominio del grupo debe contener las asignaciones consistentes
        # Soluciones consistentes para x!=y, y!=z con x,y,z en [1,2]:
        # (x=1, y=2, z=1)
        # (x=2, y=1, z=2)
        expected_group_domain = [
            {'x': 1, 'y': 2, 'z': 1},
            {'x': 2, 'y': 1, 'z': 2}
        ]
        assert len(optimized_domains[group_name]) == 2
        for assignment in expected_group_domain:
            assert assignment in optimized_domains[group_name]

        # La jerarquía optimizada debe tener una restricción (la soft)
        assert len(optimized_hierarchy.get_all_constraints()) > 0

        # 4. Simular una solución del problema optimizado
        # Supongamos que el solver elige la primera asignación consistente, que tiene x=1
        # y por lo tanto satisface la restricción soft (energía 0).
        optimized_solution = {group_name: {'x': 1, 'y': 2, 'z': 1}}

        # 5. Descompilar la solución
        decompiled_solution = compiler.decompile_solution(optimized_solution, metadata)

        # 6. Verificar la descompilación
        assert decompiled_solution == {'x': 1, 'y': 2, 'z': 1}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

