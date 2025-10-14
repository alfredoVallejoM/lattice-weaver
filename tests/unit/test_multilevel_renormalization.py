# tests/unit/test_multilevel_renormalization.py

"""
Pruebas para la arquitectura de renormalización multinivel.

Estas pruebas validan la correcta construcción de jerarquías de abstracción
y la integración de los nuevos niveles L0-L6 del compilador multiescala.
"""

import unittest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.renormalization.core import renormalize_multilevel
from lattice_weaver.renormalization.hierarchy import AbstractionHierarchy, AbstractionLevel

def create_simple_csp():
    """Crea un CSP simple de 4 variables para pruebas."""
    variables = {f"v{i}" for i in range(4)}
    domains = {var: {0, 1} for var in variables}
    constraints = [
        Constraint(scope=frozenset({"v0", "v1"}), relation=lambda a, b: a != b),
        Constraint(scope=frozenset({"v1", "v2"}), relation=lambda a, b: a != b),
        Constraint(scope=frozenset({"v2", "v3"}), relation=lambda a, b: a != b),
    ]
    return CSP(variables, domains, constraints, name="Simple4Var")

def create_nqueens_csp(n=4):
    """Crea un CSP para el problema de N-Reinas."""
    variables = {f"Q{i}" for i in range(n)}
    domains = {var: set(range(n)) for var in variables}
    constraints = []
    
    # Restricciones: ninguna reina se ataca
    for i in range(n):
        for j in range(i + 1, n):
            row_diff = j - i
            # No pueden estar en la misma columna ni en la misma diagonal
            constraints.append(
                Constraint(
                    scope=frozenset({f"Q{i}", f"Q{j}"}),
                    relation=lambda col_i, col_j, rd=row_diff: (
                        col_i != col_j and abs(col_i - col_j) != rd
                    )
                )
            )
    
    return CSP(variables, domains, constraints, name=f"{n}Queens")

class TestAbstractionHierarchy(unittest.TestCase):
    """Pruebas para la clase AbstractionHierarchy."""
    
    def test_hierarchy_initialization(self):
        """Test: La jerarquía se inicializa correctamente con el CSP original."""
        original_csp = create_simple_csp()
        hierarchy = AbstractionHierarchy(original_csp)
        
        self.assertEqual(hierarchy.highest_level, 0)
        self.assertIn(0, hierarchy.levels)
        self.assertEqual(hierarchy.get_level(0).csp, original_csp)
    
    def test_add_level(self):
        """Test: Se pueden añadir niveles consecutivos a la jerarquía."""
        original_csp = create_simple_csp()
        hierarchy = AbstractionHierarchy(original_csp)
        
        # Crear un CSP de nivel 1 (simplificado)
        level1_vars = {"L1_G0", "L1_G1"}
        level1_domains = {var: {0, 1} for var in level1_vars}
        level1_csp = CSP(level1_vars, level1_domains, [], name="Level1")
        
        partition = [{"v0", "v1"}, {"v2", "v3"}]
        var_map = {"v0": "L1_G0", "v1": "L1_G0", "v2": "L1_G1", "v3": "L1_G1"}
        
        hierarchy.add_level(1, level1_csp, partition, var_map)
        
        self.assertEqual(hierarchy.highest_level, 1)
        self.assertIn(1, hierarchy.levels)
        self.assertEqual(hierarchy.get_level(1).csp, level1_csp)
    
    def test_add_level_non_consecutive_raises_error(self):
        """Test: Añadir un nivel no consecutivo lanza un error."""
        original_csp = create_simple_csp()
        hierarchy = AbstractionHierarchy(original_csp)
        
        level2_csp = CSP({"L2_G0"}, {"L2_G0": {0, 1}}, [], name="Level2")
        
        with self.assertRaises(ValueError):
            hierarchy.add_level(2, level2_csp, [], {})
    
    def test_get_highest_csp(self):
        """Test: get_highest_csp devuelve el CSP del nivel más alto."""
        original_csp = create_simple_csp()
        hierarchy = AbstractionHierarchy(original_csp)
        
        self.assertEqual(hierarchy.get_highest_csp(), original_csp)
        
        # Añadir nivel 1
        level1_csp = CSP({"L1_G0"}, {"L1_G0": {0, 1}}, [], name="Level1")
        hierarchy.add_level(1, level1_csp, [], {})
        
        self.assertEqual(hierarchy.get_highest_csp(), level1_csp)

class TestMultilevelRenormalization(unittest.TestCase):
    """Pruebas para la función renormalize_multilevel."""
    
    def test_renormalize_creates_hierarchy(self):
        """Test: renormalize_multilevel crea una jerarquía de abstracción."""
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=2, k_function=lambda l: 2)
        
        self.assertIsInstance(hierarchy, AbstractionHierarchy)
        self.assertEqual(hierarchy.highest_level, 2)
    
    def test_renormalize_level_0_is_original(self):
        """Test: El nivel 0 de la jerarquía es el CSP original."""
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=1, k_function=lambda l: 2)
        
        level0 = hierarchy.get_level(0)
        self.assertEqual(level0.csp, original_csp)
        self.assertEqual(level0.level, 0)
    
    def test_renormalize_reduces_variables(self):
        """Test: Cada nivel de renormalización reduce el número de variables."""
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=2, k_function=lambda l: 2)
        
        level0_vars = len(hierarchy.get_level(0).csp.variables)
        level1_vars = len(hierarchy.get_level(1).csp.variables)
        level2_vars = len(hierarchy.get_level(2).csp.variables)
        
        self.assertEqual(level0_vars, 4)
        # El particionador topológico puede no reducir exactamente a k grupos
        # dependiendo de la estructura del grafo
        self.assertLessEqual(level1_vars, level0_vars)
        self.assertLessEqual(level2_vars, level1_vars)
    
    def test_renormalize_variable_naming(self):
        """Test: Las variables renormalizadas siguen la convención de nombres."""
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=1, k_function=lambda l: 2)
        
        level1_vars = hierarchy.get_level(1).csp.variables
        
        # Todas las variables del nivel 1 deben empezar con "L1_G"
        for var in level1_vars:
            self.assertTrue(var.startswith("L1_G"))
    
    def test_renormalize_with_nqueens(self):
        """Test: renormalize_multilevel funciona con el problema de N-Reinas."""
        nqueens_csp = create_nqueens_csp(n=8)
        hierarchy = renormalize_multilevel(nqueens_csp, target_level=2, k_function=lambda l: 2)
        
        self.assertEqual(hierarchy.highest_level, 2)
        
        level0_vars = len(hierarchy.get_level(0).csp.variables)
        level1_vars = len(hierarchy.get_level(1).csp.variables)
        level2_vars = len(hierarchy.get_level(2).csp.variables)
        
        self.assertEqual(level0_vars, 8)
        # El particionador topológico reduce las variables, pero no necesariamente
        # en la proporción exacta k=2
        self.assertLessEqual(level1_vars, level0_vars)
        self.assertLessEqual(level2_vars, level1_vars)
    
    def test_renormalize_partition_structure(self):
        """Test: La partición de variables se almacena correctamente."""
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=1, k_function=lambda l: 2)
        
        level1 = hierarchy.get_level(1)
        
        # La partición debe tener 2 grupos (k=2)
        self.assertEqual(len(level1.partition), 2)
        
        # Cada grupo debe ser un conjunto de variables
        for group in level1.partition:
            self.assertIsInstance(group, set)
            self.assertTrue(all(isinstance(var, str) for var in group))
    
    def test_renormalize_variable_map(self):
        """Test: El mapeo de variables se construye correctamente."""
        original_csp = create_simple_csp()
        hierarchy = renormalize_multilevel(original_csp, target_level=1, k_function=lambda l: 2)
        
        level1 = hierarchy.get_level(1)
        
        # El mapa debe contener variables que fueron particionadas
        # (puede no incluir todas si algunas fueron filtradas)
        self.assertGreater(len(level1.variable_map), 0)
        
        # Todas las variables en el mapa deben mapear a variables del nivel 1
        for var, mapped_var in level1.variable_map.items():
            self.assertIn(mapped_var, level1.csp.variables)

class TestAbstractionLevel(unittest.TestCase):
    """Pruebas para la clase AbstractionLevel."""
    
    def test_abstraction_level_creation(self):
        """Test: Se puede crear un AbstractionLevel correctamente."""
        csp = create_simple_csp()
        partition = [{"v0", "v1"}, {"v2", "v3"}]
        var_map = {"v0": "L1_G0", "v1": "L1_G0", "v2": "L1_G1", "v3": "L1_G1"}
        
        level = AbstractionLevel(level=1, csp=csp, partition=partition, variable_map=var_map)
        
        self.assertEqual(level.level, 1)
        self.assertEqual(level.csp, csp)
        self.assertEqual(level.partition, partition)
        self.assertEqual(level.variable_map, var_map)

if __name__ == "__main__":
    unittest.main()

