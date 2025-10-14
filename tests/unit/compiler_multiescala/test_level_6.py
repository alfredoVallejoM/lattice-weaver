"""
Pruebas unitarias para el Nivel L6 (Problema Completo)
"""

import unittest
from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.compiler_multiescala import (
    Level0, Level1, Level2, Level3, Level4, Level5, Level6,
    ProblemDescription
)


class TestProblemDescription(unittest.TestCase):
    """Pruebas para ProblemDescription"""
    
    def test_init(self):
        """Prueba la inicialización de ProblemDescription"""
        problem = ProblemDescription(
            name="Test Problem",
            description="A test CSP problem",
            domain="Testing"
        )
        
        self.assertEqual(problem.name, "Test Problem")
        self.assertEqual(problem.description, "A test CSP problem")
        self.assertEqual(problem.domain, "Testing")
        self.assertEqual(problem.properties, {})
    
    def test_equality(self):
        """Prueba la igualdad de ProblemDescription"""
        problem1 = ProblemDescription(
            name="Test Problem",
            description="A test CSP problem",
            domain="Testing"
        )
        problem2 = ProblemDescription(
            name="Test Problem",
            description="A test CSP problem",
            domain="Testing"
        )
        
        self.assertEqual(hash(problem1), hash(problem2))


class TestLevel6Initialization(unittest.TestCase):
    """Pruebas para la inicialización de Level6"""
    
    def test_init_with_problem(self):
        """Prueba la inicialización de Level6 con un problema"""
        problem = ProblemDescription(
            name="Test Problem",
            description="A test CSP problem",
            domain="Testing"
        )
        l6 = Level6(problem=problem)
        
        self.assertEqual(l6.problem, problem)
        self.assertIsNone(l6.level5)


class TestLevel6BuildFromLower(unittest.TestCase):
    """Pruebas para la construcción de Level6 desde Level5"""
    
    def test_build_from_l5_simple(self):
        """Prueba la construcción de L6 desde L5"""
        # Crear un CSP simple
        csp = CSP(
            variables={"v0", "v1"},
            domains={"v0": frozenset([0, 1]), "v1": frozenset([0, 1])},
            constraints=[
                Constraint(scope=frozenset(["v0", "v1"]), relation=lambda v0, v1: v0 != v1)
            ]
        )
        
        # Construir L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
        l0 = Level0(csp)
        l1 = Level1([], [], config={'original_domains': l0.csp.domains})
        l1.build_from_lower(l0)
        l2 = Level2([], [], [], config={'original_domains': l0.csp.domains})
        l2.build_from_lower(l1)
        l3 = Level3([], [], [], [], config={'original_domains': l0.csp.domains})
        l3.build_from_lower(l2)
        l4 = Level4([], [], [], config={'original_domains': l0.csp.domains})
        l4.build_from_lower(l3)
        l5 = Level5([], [], [], config={'original_domains': l0.csp.domains})
        l5.build_from_lower(l4)
        l6 = Level6(
            problem=ProblemDescription(
                name="Simple CSP",
                description="A simple CSP with 2 variables",
                domain="Testing"
            )
        )
        l6.build_from_lower(
            l5,
            problem_name="Simple CSP",
            problem_description="A simple CSP with 2 variables",
            problem_domain="Testing"
        )
        
        self.assertIsNotNone(l6)
        self.assertEqual(l6.problem.name, "Simple CSP")
        self.assertEqual(l6.problem.description, "A simple CSP with 2 variables")
        self.assertEqual(l6.problem.domain, "Testing")
        self.assertIsNotNone(l6.level5)


class TestLevel6RefineToLower(unittest.TestCase):
    """Pruebas para el refinamiento de Level6 a Level5"""
    
    def test_refine_to_l5_simple(self):
        """Prueba el refinamiento de L6 a L5"""
        # Crear un CSP simple
        csp = CSP(
            variables={"v0", "v1"},
            domains={"v0": frozenset([0, 1]), "v1": frozenset([0, 1])},
            constraints=[
                Constraint(scope=frozenset(["v0", "v1"]), relation=lambda v0, v1: v0 != v1)
            ]
        )
        
        # Construir L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
        l0 = Level0(csp)
        l1 = Level1([], [], config={'original_domains': l0.csp.domains})
        l1.build_from_lower(l0)
        l2 = Level2([], [], [], config={'original_domains': l0.csp.domains})
        l2.build_from_lower(l1)
        l3 = Level3([], [], [], [], config={'original_domains': l0.csp.domains})
        l3.build_from_lower(l2)
        l4 = Level4([], [], [], config={'original_domains': l0.csp.domains})
        l4.build_from_lower(l3)
        l5 = Level5([], [], [], config={'original_domains': l0.csp.domains})
        l5.build_from_lower(l4)
        l6 = Level6(
            problem=ProblemDescription(
                name="CSP Problem",
                description="Generic CSP Problem",
                domain="General"
            )
        )
        l6.build_from_lower(l5)
        
        # Refinar L6 -> L5
        l5_refined = l6.refine_to_lower()
        
        self.assertIsNotNone(l5_refined)
        self.assertEqual(l5_refined.meta_patterns, l5.meta_patterns)
        self.assertEqual(l5_refined.isolated_concepts, l5.isolated_concepts)


class TestLevel6Validation(unittest.TestCase):
    """Pruebas para la validación de Level6"""
    
    def test_validate_valid_l6(self):
        """Prueba la validación de un L6 válido"""
        # Crear un CSP simple
        csp = CSP(
            variables={"v0", "v1"},
            domains={"v0": frozenset([0, 1]), "v1": frozenset([0, 1])},
            constraints=[
                Constraint(scope=frozenset(["v0", "v1"]), relation=lambda v0, v1: v0 != v1)
            ]
        )
        
        # Construir L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
        l0 = Level0(csp)
        l1 = Level1([], [], config={'original_domains': l0.csp.domains})
        l1.build_from_lower(l0)
        l2 = Level2([], [], [], config={'original_domains': l0.csp.domains})
        l2.build_from_lower(l1)
        l3 = Level3([], [], [], [], config={'original_domains': l0.csp.domains})
        l3.build_from_lower(l2)
        l4 = Level4([], [], [], config={'original_domains': l0.csp.domains})
        l4.build_from_lower(l3)
        l5 = Level5([], [], [], config={'original_domains': l0.csp.domains})
        l5.build_from_lower(l4)
        l6 = Level6(
            problem=ProblemDescription(
                name="CSP Problem",
                description="Generic CSP Problem",
                domain="General"
            )
        )
        l6.build_from_lower(l5)
        
        self.assertTrue(l6.validate())
    
    def test_validate_problem_without_name(self):
        """Prueba la validación de un problema sin nombre"""
        problem = ProblemDescription(
            name="",
            description="A test CSP problem",
            domain="Testing"
        )
        l6 = Level6(problem=problem)
        
        self.assertFalse(l6.validate())


class TestLevel6Complexity(unittest.TestCase):
    """Pruebas para el cálculo de complejidad de Level6"""
    
    def test_complexity_with_l5(self):
        """Prueba el cálculo de complejidad de L6 con L5"""
        # Crear un CSP simple
        csp = CSP(
            variables={"v0", "v1"},
            domains={"v0": frozenset([0, 1]), "v1": frozenset([0, 1])},
            constraints=[
                Constraint(scope=frozenset(["v0", "v1"]), relation=lambda v0, v1: v0 != v1)
            ]
        )
        
        # Construir L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
        l0 = Level0(csp)
        l1 = Level1([], [], config={'original_domains': l0.csp.domains})
        l1.build_from_lower(l0)
        l2 = Level2([], [], [], config={'original_domains': l0.csp.domains})
        l2.build_from_lower(l1)
        l3 = Level3([], [], [], [], config={'original_domains': l0.csp.domains})
        l3.build_from_lower(l2)
        l4 = Level4([], [], [], config={'original_domains': l0.csp.domains})
        l4.build_from_lower(l3)
        l5 = Level5([], [], [], config={'original_domains': l0.csp.domains})
        l5.build_from_lower(l4)
        l6 = Level6(
            problem=ProblemDescription(
                name="CSP Problem",
                description="Generic CSP Problem",
                domain="General"
            )
        )
        l6.build_from_lower(l5)
        
        self.assertEqual(l6.complexity, l5.complexity)
    
    def test_complexity_empty_l6(self):
        """Prueba el cálculo de complejidad de un L6 vacío"""
        problem = ProblemDescription(
            name="Empty Problem",
            description="An empty CSP problem",
            domain="Testing"
        )
        l6 = Level6(problem=problem)
        
        self.assertEqual(l6.complexity, 0.0)


class TestLevel6Statistics(unittest.TestCase):
    """Pruebas para las estadísticas de Level6"""
    
    def test_get_statistics_with_l5(self):
        """Prueba las estadísticas de L6 con L5"""
        # Crear un CSP simple
        csp = CSP(
            variables={"v0", "v1"},
            domains={"v0": frozenset([0, 1]), "v1": frozenset([0, 1])},
            constraints=[
                Constraint(scope=frozenset(["v0", "v1"]), relation=lambda v0, v1: v0 != v1)
            ]
        )
        
        # Construir L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
        l0 = Level0(csp)
        l1 = Level1([], [], config={'original_domains': l0.csp.domains})
        l1.build_from_lower(l0)
        l2 = Level2([], [], [], config={'original_domains': l0.csp.domains})
        l2.build_from_lower(l1)
        l3 = Level3([], [], [], [], config={'original_domains': l0.csp.domains})
        l3.build_from_lower(l2)
        l4 = Level4([], [], [], config={'original_domains': l0.csp.domains})
        l4.build_from_lower(l3)
        l5 = Level5([], [], [], config={'original_domains': l0.csp.domains})
        l5.build_from_lower(l4)
        l6 = Level6(
            problem=ProblemDescription(
                name="Simple CSP",
                description="A simple CSP with 2 variables",
                domain="Testing"
            )
        )
        l6.build_from_lower(
            l5,
            problem_name="Simple CSP",
            problem_description="A simple CSP with 2 variables",
            problem_domain="Testing"
        )
        
        stats = l6.get_statistics()
        
        self.assertEqual(stats['problem_name'], "Simple CSP")
        self.assertEqual(stats['problem_description'], "A simple CSP with 2 variables")
        self.assertEqual(stats['problem_domain'], "Testing")
        self.assertIn('level5_stats', stats)


class TestLevel6EdgeCases(unittest.TestCase):
    """Pruebas para casos extremos de Level6"""
    
    def test_empty_l6(self):
        """Prueba un L6 vacío"""
        problem = ProblemDescription(
            name="Empty Problem",
            description="An empty CSP problem",
            domain="Testing"
        )
        l6 = Level6(problem=problem)
        
        self.assertIsNone(l6.level5)
        self.assertEqual(l6.complexity, 0.0)


class TestLevel6Integration(unittest.TestCase):
    """Pruebas de integración completa de Level6"""
    
    def test_full_integration_l0_l1_l2_l3_l4_l5_l6(self):
        """Prueba la integración completa L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L5 -> L4 -> L3 -> L2 -> L1 -> L0"""
        # Crear un CSP simple
        csp = CSP(
            variables={"v0", "v1", "v2"},
            domains={"v0": frozenset([0, 1]), "v1": frozenset([0, 1]), "v2": frozenset([0, 1])},
            constraints=[
                Constraint(scope=frozenset(["v0", "v1"]), relation=lambda v0, v1: v0 != v1),
                Constraint(scope=frozenset(["v1", "v2"]), relation=lambda v1, v2: v1 != v2)
            ]
        )
        
        # Construir L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6
        l0 = Level0(csp)
        l1 = Level1([], [], config={'original_domains': l0.csp.domains})
        l1.build_from_lower(l0)
        l2 = Level2([], [], [], config={'original_domains': l0.csp.domains})
        l2.build_from_lower(l1)
        l3 = Level3([], [], [], [], config={'original_domains': l0.csp.domains})
        l3.build_from_lower(l2)
        l4 = Level4([], [], [], config={'original_domains': l0.csp.domains})
        l4.build_from_lower(l3)
        l5 = Level5([], [], [], config={'original_domains': l0.csp.domains})
        l5.build_from_lower(l4)
        l6 = Level6(
            problem=ProblemDescription(
                name="Integration Test CSP",
                description="A CSP for integration testing",
                domain="Testing"
            )
        )
        l6.build_from_lower(
            l5,
            problem_name="Integration Test CSP",
            problem_description="A CSP for integration testing",
            problem_domain="Testing"
        )
        
        # Refinar L6 -> L5 -> L4 -> L3 -> L2 -> L1 -> L0
        l5_refined = l6.refine_to_lower()
        l4_refined = l5_refined.refine_to_lower()
        l3_refined = l4_refined.refine_to_lower()
        l2_refined = l3_refined.refine_to_lower()
        l1_refined = l2_refined.refine_to_lower()
        l0_refined = l1_refined.refine_to_lower()
        
        # Verificar que el CSP original se preserva
        self.assertEqual(set(l0_refined.csp.variables), set(csp.variables))
        self.assertEqual(len(l0_refined.csp.constraints), len(csp.constraints))


if __name__ == '__main__':
    unittest.main()
