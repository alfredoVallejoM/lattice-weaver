import unittest
import os
import shutil
from pathlib import Path

from lattice_weaver.paging.page import Page
from lattice_weaver.paging.cache_levels import L1Cache, L2Cache, L3Cache
from lattice_weaver.paging.page_manager import PageManager

class TestL1Cache(unittest.TestCase):
    def setUp(self):
        self.cache = L1Cache(capacity=2)

    def test_put_and_get(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        page2 = Page("page2", b"data2", page_type="test_type", abstraction_level=1)
        page3 = Page("page3", b"data3", page_type="test_type", abstraction_level=1)

        self.assertIsNone(self.cache.put(page1))
        self.assertEqual(self.cache.get("page1").id, "page1")
        self.assertIsNone(self.cache.put(page2))
        self.assertEqual(self.cache.get("page2").id, "page2")
        self.assertEqual(len(self.cache), 2)

        # Test LRU eviction
        evicted = self.cache.put(page3)
        self.assertEqual(evicted.id, "page1")
        self.assertIsNone(self.cache.get("page1"))
        self.assertEqual(self.cache.get("page3").id, "page3")
        self.assertEqual(len(self.cache), 2)

    def test_remove(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        self.cache.put(page1)
        self.assertEqual(self.cache.remove("page1").id, "page1")
        self.assertIsNone(self.cache.get("page1"))
        self.assertEqual(len(self.cache), 0)
        self.assertIsNone(self.cache.remove("nonexistent"))

    def test_hit_miss_rate(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        self.cache.put(page1)
        self.cache.get("page1") # Hit
        self.cache.get("page2") # Miss
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 1)
        self.assertAlmostEqual(self.cache.get_hit_rate(), 0.5)

class TestL2Cache(unittest.TestCase):
    def setUp(self):
        self.cache = L2Cache(capacity=2)

    def test_put_and_get(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        page2 = Page("page2", b"data2", page_type="test_type", abstraction_level=1)
        page3 = Page("page3", b"data3", page_type="test_type", abstraction_level=1)

        self.assertIsNone(self.cache.put(page1))
        self.assertEqual(self.cache.get("page1").id, "page1")
        self.assertIsNone(self.cache.put(page2))
        self.assertEqual(self.cache.get("page2").id, "page2")
        self.assertEqual(len(self.cache), 2)

        # Test LRU eviction
        evicted = self.cache.put(page3)
        self.assertEqual(evicted.id, "page1")
        self.assertIsNone(self.cache.get("page1"))
        self.assertEqual(self.cache.get("page3").id, "page3")
        self.assertEqual(len(self.cache), 2)

    def test_remove(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        self.cache.put(page1)
        self.assertEqual(self.cache.remove("page1").id, "page1")
        self.assertIsNone(self.cache.get("page1"))
        self.assertEqual(len(self.cache), 0)
        self.assertIsNone(self.cache.remove("nonexistent"))

class TestL3Cache(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_page_storage")
        self.cache = L3Cache(capacity=2, storage_dir=str(self.test_dir))

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_put_and_get(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        page2 = Page("page2", b"data2", page_type="test_type", abstraction_level=1)
        page3 = Page("page3", b"data3", page_type="test_type", abstraction_level=1)

        self.assertIsNone(self.cache.put(page1))
        self.assertTrue((self.test_dir / "page1.page").exists())
        self.assertEqual(self.cache.get("page1").id, "page1")
        self.assertIsNone(self.cache.put(page2))
        self.assertTrue((self.test_dir / "page2.page").exists())
        self.assertEqual(self.cache.get("page2").id, "page2")
        self.assertEqual(len(self.cache), 2)

        # Test LRU eviction
        evicted = self.cache.put(page3)
        self.assertEqual(evicted.id, "page1")
        self.assertFalse((self.test_dir / "page1.page").exists())
        self.assertIsNone(self.cache.get("page1"))
        self.assertEqual(self.cache.get("page3").id, "page3")
        self.assertTrue((self.test_dir / "page3.page").exists())
        self.assertEqual(len(self.cache), 2)

    def test_remove(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        self.cache.put(page1)
        self.assertTrue((self.test_dir / "page1.page").exists())
        self.assertEqual(self.cache.remove("page1").id, "page1")
        self.assertFalse((self.test_dir / "page1.page").exists())
        self.assertIsNone(self.cache.get("page1"))
        self.assertEqual(len(self.cache), 0)
        self.assertIsNone(self.cache.remove("nonexistent"))

class TestPageManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_page_storage_manager")
        self.manager = PageManager(l1_capacity=1, l2_capacity=1, l3_capacity=1, l3_storage_dir=str(self.test_dir))

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_put_page_propagation(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        page2 = Page("page2", b"data2", page_type="test_type", abstraction_level=1)
        page3 = Page("page3", b"data3", page_type="test_type", abstraction_level=1)

        self.manager.put_page(page1) # L1: page1
        self.assertEqual(self.manager.l1_cache.get("page1").id, "page1")
        self.assertEqual(len(self.manager.l1_cache), 1)
        self.assertEqual(len(self.manager.l2_cache), 0)
        self.assertEqual(len(self.manager.l3_cache), 0)

        self.manager.put_page(page2) # L1: page2, L2: page1
        self.assertEqual(self.manager.l1_cache.get("page2").id, "page2")
        self.assertEqual(self.manager.l2_cache.get("page1").id, "page1")
        self.assertEqual(len(self.manager.l1_cache), 1)
        self.assertEqual(len(self.manager.l2_cache), 1)
        self.assertEqual(len(self.manager.l3_cache), 0)

        self.manager.put_page(page3) # L1: page3, L2: page2, L3: page1
        self.assertEqual(self.manager.l1_cache.get("page3").id, "page3")
        self.assertEqual(self.manager.l2_cache.get("page2").id, "page2")
        self.assertEqual(self.manager.l3_cache.get("page1").id, "page1")
        self.assertEqual(len(self.manager.l1_cache), 1)
        self.assertEqual(len(self.manager.l2_cache), 1)
        self.assertEqual(len(self.manager.l3_cache), 1)

    def test_get_page_promotion(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        page2 = Page("page2", b"data2", page_type="test_type", abstraction_level=1)
        page3 = Page("page3", b"data3", page_type="test_type", abstraction_level=1)

        self.manager.put_page(page1) # L1: page1
        self.manager.put_page(page2) # L1: page2, L2: page1
        self.manager.put_page(page3) # L1: page3, L2: page2, L3: page1

        # Get page1 from L3, should promote to L1
        retrieved_page = self.manager.get_page("page1") # L1: page1, L2: page3, L3: page2
        self.assertEqual(retrieved_page.id, "page1")
        self.assertEqual(self.manager.l1_cache.get("page1").id, "page1")
        self.assertIsNone(self.manager.l2_cache.get("page1"))
        self.assertIsNone(self.manager.l3_cache.get("page1"))

        # Check other caches after promotion
        self.assertEqual(self.manager.l2_cache.get("page3").id, "page3")
        self.assertEqual(self.manager.l3_cache.get("page2").id, "page2")
        self.assertEqual(len(self.manager.l1_cache), 1)
        self.assertEqual(len(self.manager.l2_cache), 1)
        self.assertEqual(len(self.manager.l3_cache), 1)

        # Get page2 from L3, should promote to L1
        retrieved_page = self.manager.get_page("page2") # L1: page2, L2: page1, L3: page3
        self.assertEqual(retrieved_page.id, "page2")
        self.assertEqual(self.manager.l1_cache.get("page2").id, "page2")
        self.assertIsNone(self.manager.l2_cache.get("page2"))
        self.assertIsNone(self.manager.l3_cache.get("page2"))

        # Check other caches after promotion
        self.assertEqual(self.manager.l2_cache.get("page1").id, "page1")
        self.assertEqual(self.manager.l3_cache.get("page3").id, "page3")
        self.assertEqual(len(self.manager.l1_cache), 1)
        self.assertEqual(len(self.manager.l2_cache), 1)
        self.assertEqual(len(self.manager.l3_cache), 1)

    def test_remove_page(self):
        page1 = Page("page1", b"data1", page_type="test_type", abstraction_level=1)
        self.manager.put_page(page1)
        self.manager.remove_page("page1")
        self.assertIsNone(self.manager.l1_cache.get("page1"))
        self.assertIsNone(self.manager.l2_cache.get("page1"))
        self.assertIsNone(self.manager.l3_cache.get("page1"))
        self.assertFalse((self.test_dir / "page1.page").exists())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)




from lattice_weaver.core.csp_problem import CSP, Constraint
from lattice_weaver.core.csp_engine.solver import CSPSolution

class TestCSPCaching(unittest.TestCase):
    def setUp(self):
        self.manager = PageManager(l1_capacity=2, l2_capacity=2, l3_capacity=2, l3_storage_dir="./test_csp_cache_storage")

    def tearDown(self):
        if Path("./test_csp_cache_storage").exists():
            shutil.rmtree("./test_csp_cache_storage")

    def create_simple_csp(self):
        variables = {"A", "B"}
        domains = {"A": frozenset([1, 2]), "B": frozenset([1, 2])}
        constraints = [Constraint(["A", "B"], lambda a, b: a != b)]
        return CSP(variables, domains, constraints)

    def test_cache_csp_problem(self):
        csp = self.create_simple_csp()
        csp_page = Page("csp_problem_1", csp, page_type="csp_problem", abstraction_level=1)
        self.manager.put_page(csp_page)

        retrieved_csp_page = self.manager.get_page("csp_problem_1")
        self.assertIsNotNone(retrieved_csp_page)
        self.assertEqual(retrieved_csp_page.id, "csp_problem_1")
        self.assertIsInstance(retrieved_csp_page.content, CSP)
        self.assertIn("A", retrieved_csp_page.content.variables)

    def test_cache_csp_solution(self):
        solution_data = {"A": 1, "B": 2}
        csp_solution = CSPSolution(assignment=solution_data)
        solution_page = Page("csp_solution_1", csp_solution, page_type="csp_solution", abstraction_level=2)
        self.manager.put_page(solution_page)

        retrieved_solution_page = self.manager.get_page("csp_solution_1")
        self.assertIsNotNone(retrieved_solution_page)
        self.assertEqual(retrieved_solution_page.id, "csp_solution_1")
        self.assertIsInstance(retrieved_solution_page.content, CSPSolution)
        self.assertEqual(retrieved_solution_page.content.assignment["A"], 1)

    def test_cache_csp_intermediate_state(self):
        # Simulate an intermediate state, e.g., a partial assignment or pruned domains
        intermediate_state = {"assignment": {"A": 1}, "domains": {"B": [2]}}
        state_page = Page("csp_intermediate_state_1", intermediate_state, page_type="csp_state", abstraction_level=1)
        self.manager.put_page(state_page)

        retrieved_state_page = self.manager.get_page("csp_intermediate_state_1")
        self.assertIsNotNone(retrieved_state_page)
        self.assertEqual(retrieved_state_page.id, "csp_intermediate_state_1")
        self.assertIsInstance(retrieved_state_page.content, dict)
        self.assertEqual(retrieved_state_page.content["assignment"]["A"], 1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

