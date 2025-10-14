import unittest
from lattice_weaver.lattice_core.context import FormalContext
from lattice_weaver.lattice_core.builder import LatticeBuilder

class TestLatticeBuilder(unittest.TestCase):

    def setUp(self):
        self.context = FormalContext()
        self.context.add_object("1")
        self.context.add_object("2")
        self.context.add_object("3")
        self.context.add_attribute("a")
        self.context.add_attribute("b")
        self.context.add_attribute("c")
        self.context.add_incidence("1", "a")
        self.context.add_incidence("2", "b")
        self.context.add_incidence("3", "c")

    def test_cbo_simple_context(self):
        builder = LatticeBuilder(self.context)
        concepts = builder.build_lattice()
        
        # Expected concepts for a simple context with 3 objects and 3 attributes
        # where each object has one unique attribute.
        expected_concepts = [
            (frozenset(), frozenset({"a", "b", "c"})),
            (frozenset({"1"}), frozenset({"a"})),
            (frozenset({"2"}), frozenset({"b"})),
            (frozenset({"3"}), frozenset({"c"})),
            (frozenset({"1", "2"}), frozenset()),
            (frozenset({"1", "3"}), frozenset()),
            (frozenset({"2", "3"}), frozenset()),
            (frozenset({"1", "2", "3"}), frozenset()),
        ]
        
        self.assertEqual(len(concepts), len(expected_concepts))
        for concept in expected_concepts:
            self.assertIn(concept, concepts)

if __name__ == '__main__':
    unittest.main()

