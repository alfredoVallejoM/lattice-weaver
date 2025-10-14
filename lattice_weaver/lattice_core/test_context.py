import unittest
from lattice_weaver.lattice_core.context import FormalContext

class TestFormalContext(unittest.TestCase):

    def setUp(self):
        self.context = FormalContext()

    def test_add_object(self):
        self.context.add_object("obj1")
        self.assertIn("obj1", self.context.objects)
        self.assertEqual(self.context.get_object_index("obj1"), 0)

    def test_add_attribute(self):
        self.context.add_attribute("attr1")
        self.assertIn("attr1", self.context.attributes)

    def test_add_incidence(self):
        self.context.add_object("obj1")
        self.context.add_attribute("attr1")
        self.context.add_incidence("obj1", "attr1")
        self.assertIn(("obj1", "attr1"), self.context.incidences)
        self.assertIn("attr1", self.context.prime_objects({"obj1"}))
        self.assertIn("obj1", self.context.prime_attributes({"attr1"}))

    def test_prime_objects(self):
        self.context.add_object("obj1")
        self.context.add_object("obj2")
        self.context.add_attribute("attr1")
        self.context.add_attribute("attr2")
        self.context.add_incidence("obj1", "attr1")
        self.context.add_incidence("obj1", "attr2")
        self.context.add_incidence("obj2", "attr1")
        
        self.assertEqual(self.context.prime_objects({"obj1", "obj2"}), frozenset({"attr1"}))
        self.assertEqual(self.context.prime_objects({"obj1"}), frozenset({"attr1", "attr2"}))

    def test_prime_attributes(self):
        self.context.add_object("obj1")
        self.context.add_object("obj2")
        self.context.add_attribute("attr1")
        self.context.add_attribute("attr2")
        self.context.add_incidence("obj1", "attr1")
        self.context.add_incidence("obj1", "attr2")
        self.context.add_incidence("obj2", "attr1")

        self.assertEqual(self.context.prime_attributes({"attr1"}), frozenset({"obj1", "obj2"}))
        self.assertEqual(self.context.prime_attributes({"attr2"}), frozenset({"obj1"}))

if __name__ == '__main__':
    unittest.main()

