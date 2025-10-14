import unittest
import shutil
import os

from lattice_weaver.paging.page_manager import PageManager
from lattice_weaver.paging.page import Page

class TestPageManagerLogic(unittest.TestCase):

    def setUp(self):
        self.storage_dir = "./test_pm_storage"
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
        self.pm = PageManager(l3_storage_dir=self.storage_dir)

    def tearDown(self):
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def test_put_and_get_simple(self):
        """Test basic put and get in PageManager."""
        page_id = "simple_page"
        original_data = {"a": 1, "b": 2}
        page = Page(page_id, original_data, page_type="test", abstraction_level=1)
        
        self.pm.put_page(page)
        
        retrieved_page = self.pm.get_page(page_id)
        
        self.assertIsNotNone(retrieved_page)
        self.assertEqual(retrieved_page.id, page_id)
        self.assertEqual(retrieved_page.content, original_data)

if __name__ == '__main__':
    unittest.main()
