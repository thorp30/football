import unittest
from scraping_football_data import get_team_urls

class TestScrapingFootballData(unittest.TestCase):
    def test_get_team_urls(self):
        # Test case 1: Check if the function returns a list
        self.assertIsInstance(get_team_urls(), list)

        # Test case 2: Check if the returned list is not empty
        self.assertTrue(len(get_team_urls()) > 0)

        # Test case 3: Check if all URLs in the list start with "https://fbref.com"
        for url in get_team_urls():
            self.assertTrue(url.startswith("https://fbref.com"))

        # Test case 4: Check if all URLs in the list contain "/squads/"
        for url in get_team_urls():
            self.assertIn("/squads/", url)

if __name__ == '__main__':
    unittest.main()import unittest
from scraping_football_data import get_team_urls

class TestScrapingFootballData(unittest.TestCase):
    def test_get_team_urls(self):
        # Test case 1: Check if the function returns a list
        self.assertIsInstance(get_team_urls(), list)

        # Test case 2: Check if the function returns non-empty URLs
        self.assertTrue(len(get_team_urls()) > 0)

        # Test case 3: Check if the function returns valid URLs
        for url in get_team_urls():
            self.assertTrue(url.startswith("https://fbref.com/squads/"))

if __name__ == '__main__':
    unittest.main()