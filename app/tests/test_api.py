import unittest
from app.main import app

class FlaskTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
    
    def test_predict_endpoint(self):
        response = self.app.post('/predict', json={"Age": 30, "Income": 60000})
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
