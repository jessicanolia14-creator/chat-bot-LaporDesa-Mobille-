import unittest
from unittest.mock import patch
from api_app import app
from config import OUT_OF_SCOPE

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        self.assertEqual(self.client.get('/api/health').json['status'], 'ok')

    def test_five_questions(self):
        self.assertEqual(len(self.client.get('/api/quick-questions').json['questions']), 5)

    def test_empty_input(self):
        self.assertEqual(self.client.post('/api/chat', json={'question': ''}).status_code, 400)

    def test_long_input(self):
        self.assertEqual(self.client.post('/api/chat', json={'question': 'a'*501}).status_code, 400)

    @patch('rag_core.is_lapordesa_question', return_value=False)
    def test_out_of_scope_template(self, _):
        from rag_core import answer_with_rag
        result = answer_with_rag('Apa resep rendang?')
        self.assertEqual(result['answer'], OUT_OF_SCOPE)
        self.assertEqual(result['category'], 'out_of_scope')

if __name__ == '__main__':
    unittest.main()
