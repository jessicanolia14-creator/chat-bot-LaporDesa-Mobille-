import os
from flask import Flask, jsonify, request
from config import QUICK_QUESTIONS
from rag_core import answer_with_rag

app = Flask(__name__)
app.json.ensure_ascii = False

@app.get('/api/health')
def health():
    return jsonify({'status': 'ok', 'service': 'LaporDesa Chatbot RAG API'})

@app.get('/api/quick-questions')
def quick_questions():
    return jsonify({'questions': QUICK_QUESTIONS})

@app.post('/api/chat')
def chat():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Kirim data JSON yang berisi question.'}), 400
    question = data.get('question')
    if not isinstance(question, str) or not question.strip():
        return jsonify({'error': 'Pertanyaan tidak boleh kosong.'}), 400
    question = question.strip()
    if len(question) > 500:
        return jsonify({'error': 'Pertanyaan maksimal 500 karakter.'}), 400
    try:
        return jsonify(answer_with_rag(question))
    except Exception:
        app.logger.exception('Proses chat gagal')
        return jsonify({'error': 'Maaf, layanan chatbot sedang bermasalah. Silakan coba beberapa saat lagi, ya.'}), 503

if __name__ == '__main__':
    app.run(host=os.getenv('FLASK_HOST', '0.0.0.0'),
            port=int(os.getenv('FLASK_PORT', '5000')), debug=False)
