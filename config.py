import os
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')

OUT_OF_SCOPE = (
    'Maaf ya, saya khusus membantu pertanyaan seputar penggunaan aplikasi LaporDesa. 😊\n\n'
    'Kamu bisa bertanya tentang cara login, membuat laporan, melihat riwayat, '
    'mengatur profil, atau menggunakan fitur lainnya di LaporDesa.'
)
NOT_FOUND = (
    'Maaf, saya belum menemukan informasi tersebut dalam panduan LaporDesa.\n\n'
    'Supaya tidak memberikan informasi yang keliru, silakan periksa menu Bantuan '
    'atau hubungi kontak bantuan yang tersedia di aplikasi, ya.'
)
QUICK_QUESTIONS = [
    'Bagaimana cara membuat laporan?',
    'Bagaimana cara menentukan lokasi laporan?',
    'Bagaimana cara menambahkan foto laporan?',
    'Bagaimana cara melihat riwayat laporan?',
    'Bagaimana cara mengubah data profil akun?',
]

def get_client():
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key or key.startswith('ISI_API_KEY'):
        raise RuntimeError('Isi GEMINI_API_KEY pada file .env backend terlebih dahulu.')
    from google import genai
    return genai.Client(api_key=key)
