
import os
from config import get_client

# Membuat koneksi Gemini dan menjaganya tetap aktif
# selama permintaan diproses.
with get_client() as client:
    response = client.models.generate_content(
        model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'),
        contents='Jawab singkat: halo!'
    )

print('Jawaban Gemini:')
print(response.text or 'Tidak ada jawaban dari Gemini.')
