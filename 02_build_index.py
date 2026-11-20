import json
import os
import numpy as np
from config import BASE, get_client
from google.genai import types

src = BASE / 'knowledge' / 'panduan_lapordesa.txt'
chunks = [p.strip() for p in src.read_text(encoding='utf-8').split('\n\n') if p.strip()]
if not chunks:
    raise SystemExit('Dokumen panduan kosong.')
client = get_client()
model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
embeddings = []
for number, chunk in enumerate(chunks, 1):
    result = client.models.embed_content(
        model=model, contents=chunk,
        config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT'),
    )
    embeddings.append(result.embeddings[0].values)
    print(f'Embedding {number}/{len(chunks)} selesai.')
folder = BASE / 'index'
folder.mkdir(exist_ok=True)
np.save(folder / 'embeddings.npy', np.asarray(embeddings, dtype=np.float32))
(folder / 'chunks.json').write_text(json.dumps({
    'embedding_model': model, 'source': src.name,
    'chunks': [{'id': i, 'text': c} for i, c in enumerate(chunks, 1)]
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Index RAG tersimpan: {len(chunks)} chunks.')
