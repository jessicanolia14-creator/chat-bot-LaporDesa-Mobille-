from rag_core import answer_with_rag
question = input('Pertanyaan tentang aplikasi LaporDesa: ').strip()
if not question:
    raise SystemExit('Pertanyaan tidak boleh kosong.')
result = answer_with_rag(question)
print('Kategori:', result['category'])
print('Jawaban:', result['answer'])
print('Sumber:', ', '.join(result['sources']))
