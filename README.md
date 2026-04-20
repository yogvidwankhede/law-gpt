# ⚖️ LawGPT

AI-powered legal research assistant built with LangChain, Pinecone, OpenAI GPT-4o, and Flask. Answers are grounded in three law school textbooks via a RAG pipeline.

## Textbooks ingested

| File | Subject | Pages |
|------|---------|-------|
| Torts: Theory and Practice (4th ed.) | Little, Lidsky & Lande | 20 |
| Copyright Law: Cases and Materials (v7.0) | Fromer & Sprigman | 729 |
| Professional Responsibility (2nd ed.) | Capra & Green | 1100 |

## Tech stack

- **Frontend** — Vanilla HTML/CSS/JS
- **Backend** — Flask
- **Embeddings** — sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- **Vector store** — Pinecone (index: lawgpt, 384-dim cosine)
- **LLM** — GPT-4o via LangChain
- **Deployment** — Render

## Project structure
law-gpt/
├── app.py                  ← Flask backend + RAG pipeline
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
├── README.md
└── templates/
└── chat.html           ← Frontend

## Local setup

1. Clone the repo
2. Create a `.env` file in the root:
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Make sure the Pinecone index `lawgpt` is already populated
   (run `LawGPT_RAG.ipynb` locally first)
5. Run the app:
```bash
   python app.py
```
6. Visit `http://localhost:8080`

## Deployment on Render

1. Push this repo to GitHub
2. Go to render.com → New → Web Service → connect the repo
3. Set environment variables in Render dashboard:
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
4. Render auto-detects `render.yaml` — click Deploy

## Important notes

- Pinecone index must be populated **before** deploying
- Use `--workers 1` in gunicorn — PyTorch is memory-heavy, multiple workers crash on Starter plan
- Free tier spins down after 15 min idle; Starter plan keeps it always-on
- `.env` and `data/` are gitignored — never commit API keys