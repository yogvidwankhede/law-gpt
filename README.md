# ⚖️ LawGPT

AI-powered legal research assistant built with LangChain, Pinecone, OpenAI GPT-4o, and Flask.
Answers are grounded in three law school textbooks via a RAG (Retrieval-Augmented Generation) pipeline.

---

## 📚 Textbooks Ingested

| Textbook | Authors | Pages |
|----------|---------|-------|
| Torts: Theory and Practice (4th ed.) | Little, Lidsky & Lande | 20 |
| Copyright Law: Cases and Materials (v7.0) | Fromer & Sprigman | 729 |
| Professional Responsibility (2nd ed.) | Capra & Green | 1100 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla HTML / CSS / JS |
| Backend | Flask |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | Pinecone (index: `lawgpt`, 384-dim cosine) |
| LLM | GPT-4o via LangChain |
| Deployment | Render |

---

## 📁 Project Structure

```
law-gpt/
├── app.py                  ← Flask backend + RAG pipeline
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
├── README.md
└── templates/
    └── chat.html           ← Frontend
```

---

## ⚙️ Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yogvidwankhede/law-gpt.git
   cd law-gpt
   ```

2. **Create a `.env` file in the root**
   ```
   PINECONE_API_KEY=your_pinecone_key
   OPENAI_API_KEY=your_openai_key
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Populate the Pinecone index**
   Run `LawGPT_RAG.ipynb` locally first to embed and upsert all textbook chunks into Pinecone.
   The app connects to an existing index — it does not create one at runtime.

5. **Run the app**
   ```bash
   python app.py
   ```

6. **Visit** `http://localhost:8080`

---

## 🚀 Deployment on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service** → connect the repo
3. Render auto-detects `render.yaml` — confirm these settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Instance Type:** Starter
4. Add environment variables in the Render dashboard:
   - `PINECONE_API_KEY`
   - `OPENAI_API_KEY`
5. Click **Deploy**

---

## ⚠️ Important Notes

- **Pinecone index must be populated before deploying** — run the notebook locally first
- **1 worker only** — PyTorch + sentence-transformers is memory-heavy; multiple workers will crash on Starter plan
- **Free tier** spins down after 15 min of inactivity; upgrade to Starter ($7/mo) to keep it always-on
- **Never commit API keys** — `.env` and `data/` are gitignored

---

## 📄 Legal Disclaimer

LawGPT provides general legal information grounded in the above textbooks only — not legal advice.
Always consult a licensed attorney for your specific situation.