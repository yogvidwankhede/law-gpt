from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone

# ── Load environment ──────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Build RAG pipeline (runs once at startup) ─────────────────────────────
print("[LawGPT] Loading embeddings model...")
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("[LawGPT] Connecting to Pinecone index 'lawgpt'...")
pc = Pinecone(api_key=PINECONE_API_KEY)
docsearch = PineconeVectorStore.from_existing_index(
    index_name="lawgpt",
    embedding=embedding,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

print("[LawGPT] Initialising LLM...")
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

LEGAL_SYSTEM_PROMPT = (
    "You are LawGPT, an expert AI legal research assistant. "
    "You have been given excerpts from the following law school textbooks as context:\n"
    "  • Torts: Theory and Practice (4th ed.) — Little, Lidsky & Lande\n"
    "  • Copyright Law: Cases and Materials (v7.0) — Fromer & Sprigman\n"
    "  • Professional Responsibility: A Contemporary Approach (2nd ed.) — Capra & Green\n\n"
    "Rules:\n"
    "1. Base your answer STRICTLY on the provided context. Do not fabricate cases or statutes.\n"
    "2. Cite the source textbook and any case names mentioned in the context.\n"
    "3. Use proper legal terminology (elements, holdings, dicta, majority/minority rule, etc.).\n"
    "4. When a rule varies by jurisdiction, say so explicitly.\n"
    "5. If the context does not contain enough information, say: "
    "   'The provided materials do not cover this topic in sufficient detail.'\n"
    "6. End every substantive answer with: "
    "   '⚠️ This is general legal information, not legal advice. "
    "Consult a licensed attorney for your specific situation.'\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", LEGAL_SYSTEM_PROMPT),
    ("human",  "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("[LawGPT] RAG pipeline ready.")

# ── Flask app ─────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form.get("msg", "").strip()

    if not user_msg:
        return jsonify({"answer": "Please enter a question.", "sources": []}), 400

    capability_queries = {
        "what can you do", "what all can you do", "what do you do",
        "what are your capabilities", "what are you capable of",
        "how can you help", "tell me what you can do",
        "what is your purpose", "what's your purpose",
    }
    if user_msg.lower() in capability_queries:
        return jsonify({
            "answer": (
                "I can help you research legal topics grounded in three law school "
                "textbooks: Torts (negligence, strict liability, intentional torts, "
                "defamation), Copyright Law (fair use, originality, infringement), "
                "and Professional Responsibility (attorney ethics, confidentiality, "
                "conflicts of interest). Ask me anything covered in those materials."
            ),
            "sources": [],
        })

    try:
        response = rag_chain.invoke({"input": user_msg})
        answer = response.get("answer", "No answer returned.")

        sources = list({
            doc.metadata.get("title", "")
            for doc in response.get("context", [])
            if doc.metadata.get("title")
        })

        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        print(f"[LawGPT] Error: {e}")
        return jsonify({
            "answer": "An error occurred while processing your request. Please try again.",
            "sources": [],
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
