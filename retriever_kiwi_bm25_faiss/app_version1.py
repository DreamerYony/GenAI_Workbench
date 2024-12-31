from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer, models
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time


app = Flask(__name__)
app.secret_key = "supersecretkey"

@app.before_request
def start_timer():
  request.start_time = time.time()

@app.after_request
def log_request(response):
  latency = time.time() - request.start_time
  print(f"Request took {latency:.4f} seconds")

def load_all_txt_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap)
            chunks = text_splitter.create_documents([text])
            for chunk in chunks:
                documents.append(Document(page_content=chunk.page_content, 
                                          metadata={"source": filename}))
    return documents

folder_path = "./docs_txtfiles"
all_documents =  load_all_txt_documents(folder_path)

class CustomLangchainEmbeddings(Embeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def embed_documents(self, texts):
        return self.embedding_model.encode(
            texts, show_progress_bar=True, batch_size=32, convert_to_tensor=False
        )

    def embed_query(self, text):
        return self.embedding_model.encode(
            [text], show_progress_bar=False, batch_size=1, convert_to_tensor=False
        )[0]

embedding_model_path = "./models/e5/onnx"  # "intfloat/multilingual-e5-large-instruct"
pooling = models.Pooling(word_embedding_dimension=1024, pooling_mode_mean_tokens=True)
embedding_model = SentenceTransformer(
    modules=[models.Transformer(embedding_model_path), pooling],
    device="cuda"  
)

custom_embeddings = CustomLangchainEmbeddings(embedding_model)

vectorstore = FAISS.from_documents(
    documents=all_documents,  
    embedding=custom_embeddings 
)

class KiwiBM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [doc.page_content for doc in documents]
        self.kiwi = Kiwi() 
        self.tokenized_corpus = [self.tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def tokenize(self, text):
        return [token.form for token in self.kiwi.tokenize(text)]

    def get_relevant_documents(self, query, top_k=3):
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in top_indices]

class EnsembleRetriever:
    def __init__(self, retrievers, weights):
        self.retrievers = retrievers  # 검색기 리스트
        self.weights = weights  # 각 검색기의 가중치

    def get_relevant_documents(self, query: str, top_k: int = 3):
        combined_results = []

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever(query)
            for doc in results:
                combined_results.append((doc, weight))

        doc_scores = {}
        for doc, weight in combined_results:
            if doc.page_content in doc_scores:
                doc_scores[doc.page_content] += weight
            else:
                doc_scores[doc.page_content] = weight

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [Document(page_content=doc[0], metadata={"file_name":"Unknown"}) for doc in sorted_docs]

kiwi_bm25_retriever = KiwiBM25Retriever(all_documents)

def faiss_search(query, top_k=3):
    return vectorstore.similarity_search(query, k=top_k)

ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_search, kiwi_bm25_retriever.get_relevant_documents],
    weights=[0.7, 0.3] 
)

MODEL_PATH = "./models/exaone24"  # "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_compute_dtype=torch.float16,  
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    # quantization_config=bnb_config,  # 양자화 설정 (옵션)
    device_map="auto", 
    torch_dtype="auto",
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def search_relevant_documents(query, top_k=3):
    results = ensemble_retriever.get_relevant_documents(query, top_k=top_k)
    context = " ".join([doc.page_content for doc in results])
    return context

def llm_generate(query, context):
    prompt = f"""
    아래 참고 정보에 기반하여 질문에 답변하세요.
    참고 정보에서 답을 찾을 수 없는 경우 '정보에 없습니다'라고 답변하세요.
    참고 정보: {context}
    질문: {query}
    답변:"""

    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).input_ids.to("cuda")

    output = model.generate(
        input_ids,
        max_new_tokens=150,  
        do_sample=False,    
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def extract_answer(output):
    if "답변:" in output:
        text = output.split("답변:")[-1].strip()
    else:
        text = output.strip()

    sentences = text.split(". ")
    if not sentences[-1].endswith((".", "!", "?")):
        sentences = sentences[:-1] 

    return ". ".join(sentences).strip()

@app.route("/")
def index_ver2():
    if "history" not in session:
    session["history"] = []
    return render_template("index_ver222.html", history=session["history"])

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    response = call_model(user_input)
    session["history"].append({"user": user_input, "ai": response})
    return render_template("index_ver222.html", history=session["history"])

def call_model(query):
    context = search_relevant_documents(query, top_k=3)
    response = llm_generaate(query, context)
    clean_response = extract_answer(response)
    return clean_response

if __name__ == "__main__":
    app.run(debug=False)

