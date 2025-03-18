from flask import Flask, render_template, request, session, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer,  models
from langchain.embeddings.base import Embeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"
 
@app.before_request
def start_timer():
    request.start_time = time.time()

@app.after_request
def log_request(response):
    latency = time.time() - request.start_time
    print(f"Request took {latency:.4f} seconds")
    return response

def load_all_txt_documnets(file_path):
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size = 400,
                                        chunk_overlap = 50)
            chunks = text_splitter.create_documents([text])

            for chunk in chunks:
                documents.append(Document(page_content=chunk.page_content,
                                          metadata = {"file_name": file_name}))
    return documents

folder_path = "C:/Users/이다연/myllm/docs_txtfiles"
all_texts = load_all_txt_documnets(folder_path)

class CustomLangchainEmbeddings(Embeddings):
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model 

    def embed_documents(self, texts):
        return self.embedding_model.encode(texts,
                                           show_progress_bar=True,
                                           batch_size=32,
                                           convert_to_tensor=False)
    
    def embed_query(self, text):
        return self.embedding_model.encode([text],
                                           show_progress_bar=True,
                                           batch_size=32,
                                           convert_to_tensor=False)[0]
    

embedding_model_path = "./models/e5/onnx"

pooling = models.Pooling(word_embedding_dimension=1024, pooling_mode_mean_tokens=True)
embedding_model = SentenceTransformer(
                    modules=[models.Transformer(embedding_model_path), pooling],
                    device = "cuda")

custom_embeddings = CustomLangchainEmbeddings(embedding_model)

vectorstore = FAISS.from_documents(
                    embedding=custom_embeddings,
                    documents = all_texts)

def faiss_search(query, top_k=3):
    return vectorstore.similarity_search(query, k=top_k)

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


class ReciprocalRankFusion:
    def __init__(self, retrievers, c=60):
        self.retrievers = retrievers
        self.c = c

    def get_relevant_documents(self, query: str, top_k: int = 5):
        results = [retriever(query, top_k = top_k) for retriever in self.retrievers]
        scores = {}
        for retriever_results in results:
            for rank, doc in enumerate(retriever_results):
                if doc.page_content not in scores:
                    scores[doc.page_content] = 0
                scores[doc.page_content] += 1 / (rank + 1 + self.c)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1],reverse=True)
        return [Document(page_content=doc[0], metadata={"score": doc[1]}) for doc in sorted_docs]   


kiwi_bm25_retriever = KiwiBM25Retriever(all_texts)

rrf_retriever = ReciprocalRankFusion(
    retrievers = [faiss_search, kiwi_bm25_retriever.get_relevant_documents])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4비트 양자화 활성화
    bnb_4bit_compute_dtype=torch.float16,  # 계산을 FP16으로 설정
    #bnb_4bit_use_double_quant=True,       # 더블 양자화 활성화
    #bnb_4bit_quant_type="nf4",            # NF4 양자화 방식 사용
)

MODEL_PATH = "./models/exaone24"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    #quantization_config=bnb_config,  # BitsAndBytesConfig 적용
    torch_dtype=torch.float16,      # bfloat16에 최적화(a100 지원)
    device_map="auto",              # GPU 자동 매핑
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


class LocalHydeQueryTransform:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def transform(self, query):
        prompt = f"""
        다음 질문에 대한 가설적 답변을 생성하세요. 답변은 간결하고 사실에 기반해야 하며, 추측이나 가정을 포함하지 마세요.
        질문 : {query}
        가설적 답변:
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
        output = self.model.generate(input_ids, max_new_tokens=150, do_sample=True, temperature=0.2, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

local_hyde_transform = LocalHydeQueryTransform(model, tokenizer)

def search_relevant_documents(query, top_k=3):
    transformed_query = local_hyde_transform.transform(query)
    print(f"[transformed_query]:{transformed_query}")
    results = rrf_retriever.get_relevant_documents(transformed_query, top_k=top_k)
    context = " ".join([doc.page_content for doc in results])
    return context

def llm_generate(query, context):
    prompt =  f"""
    제공된 참고 정보를 바탕으로 질문에 답하세요. 
    반드시 아래 참고 정보에서만 답을 생성하되 자연스러운 한국어 문장으로 완성해서 생성하고, 정보에 없는 내용은 '정보에 없습니다'라고 답하세요.
    참고 정보: {context}
    질문: {query}
    답변:"""

    input_ids = tokenizer(
        prompt,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = 300
    ).input_ids.to("cuda")

    output = model.generate(
        input_ids,
        max_new_tokens = 150,
        do_sample=False,
        pad_token_id = tokenizer.pad_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def extract_answer(output):
    if "답변:" in output:
        return output.split("답변:")[-1].strip()
    return output.strip()

# def extract_answer(output):
#     if "답변:" in output:
#         text = output.split("답변:")[-1].strip()
#     else:
#         text = output.strip()

#     sentences = text.split(". ")
#     if not sentences[-1].endswith((".", "!", "?")):
#         sentences = sentences[:-1]
#     return ". ".join(sentences).strip() 

def call_model(query):
    context = search_relevant_documents(query, top_k=3)
    response = llm_generate(query, context)
    clean_response = extract_answer(response)
    return clean_response


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
    print(session["history"])
    return render_template("index_ver222.html", history=session["history"], page_title = "ask")


if __name__ == "__main__":
    app.run(debug=False) 

