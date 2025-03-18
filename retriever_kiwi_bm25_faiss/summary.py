#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, session, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
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
    return response


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4비트 양자화 활성화
    bnb_4bit_compute_dtype=torch.float16,  # 계산을 FP16으로 설정
    #bnb_4bit_use_double_quant=True,       # 더블 양자화 활성화
    #bnb_4bit_quant_type="nf4",            # NF4 양자화 방식 사용
)

MODEL_PATH = "./models/exaone-2b-summary" # ./models/exaone-2b-summary
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

pipe_finetuned = pipeline("text-generation", 
                          model=model, 
                          tokenizer=tokenizer, 
                          max_new_tokens=500)

def call_model(user_input):
    
    messages = [
        {
            "role": "user",
            "content": "다음 글을 요약해주세요:\n\n{}".format(user_input)
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = pipe_finetuned(
    prompt,
    do_sample=True, # 확률적으로 다양한 답변 생성하고 싶으면 True지만 그럼 속도 느려짐 
    temperature=0.2,
    top_k=60,
    top_p=0.95,
    add_special_tokens=True,
    repetition_penalty=1.2
    )

    summary = outputs[0]["generated_text"][len(prompt):] # 프롬프트 이후의 텍스트만 추출

    return summary.strip()


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

