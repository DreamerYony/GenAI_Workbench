## Retriever - Kiwi tokenizer & BM25 & FAISS 🍎

- 프로토타입용으로 구현한 RAG(Retrieval-Augmented Generation) 기반 챗봇 모델입니다.
- 로컬 환경에서 사전에 다운로드한 언어 모델을 활용하며, On-prem 환경에 적합하도록 설계되었습니다.
- 외부 API와의 연동 없이, 자체적으로 모델과 검색 시스템을 실행합니다.
- 벡터 인덱싱은 FAISS를 사용하여 구현하였습니다.
  
### 파일 설명
- app_basic.py : Flask 기반 웹 애플리케이션으로, 검색 및 답변 생성을 처리합니다. Kiwi 토크나이저, BM25, FAISS를 활용한 검색 및 LLM 기반 응답 생성 포함합니다.
- app_rrf.py : Kiwi 토크나이저, BM25, FAISS에 RRF 기법을 추가하였습니다.
- app_rrfhyde.py : Kiwi 토크나이저, BM25, FAISS, RRF에 HyDE 기법을 추가하였습니다.
- app_summary.py : QLoRA로 한국어 특화 요약 모델을 만들었습니다.
- index_ver222.html : 웹 애플리케이션의 기본 HTML 템플릿입니다. 질문 입력 및 결과 출력을 위한 UI 구조를 포함합니다.
- script.js : 사용자 인터페이스의 동적 동작을 지원하는 JavaScript 파일입니다. 
- style.css : UI 스타일링을 위한 CSS 파일입니다. 사용자 친화적인 UX를 구현하기 위해 디자인 요소를 정의했습니다.
