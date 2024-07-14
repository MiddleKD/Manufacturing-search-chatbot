# 🎨 Manufacturing-search-chatbot
<!-- ![git_header](assets/favorfit_git_header.png) -->
도면과 메타데이터를 Multimodal embedding하여 문서로 변환하고, 매우 간편하게 검색할 수 있는 챗봇입니다.

## 🎬 Demo
<details>
    <summary><strong>Interface</strong></summary>
    <ul>
        <img src="assets/interface.png" alt="interface">
    </ul>
</details>
<details>
    <summary><strong>Only text query</strong></summary>
    <ul>
        <img src="assets/demo1.png" alt="demo1">
        <img src="assets/demo2.png" alt="demo2">
        <img src="assets/demo3.png" alt="demo3">
        <img src="assets/demo4.png" alt="demo4">
        <img src="assets/demo5.png" alt="demo5">
    </ul>
</details>
<details>
    <summary><strong>Image & text query</strong></summary>
    <ul>
        <img src="assets/demo6.png" alt="demo6">
        <img src="assets/demo7.png" alt="demo7">
    </ul>
</details>
<details>
    <summary><strong>DB search by ID</strong></summary>
    <ul>
        <img src="assets/demo8.png" alt="demo8">
    </ul>
</details>

## 📌 Index
1. [Introduction](#-introduction)
2. [Features](#-features)
3. [Approach](#-approach)
4. [Install](#-install)
5. [How to use](#-how-to-use)
6. [Contact](#-contact)

## 🚀 Introduction
Manufacturing-search-chatbot 프로젝트는 제조업체 및 기술 지원팀을 위한 흥미로운 솔루션입니다. 이 시스템은 도면, 메타데이터, 그리고 AI 기술을 활용하여 **제조 부품에 대한 검색 혹은 질문에 신속하고 정확한 답변을 제공**합니다. RAG(Retrieval-Augmented Generation) 기술을 기반으로 하여 사용자 질의에 대해 맞춤형 응답을 생성하여 제조 프로세스의 효율성을 향상시킵니다.

## 💡 Features
### 1. 도면 기반 질의 응답
- 제품 도면과 함께 질의가 가능합니다.
- 도면 이미지와 도면 묘사를 통한 고급 검색 및 질의 기능이 제공됩니다.
- 기존 DB 쿼리 방식을 넘어선 흥미로운 제조부품 검색 솔루션입니다.

### 2. LLM 기반 자연어 처리
- 유연한 언어 입력으로 검색이 가능합니다. (예: "직경 3cm" ≈ "길이 3cm")
- 정확한 전문 용어 없이도 효과적인 검색을 지원하기에 비전문가도 쉽게 사용할 수 있습니다.

### 3. 지식 베이스 구축
- 새로운 제조부품 정보를 쉽게 업데이트 및 통합할 수 있습니다.
- 엑셀 파일과 도면 이미지가 필요합니다.

### 4. RAG 시스템
- 할루시네이션 감소를 위한 LLM 기술 적용이 필요합니다.
- 민감한 제조부품 데이터에 대한 높은 정확성 보장합니다.
- 신뢰할 수 있는 정보 제공 시스템

### 5. GUI 인터페이스
- Gradio앱을 GUI를 제공합니다. 비전문가도 쉽게 사용할 수 있습니다.

## 🛠 Approach
### 1. Background: 문제 인식
현대 제조업체 및 기술 지원 팀은 제품 부품에 대한 정확하고 신속한 정보 접근이 중요합니다. 그러나 다음과 같은 문제점들이 있습니다:

- **정보 분산**: 도면, 메타데이터, 텍스트 문서 등의 정보가 분산되어 있어 검색 및 접근이 어렵습니다.
- **시간 소모**: 사용자가 필요로 하는 정보를 찾는 과정이 수작업이거나 복잡하여 시간이 많이 소요될 수 있습니다.
- **정보 오류**: 다양한 소스로부터 수집된 정보의 일관성과 정확성을 유지하는 것이 어렵습니다.
- **전문 지식 의존성**: 복잡한 제품 정보를 이해하고 해석하는 데 전문가의 개입이 자주 필요합니다.
- **실시간 업데이트 부족**: 제품 정보가 빠르게 변경될 수 있지만, 기존 시스템은 이를 신속하게 반영하지 못합니다.

제조업 분야에서는 복잡한 부품 정보에 대한 신속하고 정확한 접근이 필요합니다. 기존의 검색 시스템은 종종 부정확하거나 시간이 많이 소요되어 생산성 저하의 원인이 되었습니다.

### 2. Solution: Multimodal embedding document
- **Multimodal embedding**: **제품 도면과 상세 메타데이터를 하나의 임베딩 벡터에 통합**하여 보다 유연한 검색 문서를 마련합니다.
- **퓨샷 프롬프트 엔지니어링**: LLM의 결과물을 적절한 형태의 응답으로 유도하여, **도면과 메타데이터의 정보손실을 최소화**합니다.
- **벡터 데이터베이스 활용**: VoyageAPI와 chromaDB를 사용하여 **효율적인 데이터 저장 및 검색 시스템** 구현합니다.
- **RAG 기술 적용**: 벡터 DB에 저장된 정보를 효과적으로 검색하고 retrieve된 문서와 유저의 쿼리를 기반으로 새로운 응답
- **LLM 챗봇 적용**: LLM의 유연한 언어능력을 기반으로 **다양한 도메인(전문가, 비전문가 etc)의 쿼리를 통합적으로 처리**합니다.

## 📥 Install
```bash
# 저장소 클론
git clone {this repository}
cd Manufacturing-part-search-chatbot

# 가상 환경 생성 및 활성화
conda create -n manufac-chat python=3.10
conda activate manufac-chat

# 필요한 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 필요한 API 키와 설정을 입력하세요
vim .env
```

## 🖥 How to use
1. 시스템 실행
    ```bash
    python3 app.py
    ```
2. 인터페이스에서 제품 부품에 대한 질문 입력
3. 필요한 경우 관련 도면 이미지 업로드
4. '제출' 버튼 클릭하여 응답 확인

## 📞 Contact
middlek - middlekcenter@gmail.com

<!-- favorfit - lab@favorfit.ai -->