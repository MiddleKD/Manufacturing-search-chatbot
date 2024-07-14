from typing import Union
import chromadb
from api import query_to_claude_stream, text_embedding_with_voyage

# LLM API query를 위한 시스템 프롬프트
# 직접적인 유저와의 RAG 소통을 위함임
default_system_prompt = f"""
지시:
명확하고 신뢰가 가는 언어를 사용해.
너는 제조부품 전문가야 추가정보를 바탕으로 최대한 자세하고 전문적으로 대답해.
이건 나의 커리어에 매우 중요한 일이야 잘 해내면 팁을 줄게.

순서:
1. 유저의 질문 의도를 파악한다.
2. 유저가 원하는 바를 추가정보에서 검색하여 대답한다.
3. 이외에 도움이 되는 정보를 생각하여 전달한다.
4. 마지막에 어떤 추가정보를 사용했는지 id를 출력한다.

규칙:
1. 반드시 추가정보를 기반으로 대답할 것
2. 첫번째 규칙을 명시할 것
3. 한국어로 대답할 것
5. 어떤 추가정보를 사용했는지 id를 출력할 것
6. 출력에 추가정보라는 텍스트 사용을 자제할 것

출력 예시:
{{TEXT}}

제조부품 info 출력
제조부품 info 출력
제조부품 info 출력
...
{{도움이 되는 정보}}

*부품 id: [{{id}}, {{id}}, {{id}} ...]*

추가정보:
"""

class KnowledgeBaseManager:
    """
    Knowldedge base. RAG의 지식베이스를 관리
    """
    client = None  # 데이터베이스 클라이언트
    collection = None  # 데이터베이스 컬렉션
    prior_query = []  # 이전 유저의 query
    knowledge_base_cache = []  # 저장된 retrieve된 document

    def __init__(self, db_dir="./chroma_db", collection_name="knowledge_base"):
        """
        클래스 초기화 함수입니다. 데이터베이스와 컬렉션을 설정하고 초기 메타데이터를 출력합니다.

        Args:
            db_dir (str, optional): 데이터베이스 디렉토리 경로. 기본값은 "./chroma_db"입니다.
            collection_name (str, optional): 사용할 컬렉션 이름. 기본값은 "knowledge_base"입니다.
        """
        self.client = chromadb.PersistentClient(db_dir)
        self.collection = self.client.get_collection(collection_name)
        print(f"DB({collection_name})에는 {self.collection.count()}개의 데이터가 저장되어 있습니다.")

    def update_knowledge_base_cache(self, query:str, top_n=30):
        """
        지식 베이스 캐시를 업데이트하는 함수입니다. 주어진 쿼리에 기반하여 새로운 데이터를 캐시에 추가합니다.

        Args:
            query (str): 업데이트할 쿼리 문자열.
            top_n (int, optional): 검색 결과로 반환할 최대 지식기반 문서 수. 기본값은 30입니다.

        Returns:
            bool: 새로운 데이터가 캐시에 추가되었는지 여부를 나타내는 boolean 값.
        """
        embed_model_name = self.collection.metadata.get("text_embedding_model_name", "voyage-2")
        
        query_embd = text_embedding_with_voyage(
            text=query, 
            model_name=embed_model_name, 
            input_type="query")[0]
        
        knowledge_base = self.collection.query(
            query_embeddings=query_embd,
            n_results=top_n,
            include=["documents"]
        )
        parsed_kb = [{"id":int(id), "info": doc} 
                     for id, doc in 
                     zip(knowledge_base["ids"][0], knowledge_base["documents"][0])]
        
        self.prior_query.append(query)

        is_new_something = False
        key_list = [cur["id"] for cur in self.knowledge_base_cache]
        for record in parsed_kb:
            if record["id"] not in key_list:
                self.knowledge_base_cache.append(record)
                is_new_something = True
            else:
                continue
        
        return is_new_something
    
    def get_data_from_id(self, id):
        """
        지정된 ID에 해당하는 데이터와 도면 정보를 반환하는 함수입니다.

        Args:
            id (any): 검색할 데이터의 ID.

        Returns:
            tuple: 데이터(document)와 도면(blue_print) 정보를 포함한 튜플.
        """
        id = str(id)
        data = self.collection.get(ids=id, include=["metadatas", "documents"])
        
        blue_print = data["metadatas"][0]["blue_print"]
        document = data["documents"][0]
        
        return document, blue_print

    def get_knowledge_base_cache(self):
        return self.knowledge_base_cache
    
    def clean_knowledge_base_cache(self):
        """
        Knowldedge base 관리자에 저장된 이전 retrieve된 document를 초기화합니다.
        """
        self.knowledge_base_cache = []
    
    def get_prior_query(self, n_th=None):
        """
        이전에 수행한 쿼리 기록을 반환하는 함수입니다.

        Args:
            n_th (int or None, optional): 반환할 기록의 인덱스. 기본값은 None으로 전체 기록을 반환합니다.

        Returns:
            str or list: 특정 인덱스의 쿼리 문자열 또는 전체 쿼리 기록 리스트.
        """
        if n_th == None:
            return self.prior_query
        else:
            return self.prior_query[n_th]
    
def make_message_contexts_from_histories(histories):
    """
    gradio의 hisory에서 claude API를 위한 형태(message contexts)로 파싱합니다.
    """
    message_contexts= []
    histories = [cur for cur in histories if cur[1] is not None]
    for history in histories:
        user_content, assistant_content = history
        message_context = [{
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        }]
        message_contexts.extend(message_context)
    
    return message_contexts
