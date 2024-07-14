"""
make_knowledge_base.py에서 생성한 knowledge base를 chroma db에 업데이트
"""
import os
import json
import chromadb
from dotenv import load_dotenv

load_dotenv()

DB_DIR = os.getenv("DB_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DATA_DIR = os.getenv("FILE_DATA_DIR")


print("Chroma Client 생성 중...")
client = chromadb.PersistentClient(DB_DIR)


print(f"{COLLECTION_NAME} Collection 불러오는 중...")
try:
    collection = client.get_collection(COLLECTION_NAME)
except ValueError:
    # 콜렉션이 없는 경우
    print(f"Collection이 존재하지 않습니다. {COLLECTION_NAME}을 새로 생성합니다.")
    collection = client.create_collection(COLLECTION_NAME)


print(f"{DATA_DIR}에서 데이터 불러오는 중...")
with open(os.path.join(DATA_DIR, "knowledge_base_text.json"), mode="r") as f:
    jsonlike_text = json.load(f)
with open(os.path.join(DATA_DIR, "knowledge_base_vector.json"), mode="r") as f:
    jsonlike_vector = json.load(f)
with open(os.path.join(DATA_DIR, "metadata.json"), mode="r") as f:
    jsonlike_meta = json.load(f)
    record_meta = jsonlike_meta.pop("metadata")
    collection_meta = jsonlike_meta

if not (len(jsonlike_text)==len(jsonlike_vector)==len(record_meta)):
    raise ValueError("데이터가 무결하지 않습니다.")


print("Chroma DB에 데이터 업데이트 중...")
collection.modify(metadata=collection_meta)
pre_count = collection.count()
collection.add(
    documents=jsonlike_text,
    embeddings=jsonlike_vector,
    metadatas=record_meta,
    ids=[str(cur) for cur in range(pre_count, len(jsonlike_text))]
)

print(f"{pre_count-collection.count()}개의 데이터가 {DB_DIR}에 저장되었습니다.")