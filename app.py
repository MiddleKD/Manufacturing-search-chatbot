import os
import gradio as gr
from utils import open_pil, pil_to_b64
from rag_func import (KnowledgeBaseManager, 
                      default_system_prompt,
                      make_message_contexts_from_histories)
from api import query_to_claude_with_additional_datas
from dotenv import load_dotenv

load_dotenv()

DB_DIR = os.environ.get("DB_DIR")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
BLUEPRINT_PATH = os.environ.get("BLUEPRINT_PATH")

# Knowldedge base 관리자 객체 생성. RAG의 지식베이스의 전체적 흐름을 관리.
kb_manager = KnowledgeBaseManager(db_dir=DB_DIR, collection_name=COLLECTION_NAME)

def send_chat(query:str, history:list, num_of_doc:int):
    """
    질문과 history(이전 대화), retrieve 문서 개수를 받아 api를 통해 답변을 생성하고 반환하는 함수.

    Args:
        query (str): 사용자의 질문 텍스트.
        history (list): 이전 대화 내역을 담고 있는 gradio의 history 리스트.
        num_of_doc (int): 지식 베이스에서 추천할 문서의 개수.

    Yields:
        str: 챗봇이 생성한 각 답변 텍스트를 하나씩 반환.
    """
    query_text = query.get("text", "")
    query_image = query.get("files", [])

    if len(query_text) < 3:
        yield "질문을 입력해주시면 최선을 다해 답변드리겠습니다."
        return
    if len(query_image) == 1:
        try:
            pil_img = open_pil(query_image[0])
            query_image = pil_to_b64(pil_img)
        except:
            yield "이미지 파일만 업로드 가능합니다."
            return
    elif query_image == []:
        query_image = None

    has_new_kb = kb_manager.update_knowledge_base_cache(query_text, top_n=num_of_doc)

    if has_new_kb == False:
        yield "더 구체적인 정보가 필요합니다."
        return
    
    message_contexts= make_message_contexts_from_histories(history)
    output_chat = ""
    for text in query_to_claude_with_additional_datas(
        query=query_text,
        message_contexts=message_contexts,
        additional_data_text=kb_manager.get_knowledge_base_cache(),
        system_prompt=default_system_prompt,
        base64_image=query_image,
        model_name="claude-3-haiku-20240307",
    ):
        output_chat+=text
        yield output_chat

def get_data(id):
    """
    제품 부품 ID를 입력받아 해당 ID에 대한 정보와 이미지를 반환하는 함수.

    Args:
        id (any): 제품 부품 ID.

    Returns:
        tuple: ID에 대한 정보와 이미지를 포함한 튜플.
            첫 번째 요소는 부품 정보를 담고 있고, 두 번째 요소는 PIL 이미지 객체입니다.
            ID가 존재하지 않는 경우 ("존재하지 않는 ID", None)를 반환합니다.
    """
    try:
        info, img_fn = kb_manager.get_data_from_id(id)
        img = open_pil(os.path.join(BLUEPRINT_PATH, img_fn+".jpg"))
    except:
        return "존재하지 않는 ID", None
    return info, img

def clean_cache():
    kb_manager.clean_knowledge_base_cache()

# Gradio app 데모

if __name__ == "__main__":
    clear_btn = gr.Button(value="🗑️  Clear", variant="secondary")
    with gr.ChatInterface(
            fn=send_chat,
            examples=[[{"text": "D가 10이고 H가 3인 나비 모양 부품은 뭐가 있지?", "files": ['./image/example.jpg']}, 50],
                    [{"text": "길이가 16이고 직경이 5인 볼트는 뭐가 있지?", "files": []}, 30]],
            title="제조 부품 검색 Chat AI",
            clear_btn=clear_btn,
            retry_btn=None,
            undo_btn=None,
            stop_btn=None,
            multimodal=True,
            additional_inputs=[
                gr.Slider(10, 200, value=30, label="추천 후보 개수", 
                    info="증가할수록 높은 정확도를 보이지만, 시간이 더 걸리고, 더 많은 요금이 지불됩니다.")
            ],
            fill_height=False
        )as demo:

        with gr.Accordion("제품 부품 ID 검색", open=False):
            with gr.Row():
                id_box = gr.Number(show_label=False)
                search_btn = gr.Button(value="검색", size="sm")
            with gr.Row():    
                bp_img_box = gr.Image(format="jpg", interactive=False, show_download_button=False, show_label=False, image_mode="RGB")
                info_box = gr.TextArea(show_label=False, text_align="left")

            search_btn.click(get_data, inputs=[id_box], outputs=[info_box, bp_img_box])

        clear_btn.click(clean_cache, None, None)
    demo.launch(share=True)
