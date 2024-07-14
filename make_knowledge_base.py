"""
RAG로 retrieve하는 데이터(Knowledge base)들을 정제 및 전처리하여 작업하는 코드
"""
import os
import re
import json
from PIL import Image
from utils import pil_to_b64
from glob import glob

import pandas as pd
from api import query_to_claude, text_embedding_with_voyage
from dotenv import load_dotenv

load_dotenv()

# 도면과 features를 기준으로 Knowledge base를 만들기 위한 프롬프트
# 도면과 features를 하나의 임베딩 벡터로 압축하는 것이 목표
# Few shot 프롬프트
example_output = """
"-분류정보-
품종: 나비 너트_1종_M2
분류: 너트>나비너트>1종
-치수정보-
너트의 폭:4.0(A')
너트의 높이:3.0(B')
너트의 두께:2.0(C)
너트의 전체 길이:12.0(D)
너트의 전체 높이:6.0(H)
너트의 구멍 직경:2.5(G1)"

"-분류정보-
품종: 스터드 볼트_1종_M10x16
분류: 볼트>스터드 볼트>1종
-치수정보-
볼트 직경:10.0(D)
볼트 머리 직경:20.0(B)
볼트 길이:16.0(Bm)
볼트 머리 높이:12.0(Z)
볼트 머리 각도:1.5(La)
볼트 머리 길이:2.0(L)"

"-분류정보-
품종: 홈붙이 접시 머리 볼트_상_납작끝_M10x16
분류: 볼트>접시 머리 볼트> 홈붙이 접시 머리 볼트_상>납작끝
-치수정보-
머리 직경:0.1(D)
나사 직경:10.0(Dk)
머리 높이:2.5(K)
전체 길이:16.0(C)
머리 폭:5.5(N)
홈 깊이:1.5(T)
볼트 머리 직경:20.0(Z)
나사 피치:2.0(L)"
"""
# System 프롬프트 생성
def get_system_prompt(example_output):

    system_prompt=\
f"""지시:
반말로 간결, 명확하고 신뢰가 가는 언어를 사용해.
너는 제조기업의 도면 분석 전문가야. 도면 이미지와 추가정보를 입력으로 받으면 최대한 자세하고 전문적으로 분석해.
이건 나의 커리어에 매우 중요한 일이야 잘 해내면 팁을 줄게.

순서:
1. 도면에서 치수를 나타내는 알파벳을 뽑는다.
2. 추가정보를 바탕으로 치수가 무엇을 의미하는지 파악한다.
3. 각각의 알파벳이 무엇을 나타내는지 간단한 제목을 매긴다.
4. 각각의 알파벳이 도면의 무엇을 의미하는지 설명한다.

규칙:
1. 도면의 치수 정보를 명확히 나타내는데 최선을 다할 것
2. 첫번째 규칙을 명시할 것 
3. 한국어로 대답할 것
4. 500 토큰 이하로 간결한 문장을 사용할 것

출력 예시:
{example_output}"""
    return system_prompt

def make_concated_df(data_path, engine="odf"):
    """
    엑셀 파일을 읽어 각 시트를 하나의 데이터프레임으로 연결(concatenate)하여 반환하는 함수.

    Args:
        data_path (str): 엑셀 파일 경로.
        engine (str, optional): 엔진 타입 (기본값: 'odf').

    Returns:
        pandas.DataFrame: 각 시트가 연결된(concatenated) 데이터프레임.
    """
    df = pd.read_excel(data_path, engine=engine, sheet_name=None)
    concated_df = pd.concat([df[cur] for cur in df])
    return concated_df


def make_feature_map(concated_df):
    """
    concat 데이터프레임에서 각 부품(부품명, 'bp')별로 특징 맵(feature map)으로 변환하여 반환하는 함수.

    Args:
        concated_df (pandas.DataFrame): 시트가 연결된(concatenated) 데이터프레임.

    Returns:
        dict: 각 부품명을 키로 하고, 해당 부품에 대한 특징을 포함하는 딕셔너리를 값으로 하는 특징 맵.
              예시: {'부품명1': {'특징1': 값1, '특징2': 값2, ...}, '부품명2': {'특징1': 값1, '특징2': 값2, ...}, ...}
    """
    bp_names = concated_df["bp"].unique().tolist()

    feature_map = {}
    for bp_name in bp_names:
        features = concated_df[concated_df["bp"] == bp_name].iloc[0].drop(["bp"]).dropna().to_dict()
        feature_map[bp_name] = features

    return feature_map


def make_prompt_template_map(feature_map, common_keys=["품종", "분류"]):
    """
    feature map을 기반으로 LLM API의 프롬프트 템플릿을 생성하여 반환.

    Args:
        feature_map (dict): 각 부품명을 키로 하고, 해당 부품에 대한 특징을 포함하는 딕셔너리를 값으로 하는 특징 맵.
                            예시: {'부품명1': {'특징1': 값1, '특징2': 값2, ...}, '부품명2': {'특징1': 값1, '특징2': 값2, ...}, ...}
        common_keys (list, optional): 모든 프롬프트에 포함될 공통 키 리스트. 기본값은 ["품종", "분류"].

    Returns:
        dict: 각 부품명을 키로 하고, LLM API의 프롬프트 템플릿을 값으로 하는 맵.
              예시: {'부품명1': "프롬프트 템플릿1", '부품명2': "프롬프트 템플릿2", ...}
    """
    template_map = {}

    for bp_name in feature_map:

        feature_keys = [cur for cur in feature_map[bp_name].keys() if cur not in common_keys]

        output_template = ""
        
        output_template = "-분류정보-\n"
        for cur in common_keys:
            output_template += cur+":{{TEXT}}"
            output_template += "\n"

        output_template += "-치수정보-\n"
        for cur in feature_keys:
            output_template += "{{DETAIL TEXT ONLY DESCRIPTION OF "+cur+"}}:{{ONLY NUMBER}}"+f"({cur})"
            output_template += "\n"

        template_map[bp_name] = output_template

    return template_map


def make_bs64_bp_map(feature_map, blueprint_saved_directory_name):
    """
    각 부품명에 해당하는 도면 이미지를 base64 형식으로 변환하여 매핑한 맵을 생성하여 반환하는 함수.

    Args:
        feature_map (dict): 각 부품명을 키로 하고, 해당 부품에 대한 특징을 포함하는 딕셔너리를 값으로 하는 특징 맵.
                            예시: {'부품명1': {'특징1': 값1, '특징2': 값2, ...}, '부품명2': {'특징1': 값1, '특징2': 값2, ...}, ...}
        blueprint_saved_directory_name (str): 도면 이미지가 저장된 디렉토리 이름.

    Returns:
        dict: 각 부품명을 키로 하고, 해당 부품의 도면 이미지를 base64로 변환한 값을 값으로 하는 맵.
              예시: {'부품명1': '도면 이미지 base64 문자열1', '부품명2': '도면 이미지 base64 문자열2', ...}
    """
    
    def get_bp_b64(target_bp_name):
        bp_image_path_list = glob(blueprint_saved_directory_name+f"/{target_bp_name}*")
        bp_images = [Image.open(cur).convert("RGB") for cur in bp_image_path_list]
        base64_images = [pil_to_b64(cur) for cur in bp_images]
        return base64_images
    
    bs64_bp_map = {}
    for bp_name in feature_map.keys():
        bs64_bp_map[bp_name] = get_bp_b64(bp_name)[0]

    return bs64_bp_map

def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


if __name__ == "__main__":
    print("PROCESS START!")

    metadata_file_path = os.environ.get("CAD_FEATURE_PATH")
    blueprint_saved_directory_name = os.environ.get("BLUEPRINT_PATH")
    text_embedding_model = os.environ.get("TEXT_EMBEDDING_MODEL")

    print("parsing datas...")
    # 엑셀파일의 시트를 하나의 데이터 프레임으로 통합
    concated_df = make_concated_df(metadata_file_path, engine="odf")
    # 부품명 : features 생성
    feature_map = make_feature_map(concated_df)
    # 부품명 : 프롬프트 템플릿 생성
    prompt_template_map = make_prompt_template_map(feature_map, common_keys=["품종", "분류"])
    # 부품명 : 도면 bs64 이미지 생성
    bs64_bp_map = make_bs64_bp_map(feature_map, blueprint_saved_directory_name=blueprint_saved_directory_name)
    # 부품명(도면에 대한) 리스트
    bp_names = list(feature_map.keys())

    print("query to claude...")
    result_dict = {}
    for bp_name in bp_names:
        output_template = prompt_template_map[bp_name]
        prompt_feature = feature_map[bp_name]
        base64_image = bs64_bp_map[bp_name]
        
        system_prompt = get_system_prompt(example_output)
        message = query_to_claude(
            system_prompt=system_prompt,
            max_tokens=300,
            temperature=0,
            model_name="claude-3-haiku-20240307",
            message_contexts=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{prompt_feature}\n 주어진 정보들을 사용하여, 도면을 분석하여 적절한 수치와 주석을 아래 양식에 맞게 채워.\n {output_template}"
                    }
                ],
            }
            ],
        )
        result_dict[bp_name] = message.content[0].text
    
    print("replace feature text...")
    knowledge_base_text = []
    metadata = []
    for bp_key in result_dict:

        for _, cur in concated_df[concated_df["bp"]==bp_key].drop(["bp"], axis=1).dropna(axis=1).iterrows():
            src = cur.to_dict()
            tp = result_dict[bp_key]

            for src_key in src:
                tp = re.sub(f"{src_key}:.*\n", f"{src_key}: {src[src_key]}\n", tp)
                tp = re.sub(f":.*\({src_key}\)", f":{src[src_key]}", tp)

            knowledge_base_text.append(tp)
            metadata.append({"blue_print":bp_key})
    
    DATA_DIR = os.getenv("FILE_DATA_DIR")
    with open(os.path.join(DATA_DIR, "knowledge_base_text.json"), mode="w", encoding="utf-8") as f:
        json.dump(knowledge_base_text, f, indent=4)
    

    print("embed with voyage...")
    knowledge_base_text_chunks = chunk_list(knowledge_base_text, 128)

    knowledge_base_vector = []
    for text_chunk in knowledge_base_text_chunks:
        embed_text_chunk = text_embedding_with_voyage(text_chunk, model_name=text_embedding_model)
        for embed_text in embed_text_chunk:
            knowledge_base_vector.append(embed_text)
    
    with open(os.path.join(DATA_DIR, "knowledge_base_vector.json"), mode="w") as f:
        json.dump(knowledge_base_vector, f, indent=4)
    with open(os.path.join(DATA_DIR, "metadata.json"), mode="w") as f:
        json.dump({"text_embedding_model_name":text_embedding_model,
                   "metadata":metadata}, f, indent=4)
        
    print("ALL PROCESS IS DONE!")