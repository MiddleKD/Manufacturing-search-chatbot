from typing import Union
import anthropic
import voyageai

client = anthropic.Anthropic()  # Anthropoc 클라이언트 객체 생성
vo = voyageai.Client()  # Voyage AI 클라이언트 객체 생성

def query_to_claude(
        system_prompt:str,  # 시스템 프롬프트
        message_contexts:list[dict],  # 메시지 컨텍스트: 모델에 전달되는 이전 대화 내용 - 클로드에서 지원하는 형태로
        max_tokens:int = 1000,  # 생성할 최대 토큰 수
        temperature:float = 0,  # 생성에 사용할 온도(다양성 조절 매개변수)
        model_name:str = "claude-3-sonnet-20240229",  # 사용할 모델 이름
):
    """
    Anthropoc API를 사용하여 주어진 시스템 프롬프트와 메시지 컨텍스트를 기반으로 메시지를 생성하는 함수.
    """
    message = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=message_contexts  # 이전 대화 내용 전달
    )
    return message

def query_to_claude_stream(
        system_prompt:str,  # 시스템 프롬프트
        message_contexts:list[dict],  # 메시지 컨텍스트: 모델에 전달되는 이전 대화 내용 - 클로드에서 지원하는 형태로
        max_tokens:int = 1000,  # 생성할 최대 토큰 수
        temperature:float = 0,  # 생성에 사용할 온도(다양성 조절 매개변수)
        model_name:str = "claude-3-sonnet-20240229",  # 사용할 모델 이름
):  
    """
    Anthropoc API를 사용하여 주어진 시스템 프롬프트와 메시지 컨텍스트를 기반으로 메시지 스트림을 생성하는 함수.
    """
    with client.messages.stream(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=message_contexts
    ) as streamline:    # 스트림으로 생성 결과 반환
        for text in streamline.text_stream:
            yield text

def query_to_claude_with_additional_datas(
        query:str,
        message_contexts:list[dict],
        additional_data_text:Union[str, list],
        prompt_prefix=None,
        system_prompt="",
        base64_image=None,
        model_name="claude-3-haiku-20240307"):
    """
    추가 데이터와 함께 claude에게 쿼리를 보내고, 결과를 스트림으로 반환

    Args:
        query (str): 사용자 질문이나 쿼리.
        histories (list): 이전 대화 기록 리스트.
        additional_data_text (str, list): retrieve된 document
        prompt_prefix (str, optional): 쿼리 앞에 추가할 프롬프트 접두사. 기본값은 None입니다.
        system_prompt (str, optional): 시스템 프롬프트. 기본값은 ""입니다..
        base64_image (str, optional): base64로 인코딩된 이미지 데이터. 기본값은 None입니다.
        model_name (str, optional): 사용할 모델 이름.

    Yields:
        str: API가 생성한 각각의 답변을 스트림으로 반환합니다.
    """

    content = []
    if base64_image is not None:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image,
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": f"""{prompt_prefix}\n{query}"""
        }
    )

    message_contexts.append(
        {
            "role": "user",
            "content": content,
        }
    )
    
    for cur in query_to_claude_stream(
        system_prompt=system_prompt+str(additional_data_text),
        message_contexts=message_contexts,
        max_tokens=1000,
        temperature=0,
        model_name=model_name,
    ):
        yield cur

def text_embedding_with_voyage(
        text:str,
        model_name:str = "voyage-2", 
        input_type:str = "document"
):
    """
    Voyage AI API를 사용하여 주어진 텍스트를 임베딩하는 함수.
    """
    embed = vo.embed(text, 
        model=model_name, 
        input_type=input_type
    )
    return embed.embeddings
