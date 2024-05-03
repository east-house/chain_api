from langchain.prompts import PromptTemplate

import json

system_prompt = PromptTemplate(
    template="""<|SYSTEM|>\n\n당신은 사용자의 질문에 대해서 상세한 답변을 제공하는 어시스턴트입니다.""",
    input_variables=["system"]
)

user_prompt = PromptTemplate(
    template="""아래는 현재 사용자 질문입니다. 사용자의 질문에 답변하세요:\n
<|USER|>\n
{input}\n
<|ASSISTANT|>""",
    input_variables=["input"]
)

relevant_prompt = PromptTemplate(
    template="""아래는 사용자와의 채팅 내역 중 일부입니다. 사용자 질문과 관련 있을 경우에만 답변에 참고하세요:\n
{relevant_conv}""",
    input_variables=["relevant_conv"]
)

prev_prompt = PromptTemplate(
    template="""아래는 최근 사용자와의 채팅 내역입니다:\n
{prev_conv}""",
    input_variables=["prev_conv"]
)

simple_prompt = PromptTemplate(
    template="""<|SYSTEM|>\n
당신은 사용자의 질문에 대해서 상세한 답변을 제공하는 어시스턴트입니다.
만약, 질문에 대한 답을 알지 못한다면 답에 대해 알지 못한다고 솔직하게 답변하세요.
다음의 사항에 유의하십시오.\n
- 요약 대화 내역, 관련 대화 내역, 최근 대화 내역이 있을 때만 참고하여 답변에 참고하십시오.\n
- 현재 질문에 대한 답변만 하십시오.\n\n
<|USER|>\n
"질문 : {input}\n
"<|ASSISTANT|>\n
"답변 :""",
    input_variables=['input']
)

None_prompt = PromptTemplate(
    template="",
    input_variables=[""]
)
