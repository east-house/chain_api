import uuid
import pytz

from router.m2m_llm_tool.session_history import ChatHistory
from router.m2m_llm_tool.conv_chain import ConversationChain
from datetime import datetime, timedelta, timezone

class Session:
    def __init__(self, conv_chain: ConversationChain, session_timeout: int = 1, types:str="normal"):
        self.session_uuid = str(uuid.uuid4())
        self.session_timeout = session_timeout

        self.session_start = datetime.now(pytz.timezone("Asia/Seoul"))
        self.approximation_session_end = self.session_start + timedelta(
            minutes=session_timeout
        )
        self.types=types
        self.conv_chain = conv_chain
        self.chat_history = ChatHistory()

        self.doc_fn=[] #fn04 요약 문석 이름
        self.doc_type="" #fn02 보고서양식
        self.content_info={} #fn02 세부정보

    def update_session_end(self) -> None:
        self.session_start = datetime.now(pytz.timezone("Asia/Seoul"))
        # Add another one hour
        self.approximation_session_end = self.session_start + timedelta(
            minutes=self.session_timeout
        )

    def get_session_time_left(self) -> int:
        return (
            self.approximation_session_end
            - datetime.now(pytz.timezone("Asia/Seoul")).seconds
        )

    def replace_session(self, session_id: str):
        self.session_uuid = session_id
        self.session_start = datetime.now(pytz.timezone("Asia/Seoul"))
        self.approximation_session_end = self.session_start + timedelta(
            minutes=self.session_timeout
        )
    # fn02
    def _set_doc_type(self, doc_type:str) -> None:
        self.doc_type=doc_type
    def _set_content_info(self, content_info:dict) -> None:
        self.content_info = content_info
    def _get_content_info(self):
        return self.content_info
    # fn04
    def _set_doc_fn(self, doc_fn:list) -> None:
        self.doc_fn=doc_fn