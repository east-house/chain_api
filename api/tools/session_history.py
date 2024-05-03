import pytz

from datetime import datetime

class ChatHistory:
    def __init__(self):
        self.chats = []
        
    def add_history(
        self, user_input: str, outputs: str, prev_conv: str, relevant_conv: str, all_prompt:str
    ) -> None:
        # Get current time
        now = datetime.now(pytz.timezone("Asia/Seoul"))

        chat = {
            "user_input": user_input,
            "output": outputs,
            "prev_conv": prev_conv,
            "relevant_conv": relevant_conv,
            "all_prompt":all_prompt
        }

        self.chats.append(chat)