from pydantic import BaseModel, Field
from typing import List


class ListHistoryResponse(BaseModel):
    session_id: str
    title: str
    date: str