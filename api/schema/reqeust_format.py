from typing import Union
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class ModelRequest(BaseModel):
    prompt: str
    max_length: Union[int, None] = Field(
        default=2048, title="Maximum length of the sequence to be generated."
    )
    temperature: Union[float, None] = Field(
        default=0.5, title="The value used to module the next token probabilities."
    )
    top_k: Union[int, None] = Field(
        default=50,
        title="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )
    top_p: Union[float, None] = Field(
        default=0.9,
        title="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.",
    )
    repetition_penalty: Union[float, None] = Field(
        default=1.0, title="The parameter for repetition penalty. 1.0 means no penalty."
    )
    do_sample: Union[bool, None] = Field(
        default=True,
        title="Whether or not to use sampling ; use greedy decoding otherwise.",
    )