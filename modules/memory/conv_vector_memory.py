from typing import Any, List, Dict

from langchain_core.pydantic_v1 import validator
from langchain_core.memory import BaseMemory
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.memory import (
    VectorStoreRetrieverMemory,
    ConversationBufferWindowMemory,
    CombinedMemory,
)
from modules.database.conversation import VectorDB

import json
## langchain 값 상속 받도록 class 수정
class ConvVectorMemory(BaseMemory):
    embedding_model:HuggingFaceInferenceAPIEmbeddings
    prefix_path:str
    vector_db:bool=False
    conv_buffer_window:bool=False
    memories:List=[]
    user_input_key:str=''
    ai_output_key:str=''
    top_k:int=1
    #     # runnable define
    #     self.runnable = self._init_runnable()
    def _init_memory(self)->None:
        with open(self.prefix_path, 'r') as f:
            prefix_format = json.load(f)
        self.user_input_key:str=prefix_format["USER_PREFIX"]
        self.ai_output_key:str=prefix_format["AI_PREFIX"]
        if self.vector_db:
            self.memories.append(self._init_vector_db())
        if self.conv_buffer_window:
            self.memories.append(self._init_conv_buffer_window(top_k=self.top_k))

    @property
    def memory_variables(self) -> List[str]:
        """All the memory variables that this instance provides."""
        """Collected from the all the linked memories."""
        memory_variables = []
        for memory in self.memories:
            memory_variables.extend(memory.memory_variables)
        return memory_variables

    def _init_vector_db(self) -> BaseMemory:
        # VectorDB and Retriever define
        retriever = VectorDB(self.embedding_model).retriever
        vectorstore_mem = VectorStoreRetrieverMemory(
            input_key=self.user_input_key,
            memory_key="relevant_conv",
            retriever=retriever
        )
        return vectorstore_mem

    def _init_conv_buffer_window(self, top_k:int) -> BaseMemory:
        conv_mem = ConversationBufferWindowMemory(
            input_key=self.user_input_key,
            memory_key="prev_conv",
            k=top_k,
            human_prefix=self.user_input_key,
            ai_prefix=self.ai_output_key
        )
        return conv_mem

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load all vars from sub-memories."""
        memory_data: Dict[str, Any] = {}
        # Collect vars from all sub-memories
        for memory in self.memories:
            data = memory.load_memory_variables(inputs)
            for key, value in data.items():
                if key in memory_data:
                    raise ValueError(
                        f"The variable {key} is repeated in the CombinedMemory."
                    )
                memory_data[key] = value
        return memory_data

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this session for every memory."""
        # Save context for all sub-memories
        for memory in self.memories:
            memory.save_context(inputs, outputs)

    def clear(self) -> None:
        """Clear context from this session for every memory."""
        for memory in self.memories:
            memory.clear()

    
