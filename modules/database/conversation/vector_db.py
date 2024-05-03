from typing import List
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

# from langchain_core.retrievers import BaseRetriever
import faiss

class VectorDB:
    def __init__(self, embedding_model:HuggingFaceInferenceAPIEmbeddings, dimension:int=1024, top_k:int=2)->None:
        index = faiss.IndexFlatL2(dimension)
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )
        self.retriever = vector_store.as_retriever(search_kwargs=dict(k=top_k))