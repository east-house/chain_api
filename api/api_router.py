from tpying import Union, List, Iterable
from loguru import logger

from fastapi import APIRouter, Request, Body, Form, Query
from fastapi.responses import StreamingResponse

from modules.memory import ConvVectorMemory

m2m_router = APIRouter(
    tags=["m2m"]
)

##


##
@m2m_router.post("/chain_inference")
def chain_inference(prompt:str = Body(...)):
