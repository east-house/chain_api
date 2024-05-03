import time
import json
import argparse
import logging
import asyncio
import pytz
import uvicorn
import requests

from typing import Iterable, List, Union
from threading import Thread
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, Body, Form, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from loguru import logger
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from modules.memory import ConvVectorMemory
from modules.prompts import SelectTemplate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--tgi_url", type=str, default="http://192.168.1.20:1330") # mistralai/Mixtral-8x7B-Instruct-v0.1
    parser.add_argument("--tokenizer_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--embedding_name", type=str, default="intfloat/multilingual-e5-large")

    args = parser.parse_args()
    return args

class Server:
    def __init__(self, args):
        self.args = args
        self.session_time_check = 10
        # self.tgi_url = args.tgi_url
        self.tgi_url = "http://172.17.0.10:80"
        self.tgi_model = HuggingFaceTextGenInference(
            inference_server_url=self.tgi_url,
            # max_new_tokens=4096,
            max_new_tokens=2048,
            top_k=10,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.03,
            streaming=True
        )
        self.text_embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key="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",
            api_url="http://192.168.1.21:11180",
            model_name="intfloat/multilingual-e5-large"
        )

        
        self.embedding_fn = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')


    def _delete_and_save_session(self, did: str) -> None:
        current_chats = self.active_sessions[did].current_session.chat_history.chats
        if current_chats == []:
            # Remove active session from user
            self.active_sessions[did].delete_session()

            # Delete the user from active session
            del self.active_sessions[did]
        else:
            # Add the dictionary to User directly update the user database
            session_id = self.active_sessions[did].current_session.session_uuid
            self.active_sessions[did].history_sessions[session_id] = current_chats
            self.active_sessions[did].delete_session()
            # Delete the user from active session
            del self.active_sessions[did]

    async def _check_session(self):
        """Check session."""
        # In Minutes to Seconds
        consecutive_sleep_duration = self.session_time_check * 60
        while True:
            time.sleep(consecutive_sleep_duration)
            current_time = datetime.now(pytz.timezone("Asia/Seoul"))
            if self.active_sessions:
                for _, (user_key, user_value) in enumerate(
                    self.active_sessions.copy().items()
                ):
                    end_session_time = user_value.current_session.approximation_session_end
                    if current_time > end_session_time:
                        # Before delete the user, save the user information first into database
                        # Get this session chats
                        self._delete_and_save_session(did=user_key)
            else:
                continue

    def session_wrapper(self) -> None:
        """Wrapper for session checker."""
        asyncio.run(self._check_session())

    def run(self):
        app = FastAPI()
        @app.exception_handler(HTTPException)
        def http_exception_handler(request, exc):
            return JSONResponse(
                content={
                        "state":exc.status_code,
                        "result":[],
                        "message":exc.detail
                    }
            )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 모든 origin 허용. 필요에 따라 변경 가능
            allow_credentials=True,
            allow_methods=["*"],  # 모든 HTTP 메소드 허용
            allow_headers=["*"],  # 모든 헤더 허용
        )

        @app.get('/test')
        def health_check():
            return {
                "message":True
            }

        @app.post("/embedding_test")
        def embedding_test(prompt:str = Body(...)):
            test1 = self.text_embeddings.embed_query(prompt)
            logger.info(f"test1 : {test1}")
            return {
                "message":test1
            }

        @app.post("/simple_llm")
        def simple_llm(
                prompt:str =Body(...),
                max_new_tokens:int=Body(2048),
                top_k:int=Body(10),
                top_p:float=Body(0.95),
                temperature:float=Body(0.7),
                repetition_penalty:float=Body(1.03),
            )->dict:
            """
            API for models that simply respond to user input
            """
            from modules.prompts.unit_prompt import simple_prompt
            llm = HuggingFaceTextGenInference(
                inference_server_url=self.tgi_url,
                # max_new_tokens=4096,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                streaming=True
            )            
            chain = simple_prompt|llm
            response = chain.invoke(prompt)
            # return StreamingResponse(response, media_type="text/event-stream")
            return {
                "response":response
            }

        # just chain type llm Inference API
        @app.post("/chain_inference")
        def chain_inference(
                prompt:str =Body(...),
                # max_new_tokens:int=Body(2048),
                # top_k:int=Body(100),
                # top_p:float=Body(0.95),
                # temperature:float=Body(0.7),
                # repetition_penalty:float=Body(1.03),
            ):
            # Memory define

            ### 전체 체인 함수로 변경
            llm = HuggingFaceTextGenInference(
                inference_server_url=self.tgi_url,
                # max_new_tokens=4096,
                max_new_tokens=2048,
                top_k=100,
                top_p=0.95,
                temperature=0.7,
                repetition_penalty=1.03,
                streaming=True
            ) 
            memory = ConvVectorMemory(
                embedding_model=self.text_embeddings,
                prefix_path='./data/prefix_template.json',
                # vector_db=True,
                conv_buffer_window=True
            )
            memory._init_memory()
            from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel
            memory_variables = memory.memory_variables
            runnable = RunnableParallel(
                    input=RunnablePassthrough(),
                    prev_conv=lambda x : memory.load_memory_variables({memory.user_input_key:x})["prev_conv"] if 'prev_conv' in memory_variables else '',
                    relevant_conv=lambda x : memory.load_memory_variables({memory.user_input_key:x})["relevant_conv"] if 'relevant_conv' in memory_variables else ''
                )
            
            # Prompt define
            prompt_template = SelectTemplate(template='',input_variables=[''])
            # Inference chain define
            chain = runnable | prompt_template | self.tgi_model
            ###

            # response = chain.astream(prompt) #streaming inference [generator type]
            response = chain.invoke(prompt)
            logger.warning(f"response : {response}")
            return StreamingResponse(response, media_type="text/event-stream")

        # @app.


        uvicorn.run(app, host=self.args.host, port=self.args.port)

if __name__ == "__main__":
    args = parse_args()
    server = Server(args)

    session_checker_thread = Thread(target=server.session_wrapper)
    session_checker_thread.daemon = True
    session_checker_thread.start()

    server.run()