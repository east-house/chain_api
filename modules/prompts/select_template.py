from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.string import DEFAULT_FORMATTER_MAPPING

from modules.prompts.unit_prompt import *

from loguru import logger

class SelectTemplate(PromptTemplate):
    def __init__(self, template, input_variables):
        super().__init__(template=template, input_variables=input_variables)
    
    def format(self, **kwargs: Any) -> str:
        '''
        from modules.prompts.prompt_string import system_prompt, user_prompt
        [system_prompt, user_prompt, etc.] Pure character prompts are in [modules.prompts.prompt_string.py]
        '''
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        full_template = "{system_prompt}\n\n{relevant_prompt}\n\n{prev_prompt}\n\n{user_prompt}"
        input_prompts = [
            ("system_prompt", system_prompt),
            ("relevant_prompt", relevant_prompt if kwargs.get('relevant_conv') else None_prompt),
            ("prev_prompt", prev_prompt if kwargs.get('prev_conv') else None_prompt),
            ("user_prompt", user_prompt)
        ]

        full_prompt = PromptTemplate.from_template(full_template)
        self.template = PipelinePromptTemplate(
            final_prompt=full_prompt,
            pipeline_prompts=input_prompts
        )
        # return DEFAULT_FORMATTER_MAPPING["f-string"](self.template, **kwargs)
        return self.template.format( **kwargs)