import time
from openai import OpenAI

import os
DRY_RUN = os.getenv("MEMLISTENER_DRY_RUN", "0") == "1"


class Agent(object):
    def __init__(self,system_prompt: str = None):
        self.system_prompt = system_prompt
    
    """
    ⚠️ 需要自己编辑这个文件，选择合适的大模型调用，返回字符串结果。
    """
    def call_api(self, query: str,top_p: float,temperature: float,max_length: int,llm_model='deepseek-v3.1') -> str:  
        if llm_model in ['deepseek-v3.2-exp', 'deepseek-v3.1', 'Kimi-K2', 'longcat-flash-chat', 'qwen3-235b-a22b', 'deepseek-r1-250528']:
            base_url = "https://api.modelarts-maas.com/v1" # API
            api_key = "" # your Api Key
            client = OpenAI(api_key=api_key, base_url=base_url)

            response = client.chat.completions.create(
                model = llm_model, # DeepSeek-V3, deepseek-v3.1, Kimi-K2, longcat-flash-chat, qwen3-235b-a22b, deepseek-r1-250528
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}, 
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                presence_penalty=1.05,
            )
            return response.choices[0].message.content.strip()
        elif 'gpt' in llm_model or 'gemini' in llm_model:
            base_url = ""
            api_key = ""
            client = OpenAI(api_key=api_key, base_url=base_url)

            response = client.chat.completions.create(
                model = llm_model,
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}, 
                ],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        elif 'Qwen' in llm_model:
            openai_api_key = "EMPTY"
            openai_api_base = "http://127.0.0.1:8100/v1"

            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )

            chat_response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}, 
                ],
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.05
            )
            return chat_response.choices[0].message.content.strip()
                

    def call_llm(self,
             messages: str,
             top_p: float = 0.8,
             temperature: float = 0.7,
             max_length: int = 8192,
             llm_model='deepseek-v3.1'):
        attempt = 0
        max_attempts = 5
        wait_time = 3

        if DRY_RUN:
            # 返回一个“看起来像 LLM 输出”的最小占位内容
            return ""

        while attempt < max_attempts:
            try:
                response = self.call_api(
                    query=messages,
                    top_p=top_p,
                    temperature=temperature,
                    max_length=max_length,
                    llm_model=llm_model
                )
                return response
            except Exception as e:
                print(f"Attempt {attempt+1}: Request failed due to network error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        return "Error: Failed to get response from LLM after multiple attempts."
    
    
    def run(self,
            prompt: str,
            top_p: float = 0.8,
            temperature: float = 0.7,
            max_length: int = 24576,
            max_try: int = 5,
            llm_model='longcat-flash-chat'):
        
        success = False
        try_times = 0

        while try_times < max_try:
            response = self.call_llm(
                messages=prompt,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                llm_model=llm_model
            )
            
            if response != "Error: Failed to get response from LLM after multiple attempts.":
                success = True
                break
            else:
                try_times += 1
        
        return response, success