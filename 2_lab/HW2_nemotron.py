from openrouter import OpenRouter
from Tasks import *

key = open("LLM/files/openrouter.txt").read().strip()

for task in tasks:
    with OpenRouter(
        api_key=key
    ) as client:
        response = client.chat.send(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=[
                {"role": "user", "content": task}
            ]
        )
    
        with open("LLM/llm-2026/2_lab/responses/nemotron.txt", "a") as f:
            f.write(response.choices[0].message.content + '\n'+ "//////////////////////////////////" + '\n')