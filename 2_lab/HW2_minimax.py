from openrouter import OpenRouter
from Tasks import *

key = open("LLM/files/openrouter.txt").read().strip()

for task in tasks:
    with OpenRouter(
        api_key=key
    ) as client:
        response = client.chat.send(
            model="minimax/minimax-m2.5:free",
            messages=[
                {"role": "user", "content": task}
            ]
        )
    
        with open("LLM/llm-2026/2_lab/responses/minimax.txt", "a") as f:
            f.write(response.choices[0].message.content + '\n')