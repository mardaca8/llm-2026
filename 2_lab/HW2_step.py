from openrouter import OpenRouter
from Tasks import *

key = open("LLM/files/openrouter.txt").read().strip()

for task in tasks:
    with OpenRouter(
        api_key=key
    ) as client:
        response = client.chat.send(
            model="stepfun/step-3.5-flash:free",
            messages=[
                {"role": "user", "content": task}
            ]
        )
    
        with open("LLM/llm-2026/2_lab/responses/step.txt", "a") as f:
            f.write(response.choices[0].message.content + '\n' + "//////////////////////////////////" + '\n')