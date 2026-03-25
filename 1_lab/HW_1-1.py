from google import genai
from google.genai import types

api_key = open("LLM/files/gemini.txt").read().strip()

client = genai.Client(api_key=api_key)
model = "models/gemma-3n-e4b-it"
prompt = "Summarize text to few sentences from attached file"

from pathlib import Path

folder = Path("LLM/1_lab/papers")

from termcolor import colored

sumarized_text = []
num = 0

for file_path in folder.rglob("*.txt"):
    file = client.files.upload(file=file_path)
    
    response = client.models.generate_content(
        model=model,
        contents=[prompt,file],
    )
    num+=1
    print("Sumarized file: " , num)
    print(colored(response.text, "green"))

    sumarized_text.append(response.text)

import json

with open("LLM/1_lab/data.json", "w") as json_file:
    json.dump(sumarized_text, json_file, indent=4)
    








