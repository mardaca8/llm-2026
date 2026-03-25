from google import genai
from google.genai import types

api_key = open("LLM/files/gemini.txt").read().strip()

client = genai.Client(api_key=api_key)
model = "models/gemini-2.5-flash"
prompt = "Generate pairs of numbers (x and y coordinates) for points on a circle. Print only coordinates nothing more. You should send me only coordinates in this format no json, only plain text withou this ``` symbols : [(1,2), (2,4)]."

response = client.models.generate_content(
    model=model,
    contents=prompt,
)

import ast

print(response.text)

points = ast.literal_eval(response.text)

print(points)

import matplotlib.pyplot as plt

x = [p[0] for p in points]
y = [p[1] for p in points]

x.append(points[0][0])
y.append(points[0][1])

plt.figure()
plt.plot(x, y)
plt.show()
