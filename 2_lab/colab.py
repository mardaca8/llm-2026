# phi-2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained("microsoft/phi-2", trust_remote_code=True)

if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
    config.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                             config=config,
                                             torch_dtype="auto",
                                             device_map="auto",
                                             trust_remote_code=True)

prompts = ["Capital of Lithuania ...", "Capital of France"]

for task in prompts:
    inputs = tokenizer(task, return_tensors="pt", return_attention_mask=True)
    outputs = model.generate(**inputs, max_new_tokens=1000)
    text = tokenizer.batch_decode(outputs)[0]

    print(text.split("<|endoftext|>")[0])
