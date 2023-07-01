---
license: mit
datasets:
- allenai/soda
- c4
language:
- en
library_name: transformers
pipeline_tag: conversational
---
# What is Cosmo-Tiny?

Inspired by Allen AI's **Cosmo-XL**, **Cosmo-Tiny** is a _very small_ conversational model trained off of the **SODA** dataset. **Cosmo-Tiny** is intended for inference at the edge (on something as small as a 2GB RAM Raspberry Pi).

**Cosmo-Tiny** is trained off of the **t5-small** pretrained model from Google, and is, as a result, is about 2% of the size of the **Cosmo-3B** model.

This is my first SEQ2SEQ NLP Model I've ever made! I'm very excited to share it here on HuggingFace! :)

If you have any questions, or any comments on improvements, please contact me at:  **huggingface.asmit@gmail.com**



# Google Colab Link

Here is the link to the Google Colab file, where I walk through the process of training the model and using the SODA public dataset from AI2.

https://colab.research.google.com/drive/1cx3Yujr_jGQkseqzXZW-2L0vEyEjds_s?usp=sharing

# Get Started With Cosmo-Tiny

Use the code snippet below to get started with Cosmo-Tiny!

```
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("ToddGoldfarb/Cadet-Tiny").to(device)

def set_input(situation_narrative, role_instruction, conversation_history):
    input_text = " <turn> ".join(conversation_history)

    if role_instruction != "":
        input_text = "{} <sep> {}".format(role_instruction, input_text)

    if situation_narrative != "":
        input_text = "{} <sep> {}".format(situation_narrative, input_text)

    return input_text

def generate(situation_narrative, role_instruction, conversation_history):
    """
    situation_narrative: the description of situation/context with the characters included (e.g., "David goes to an amusement park")
    role_instruction: the perspective/speaker instruction (e.g., "Imagine you are David and speak to his friend Sarah").
    conversation_history: the previous utterances in the conversation in a list
    """

    input_text = set_input(situation_narrative, role_instruction, conversation_history)

    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=128, temperature=1.0, top_p=.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return response

situation = "Cosmo had a really fun time participating in the EMNLP conference at Abu Dhabi."
instruction = "You are Cosmo and you are talking to a friend." # You can also leave the instruction empty

conversation = [
    "How was your Trip? "
]

response = generate(situation, instruction, conversation)
print(response)        
```

# Citations and Special Thanks
Special thanks to Hyunwoo Kim for discussing with me the best way to use the SODA dataset. If you haven't looked into their work with SODA, Prosocial-Dialog, or COSMO, I recommend you do so! As well, read the paper on SODA!
The article is listed below.

```
@article{kim2022soda,
    title={SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization},
    author={Hyunwoo Kim and Jack Hessel and Liwei Jiang and Peter West and Ximing Lu and Youngjae Yu and Pei Zhou and Ronan Le Bras and Malihe Alikhani and Gunhee Kim and Maarten Sap and Yejin Choi},
    journal={ArXiv},
    year={2022},
    volume={abs/2212.10465}
}
```
