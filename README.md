---
title: Phi-2 Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
---

# Phi-2 Fine-tuned Assistant Model

This model is a fine-tuned version of Microsoft's Phi-2 model on the OpenAssistant Conversations dataset (OASST1) using QLoRA.

## Model Details

- **Base Model**: microsoft/phi-2
- **Dataset**: OpenAssistant/oasst1
- **Fine-tuning Method**: QLoRA (4-bit Quantized Low-Rank Adaptation)
- **Training Steps**: 500
- **LoRA Parameters**: r=16, alpha=32, dropout=0.05
- **Target Modules**: q_proj, k_proj, v_proj, dense

## Usage

### Loading the model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    trust_remote_code=True,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2", 
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load your adapter
model = PeftModel.from_pretrained(
    base_model, 
    "Rakavi12/Phi2-model"  # Correct model repository
)
```

### Generating Text

```python
# Format input with instruction format
prompt = "Human: Write a short poem about artificial intelligence\n\nAssistant:"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate
outputs = model.generate(
    **inputs,
    max_length=300,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

# Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

The model was fine-tuned using:
- 4-bit quantization with NF4 format
- LoRA for parameter-efficient fine-tuning
- 500 training steps
- Gradient accumulation of 16 steps
- Learning rate of 2e-4


## Images
The following images are included in the `images` folder:
- ![example 1](images/sample 1.png)  
  
- ![example 2](images/sample 2.png)  
