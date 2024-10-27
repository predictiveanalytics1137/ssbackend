from django.shortcuts import render

# Create your views here.
from rest_framework.response import Response
from rest_framework.decorators import api_view
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LLaMA model
from rest_framework.response import Response
from rest_framework.decorators import api_view
from transformers import AutoModelForCausalLM, AutoTokenizer
from .models import ChatHistory

# login("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") 

# Load LLaMA model and tokenizer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load your Hugging Face access token
# HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')  # Store token in environment variable
# HUGGING_FACE_TOKEN = "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"  # Store token in environment variable

# # Load LLaMA model and tokenizer with the token
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=HUGGING_FACE_TOKEN)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=HUGGING_FACE_TOKEN)

# @api_view(['POST'])
# def chat_response(request):

#     user_input = request.data.get('message', '')
#     print(user_input)
#     print("user_input")

#     # Tokenize user input
#     # inputs = tokenizer(user_input, return_tensors="pt")

#     # # Generate response from LLaMA model
#     # outputs = model.generate(inputs.input_ids, max_length=100)
#     # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # return Response({"response": response_text})
#     return Response({"response": "hi how are you"})





# api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
hf_token = "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"
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=100, temperature=0.6, top_p=0.8):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@api_view(['POST'])
def chat_response(request):
    print("generating response")
    prompt = request.data.get("message", "")
    if not prompt:
        return Response({"error": "No message provided"}, status=400)

    # Generate response using the model
    response_text = generate_response(prompt)
    return Response({"response": response_text})
