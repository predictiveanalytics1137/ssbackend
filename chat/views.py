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


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import UploadedFile
from .serializers import UploadedFileSerializer
from django.http import Http404

import boto3
from django.conf import settings


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


import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .models import UploadedFile
from .serializers import UploadedFileSerializer

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        print("Request Data:", request.data)
        print("Request Files:", request.FILES)

        if 'name' not in request.data:
            request.data['name'] = request.FILES['file'].name  # Set the file name as 'name' if not provided

        # Create the serializer instance
        file_serializer = UploadedFileSerializer(data=request.data)

        if file_serializer.is_valid():
            try:
                # Saving file metadata to the database
                file_instance = file_serializer.save()  # Save to DB (saving instance for S3 storage)

                # Manual file upload to S3
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_S3_REGION_NAME
                )
                
                try:
                    # Upload file to S3 bucket
                    s3_key = f"uploads/{request.FILES['file'].name}"
                    response = s3.upload_fileobj(
                        request.FILES['file'],
                        settings.AWS_STORAGE_BUCKET_NAME,
                        s3_key
                    )
                    print("Upload response:", response)
                    
                    # Construct the full S3 URL for the uploaded file
                    full_file_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"
                    
                    # Update the file instance with the full S3 URL
                    file_instance.file_url = full_file_url
                    file_instance.save()  # Save the instance again with the URL
                    
                    # Returning a successful response after successful file upload
                    return Response({
                        'id': file_instance.id,
                        'name': file_instance.name,
                        'file_url': file_instance.file_url,
                        'uploaded_at': file_instance.uploaded_at,
                    }, status=status.HTTP_201_CREATED)

                except NoCredentialsError as e:
                    print("Credentials not available:", e)
                    return Response({'error': 'Invalid AWS credentials'}, status=status.HTTP_403_FORBIDDEN)
                except ClientError as e:
                    print("Error uploading file:", e)
                    return Response({'error': 'Failed to upload file to S3'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
            except Exception as e:
                print("Exception during file_serializer.save():", e)
                return Response({'error': 'Failed to upload file.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        print("Serializer Errors:", file_serializer.errors)
        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)





# File Delete View
# File delete view
class FileDeleteView(APIView):
    def delete(self, request, pk, *args, **kwargs):
        try:
            # Fetch the file instance from the database using the primary key (pk)
            file_instance = UploadedFile.objects.get(pk=pk)
        except UploadedFile.DoesNotExist:
            raise Http404

        # Extract the exact S3 key used during the upload
        s3_key = file_instance.file_url.split('.amazonaws.com/')[1]  # Get key from the full URL

        # Initialize S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME
        )
        
        try:
            # Delete file from S3 bucket
            s3.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_key)
            print(f"Deleted file from S3 bucket: {s3_key}")
        except ClientError as e:
            print(f"Error deleting file from S3: {e}")
            return Response({'error': 'Failed to delete file from S3'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Delete file record from the database
        file_instance.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)


