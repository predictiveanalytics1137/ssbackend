# # # # # # # # # # # from django.shortcuts import render

# # # # # # # # # # # # Create your views here.
# # # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # # from rest_framework.decorators import api_view
# # # # # # # # # # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # # # # # # # # # import torch

# # # # # # # # # # # # Load LLaMA model
# # # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # # from rest_framework.decorators import api_view
# # # # # # # # # # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # # # # # # # # # from .models import ChatHistory

# # # # # # # # # # # # login("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") 

# # # # # # # # # # # # Load LLaMA model and tokenizer
# # # # # # # # # # # from rest_framework.decorators import api_view
# # # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # # from transformers import AutoModelForCausalLM, AutoTokenizer
# # # # # # # # # # # import os

# # # # # # # # # # # # Load your Hugging Face access token
# # # # # # # # # # # # HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')  # Store token in environment variable
# # # # # # # # # # # # HUGGING_FACE_TOKEN = "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"  # Store token in environment variable

# # # # # # # # # # # # # Load LLaMA model and tokenizer with the token
# # # # # # # # # # # # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=HUGGING_FACE_TOKEN)
# # # # # # # # # # # # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=HUGGING_FACE_TOKEN)

# # # # # # # # # # # # @api_view(['POST'])
# # # # # # # # # # # # def chat_response(request):

# # # # # # # # # # # #     user_input = request.data.get('message', '')
# # # # # # # # # # # #     print(user_input)
# # # # # # # # # # # #     print("user_input")

# # # # # # # # # # # #     # Tokenize user input
# # # # # # # # # # # #     # inputs = tokenizer(user_input, return_tensors="pt")

# # # # # # # # # # # #     # # Generate response from LLaMA model
# # # # # # # # # # # #     # outputs = model.generate(inputs.input_ids, max_length=100)
# # # # # # # # # # # #     # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # # # # # # # # # # #     # return Response({"response": response_text})
# # # # # # # # # # # #     return Response({"response": "hi how are you"})


# # # # # # # # # # # from rest_framework.views import APIView
# # # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # # from rest_framework import status
# # # # # # # # # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # # # # # # # # from .models import UploadedFile
# # # # # # # # # # # from .serializers import UploadedFileSerializer
# # # # # # # # # # # from django.http import Http404
# # # # # # # # # # # import boto3
# # # # # # # # # # # from botocore.exceptions import NoCredentialsError, ClientError
# # # # # # # # # # # from django.conf import settings
# # # # # # # # # # # import boto3
# # # # # # # # # # # import pandas as pd
# # # # # # # # # # # from io import StringIO
# # # # # # # # # # # # api/views.py
# # # # # # # # # # # from rest_framework.decorators import api_view
# # # # # # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # # # # # # # import torch

# # # # # # # # # # # # Load the model and tokenizer
# # # # # # # # # # # hf_token = "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"
# # # # # # # # # # # model_name = "meta-llama/Llama-3.2-1B"
# # # # # # # # # # # tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
# # # # # # # # # # # model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

# # # # # # # # # # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # # model.to(device)

# # # # # # # # # # # # def generate_response(prompt, max_length=150, temperature=0.6, top_p=0.8):
# # # # # # # # # # # #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
# # # # # # # # # # # #     output = model.generate(
# # # # # # # # # # # #         **inputs,
# # # # # # # # # # # #         max_length=max_length,
# # # # # # # # # # # #         temperature=temperature,
# # # # # # # # # # # #         top_p=top_p,
# # # # # # # # # # # #         do_sample=True
# # # # # # # # # # # #     )
# # # # # # # # # # # #     response = tokenizer.decode(output[0], skip_special_tokens=True)
# # # # # # # # # # # #     return response

# # # # # # # # # # # # @api_view(['POST'])
# # # # # # # # # # # # def chat_response(request):
# # # # # # # # # # # #     print("generating response")
# # # # # # # # # # # #     prompt = request.data.get("message", "")
# # # # # # # # # # # #     if not prompt:
# # # # # # # # # # # #         return Response({"error": "No message provided"}, status=400)

# # # # # # # # # # # #     # Generate response using the model
# # # # # # # # # # # #     response_text = generate_response(prompt)
# # # # # # # # # # # #     return Response({"response": response_text})


# # # # # # # # # # # from rest_framework.decorators import api_view
# # # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # # import torch
# # # # # # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM

# # # # # # # # # # # # Load the model and tokenizer
# # # # # # # # # # # hf_token = "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"
# # # # # # # # # # # model_name = "meta-llama/Llama-3.2-1B"
# # # # # # # # # # # tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# # # # # # # # # # # model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# # # # # # # # # # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # # model.to(device)

# # # # # # # # # # # question_flow = [
# # # # # # # # # # #     {
# # # # # # # # # # #         "id": 1,
# # # # # # # # # # #         "question": "What is the subject of your prediction (e.g., customers, users, products)?",
# # # # # # # # # # #         "key": "subject",
# # # # # # # # # # #         "next_question_id": 2
# # # # # # # # # # #     },
# # # # # # # # # # #     {
# # # # # # # # # # #         "id": 2,
# # # # # # # # # # #         "question": "What is the target value you want to predict for each customer?",
# # # # # # # # # # #         "key": "target_value",
# # # # # # # # # # #         "next_question_id": 3
# # # # # # # # # # #     },
# # # # # # # # # # #     {
# # # # # # # # # # #         "id": 3,
# # # # # # # # # # #         "question": "Is there a specific future time horizon relevant to the prediction (e.g., next 7 days)?",
# # # # # # # # # # #         "key": "time_horizon",
# # # # # # # # # # #         "next_question_id": 4
# # # # # # # # # # #     },
# # # # # # # # # # #     {
# # # # # # # # # # #         "id": 4,
# # # # # # # # # # #         "question": "Do you have a unique identifier column for each customer in your data?",
# # # # # # # # # # #         "key": "unique_identifier",
# # # # # # # # # # #         "next_question_id": None  # End of flow
# # # # # # # # # # #     }
# # # # # # # # # # # ]

# # # # # # # # # # # # In-memory storage for simplicity (use a database for production)
# # # # # # # # # # # user_sessions = {}

# # # # # # # # # # # def generate_response(prompt, max_length=150, temperature=0.6, top_p=0.8):
# # # # # # # # # # #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
# # # # # # # # # # #     output = model.generate(
# # # # # # # # # # #         **inputs,
# # # # # # # # # # #         max_length=max_length,
# # # # # # # # # # #         temperature=temperature,
# # # # # # # # # # #         top_p=top_p,
# # # # # # # # # # #         do_sample=True
# # # # # # # # # # #     )
# # # # # # # # # # #     response = tokenizer.decode(output[0], skip_special_tokens=True)
# # # # # # # # # # #     return response

# # # # # # # # # # # @api_view(['POST'])
# # # # # # # # # # # def chat_response(request):
# # # # # # # # # # #     user_id = request.data.get("user_id", "default_user")  # Track user session
# # # # # # # # # # #     message = request.data.get("message", "")

# # # # # # # # # # #     # Initialize session if new user
# # # # # # # # # # #     if user_id not in user_sessions:
# # # # # # # # # # #         user_sessions[user_id] = {
# # # # # # # # # # #             "current_question_id": 1,
# # # # # # # # # # #             "answers": {}
# # # # # # # # # # #         }

# # # # # # # # # # #     session = user_sessions[user_id]
# # # # # # # # # # #     current_question_id = session["current_question_id"]

# # # # # # # # # # #     # Handle mandatory question flow
# # # # # # # # # # #     if current_question_id:
# # # # # # # # # # #         current_question = next(q for q in question_flow if q["id"] == current_question_id)
# # # # # # # # # # #         key = current_question["key"]
# # # # # # # # # # #         session["answers"][key] = message  # Save the user's answer
# # # # # # # # # # #         session["current_question_id"] = current_question["next_question_id"]  # Move to the next question

# # # # # # # # # # #         # Check if there's a next question
# # # # # # # # # # #         if session["current_question_id"]:
# # # # # # # # # # #             next_question = next(q for q in question_flow if q["id"] == session["current_question_id"])
# # # # # # # # # # #             return Response({"response": next_question["question"], "mandatory": True})
# # # # # # # # # # #         else:
# # # # # # # # # # #             return Response({"response": "Thank you for answering all mandatory questions. Feel free to ask general questions now.", "mandatory": False})

# # # # # # # # # # #     # Handle general interaction with LLaMA
# # # # # # # # # # #     response_text = generate_response(message)
# # # # # # # # # # #     return Response({"response": response_text, "mandatory": False})






# # # # # # # # # # # class FileUploadView(APIView):
# # # # # # # # # # #     parser_classes = (MultiPartParser, FormParser)

# # # # # # # # # # #     def post(self, request, *args, **kwargs):
# # # # # # # # # # #         if 'file' not in request.FILES:
# # # # # # # # # # #             return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # # #         uploaded_file = request.FILES['file']

# # # # # # # # # # #         # Validate the file size
# # # # # # # # # # #         if uploaded_file.size == 0:
# # # # # # # # # # #             return Response({'error': 'File is empty'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # # #         # Validate the file type
# # # # # # # # # # #         if not uploaded_file.name.endswith('.csv'):
# # # # # # # # # # #             return Response({'error': 'Only CSV files are allowed'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # # #         # Create a serializer instance
# # # # # # # # # # #         file_serializer = UploadedFileSerializer(data={'name': uploaded_file.name, 'file': uploaded_file})

# # # # # # # # # # #         if file_serializer.is_valid():
# # # # # # # # # # #             try:
# # # # # # # # # # #                 # Save metadata to the database
# # # # # # # # # # #                 file_instance = file_serializer.save()

# # # # # # # # # # #                 # Upload file to S3
# # # # # # # # # # #                 s3 = boto3.client(
# # # # # # # # # # #                     's3',
# # # # # # # # # # #                     aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
# # # # # # # # # # #                     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
# # # # # # # # # # #                     region_name=settings.AWS_S3_REGION_NAME
# # # # # # # # # # #                 )

# # # # # # # # # # #                 s3_key = f"uploads/{uploaded_file.name}"  # Define the S3 key
# # # # # # # # # # #                 uploaded_file.seek(0)  # Reset file pointer
# # # # # # # # # # #                 s3.upload_fileobj(
# # # # # # # # # # #                     uploaded_file,
# # # # # # # # # # #                     settings.AWS_STORAGE_BUCKET_NAME,
# # # # # # # # # # #                     s3_key
# # # # # # # # # # #                 )

# # # # # # # # # # #                 # Generate the full S3 URL
# # # # # # # # # # #                 full_file_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"

# # # # # # # # # # #                 print("full_file_url")
# # # # # # # # # # #                 # Update the file instance with the S3 URL
# # # # # # # # # # #                 file_instance.file_url = full_file_url
# # # # # # # # # # #                 file_instance.save()
# # # # # # # # # # #                 print("print saved file instance")

# # # # # # # # # # #                 # Validate the CSV and extract schema
# # # # # # # # # # #                 obj = s3.get_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_key)
# # # # # # # # # # #                 csv_content = obj['Body'].read().decode('utf-8')
# # # # # # # # # # #                 df = pd.read_csv(StringIO(csv_content))
# # # # # # # # # # #                 print(df.head(5))

# # # # # # # # # # #                 # Generate schema
# # # # # # # # # # #                 data_schema = pd.DataFrame({
# # # # # # # # # # #                     'Column Name': df.columns,
# # # # # # # # # # #                     'Data Type': [str(dtype) for dtype in df.dtypes]
# # # # # # # # # # #                 })

# # # # # # # # # # #                 # Map data types to user-friendly names
# # # # # # # # # # #                 data_schema['Data Type'] = data_schema['Data Type'].replace({
# # # # # # # # # # #                     'int64': 'Long',
# # # # # # # # # # #                     'float64': 'Double',
# # # # # # # # # # #                     'object': 'String',
# # # # # # # # # # #                     'bool': 'Boolean'
# # # # # # # # # # #                 })

# # # # # # # # # # #                 schema_result = data_schema.to_dict(orient='records')
# # # # # # # # # # #                 # print(schema_result)
# # # # # # # # # # #                 # print("schema_result")

# # # # # # # # # # #                 # Return success response
# # # # # # # # # # #                 return Response({
# # # # # # # # # # #                     'id': file_instance.id,
# # # # # # # # # # #                     'name': file_instance.name,
# # # # # # # # # # #                     'file_url': file_instance.file_url,
# # # # # # # # # # #                     'uploaded_at': file_instance.uploaded_at,
# # # # # # # # # # #                     'schema': schema_result
# # # # # # # # # # #                 }, status=status.HTTP_201_CREATED)

# # # # # # # # # # #             except NoCredentialsError:
# # # # # # # # # # #                 return Response({'error': 'AWS credentials not available'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # # # # # #             except ClientError as e:
# # # # # # # # # # #                 return Response({'error': f'S3 upload failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # # # # # #             except Exception as e:
# # # # # # # # # # #                 return Response({'error': f'File validation failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # # #         return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




# # # # # # # # # # # # File Delete View
# # # # # # # # # # # # File delete view
# # # # # # # # # # # class FileDeleteView(APIView):
# # # # # # # # # # #     def delete(self, request, pk, *args, **kwargs):
# # # # # # # # # # #         try:
# # # # # # # # # # #             # Fetch the file instance from the database using the primary key (pk)
# # # # # # # # # # #             file_instance = UploadedFile.objects.get(pk=pk)
# # # # # # # # # # #         except UploadedFile.DoesNotExist:
# # # # # # # # # # #             raise Http404

# # # # # # # # # # #         # Extract the exact S3 key used during the upload
# # # # # # # # # # #         s3_key = file_instance.file_url.split('.amazonaws.com/')[1]  # Get key from the full URL

# # # # # # # # # # #         # Initialize S3 client
# # # # # # # # # # #         s3 = boto3.client(
# # # # # # # # # # #             's3',
# # # # # # # # # # #             aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
# # # # # # # # # # #             aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
# # # # # # # # # # #             region_name=settings.AWS_S3_REGION_NAME
# # # # # # # # # # #         )
        
# # # # # # # # # # #         try:
# # # # # # # # # # #             # Delete file from S3 bucket
# # # # # # # # # # #             s3.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_key)
# # # # # # # # # # #             print(f"Deleted file from S3 bucket: {s3_key}")
# # # # # # # # # # #         except ClientError as e:
# # # # # # # # # # #             print(f"Error deleting file from S3: {e}")
# # # # # # # # # # #             return Response({'error': 'Failed to delete file from S3'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # # # # # # # # #         # Delete file record from the database
# # # # # # # # # # #         file_instance.delete()

# # # # # # # # # # #         return Response(status=status.HTTP_204_NO_CONTENT)

# # # # # # # # # # import os
# # # # # # # # # # from rest_framework.decorators import api_view
# # # # # # # # # # from django.shortcuts import render
# # # # # # # # # # from rest_framework.views import APIView
# # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # from rest_framework import status
# # # # # # # # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # # # # # # # from .models import UploadedFile, FileSchema
# # # # # # # # # # from .serializers import UploadedFileSerializer
# # # # # # # # # # from django.http import Http404
# # # # # # # # # # import boto3
# # # # # # # # # # from botocore.exceptions import NoCredentialsError, ClientError
# # # # # # # # # # from django.conf import settings
# # # # # # # # # # import pandas as pd
# # # # # # # # # # from io import BytesIO, StringIO
# # # # # # # # # # from transformers import AutoTokenizer, AutoModelForCausalLM
# # # # # # # # # # import torch

# # # # # # # # # # from chat import serializers

# # # # # # # # # # # Load the LLaMA model and tokenizer
# # # # # # # # # # hf_token = "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"
# # # # # # # # # # model_name = "meta-llama/Llama-3.2-1B"
# # # # # # # # # # tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
# # # # # # # # # # model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

# # # # # # # # # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # # # # # # # # model.to(device)

# # # # # # # # # # # Define the question flow
# # # # # # # # # # question_flow = [
# # # # # # # # # #     {
# # # # # # # # # #         "id": 1,
# # # # # # # # # #         "question": "What is the subject of your prediction (e.g., customers, users, products)?",
# # # # # # # # # #         "key": "subject",
# # # # # # # # # #         "next_question_id": 2,
# # # # # # # # # #         "expected_type": str
# # # # # # # # # #     },
# # # # # # # # # #     {
# # # # # # # # # #         "id": 2,
# # # # # # # # # #         "question": "What is the target value you want to predict for each subject?",
# # # # # # # # # #         "key": "target_value",
# # # # # # # # # #         "next_question_id": 3,
# # # # # # # # # #         "expected_type": str
# # # # # # # # # #     },
# # # # # # # # # #     {
# # # # # # # # # #         "id": 3,
# # # # # # # # # #         "question": "Is there a specific future time horizon relevant to the prediction (e.g., next 7 days)?",
# # # # # # # # # #         "key": "time_horizon",
# # # # # # # # # #         "next_question_id": 4,
# # # # # # # # # #         "expected_type": str
# # # # # # # # # #     },
# # # # # # # # # #     {
# # # # # # # # # #         "id": 4,
# # # # # # # # # #         "question": "Do you have a unique identifier column for each subject in your data?",
# # # # # # # # # #         "key": "unique_identifier",
# # # # # # # # # #         "next_question_id": None,  # End of flow
# # # # # # # # # #         "expected_type": str
# # # # # # # # # #     }
# # # # # # # # # # ]

# # # # # # # # # # # In-memory storage for simplicity (use a database for production)
# # # # # # # # # # user_sessions = {}

# # # # # # # # # # def generate_response(prompt, max_length=150, temperature=0.6, top_p=0.8):
# # # # # # # # # #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
# # # # # # # # # #     output = model.generate(
# # # # # # # # # #         **inputs,
# # # # # # # # # #         max_length=max_length,
# # # # # # # # # #         temperature=temperature,
# # # # # # # # # #         top_p=top_p,
# # # # # # # # # #         do_sample=True
# # # # # # # # # #     )
# # # # # # # # # #     response = tokenizer.decode(output[0], skip_special_tokens=True)
# # # # # # # # # #     return response

# # # # # # # # # # @api_view(['POST'])
# # # # # # # # # # def chat_response(request):
# # # # # # # # # #     user_id = request.data.get("user_id", "default_user")  # Track user session
# # # # # # # # # #     message = request.data.get("message", "")

# # # # # # # # # #     # Initialize session if new user
# # # # # # # # # #     if user_id not in user_sessions:
# # # # # # # # # #         user_sessions[user_id] = {
# # # # # # # # # #             "current_question_id": 1,
# # # # # # # # # #             "answers": {}
# # # # # # # # # #         }

# # # # # # # # # #     session = user_sessions[user_id]
# # # # # # # # # #     current_question_id = session["current_question_id"]

# # # # # # # # # #     # Check if the user wants to proceed with the question flow or have a general conversation
# # # # # # # # # #     if current_question_id:
# # # # # # # # # #         current_question = next(q for q in question_flow if q["id"] == current_question_id)
# # # # # # # # # #         key = current_question["key"]
# # # # # # # # # #         expected_type = current_question["expected_type"]

# # # # # # # # # #         # Validate the user's answer
# # # # # # # # # #         try:
# # # # # # # # # #             value = expected_type(message)
# # # # # # # # # #             session["answers"][key] = value  # Save the user's answer
# # # # # # # # # #             session["current_question_id"] = current_question["next_question_id"]  # Move to the next question
# # # # # # # # # #         except ValueError:
# # # # # # # # # #             return Response({"response": f"Please provide a valid {expected_type.__name__} value for the question."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         # Check if there's a next question
# # # # # # # # # #         if session["current_question_id"]:
# # # # # # # # # #             next_question = next(q for q in question_flow if q["id"] == session["current_question_id"])
# # # # # # # # # #             return Response({"response": next_question["question"], "mandatory": True})
# # # # # # # # # #         else:
# # # # # # # # # #             # Process the user's answers and generate the predictive model
# # # # # # # # # #             subject = session["answers"]["subject"]
# # # # # # # # # #             target_value = session["answers"]["target_value"]
# # # # # # # # # #             time_horizon = session["answers"]["time_horizon"]
# # # # # # # # # #             unique_identifier = session["answers"]["unique_identifier"]

# # # # # # # # # #             # Use the gathered information to create a predictive model
# # # # # # # # # #             # and provide the user with the necessary next steps
# # # # # # # # # #             response_text = f"Thank you for answering all mandatory questions. Based on your inputs, we will create a predictive model to forecast {target_value} for your {subject} data. The next steps are: 1) Upload your CSV file containing the relevant data columns. 2) Review the data schema and make any necessary adjustments. 3) Train the model and evaluate its performance."
# # # # # # # # # #             return Response({"response": response_text, "mandatory": False})
# # # # # # # # # #     else:
# # # # # # # # # #         # Handle general interaction with LLaMA
# # # # # # # # # #         response_text = generate_response(message)
# # # # # # # # # #         return Response({"response": response_text, "mandatory": False})


# # # # # # # # # # import boto3
# # # # # # # # # # import os
# # # # # # # # # # from io import BytesIO
# # # # # # # # # # import pandas as pd
# # # # # # # # # # import datetime
# # # # # # # # # # from rest_framework.views import APIView
# # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # from rest_framework import status
# # # # # # # # # # from botocore.exceptions import NoCredentialsError, ClientError
# # # # # # # # # # from .serializers import UploadedFileSerializer
# # # # # # # # # # from .models import UploadedFile, FileSchema


# # # # # # # # # # class FileUploadView(APIView):
# # # # # # # # # #     def post(self, request, *args, **kwargs):
# # # # # # # # # #         print("File upload request received.")

# # # # # # # # # #         # Step 1: Validate file presence
# # # # # # # # # #         if 'file' not in request.FILES:
# # # # # # # # # #             return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         uploaded_file = request.FILES['file']
# # # # # # # # # #         print(f"Received file: {uploaded_file.name} ({uploaded_file.size} bytes)")

# # # # # # # # # #         # Step 2: Validate file size
# # # # # # # # # #         if uploaded_file.size == 0:
# # # # # # # # # #             return Response({'error': 'File is empty'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         # Step 3: Validate file format
# # # # # # # # # #         file_format = uploaded_file.name.split('.')[-1].lower()
# # # # # # # # # #         if file_format not in ['csv', 'xlsx']:
# # # # # # # # # #             return Response({'error': 'Unsupported file format. Only CSV and Excel are allowed.'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         try:
# # # # # # # # # #             # Step 4: Load file into Pandas
# # # # # # # # # #             if file_format == 'csv':
# # # # # # # # # #                 df = pd.read_csv(uploaded_file, low_memory=False)
# # # # # # # # # #             elif file_format == 'xlsx':
# # # # # # # # # #                 df = pd.read_excel(uploaded_file)

# # # # # # # # # #             # Normalize column headers
# # # # # # # # # #             df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # # # # # #             print(f"DataFrame columns: {df.columns.tolist()}")

# # # # # # # # # #             # Step 5: Improved Data Type Inference
# # # # # # # # # #             def infer_column_dtype(series):
# # # # # # # # # #                 """
# # # # # # # # # #                 Infers the correct data type for a column by handling mixed types.
# # # # # # # # # #                 """
# # # # # # # # # #                 series = series.dropna().astype(str).str.strip()  # Handle mixed types and strip whitespace

# # # # # # # # # #                 # Try boolean
# # # # # # # # # #                 boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # # # # # #                 unique_values = set(series.str.lower().unique())
# # # # # # # # # #                 if unique_values.issubset(boolean_values):
# # # # # # # # # #                     return "boolean"

# # # # # # # # # #                 # Try integer
# # # # # # # # # #                 try:
# # # # # # # # # #                     int_series = pd.to_numeric(series, errors='raise')
# # # # # # # # # #                     if (int_series % 1 == 0).all():
# # # # # # # # # #                         int_min = int_series.min()
# # # # # # # # # #                         int_max = int_series.max()
# # # # # # # # # #                         if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # # # # # #                             return "int"
# # # # # # # # # #                         else:
# # # # # # # # # #                             return "bigint"
# # # # # # # # # #                 except ValueError:
# # # # # # # # # #                     pass

# # # # # # # # # #                 # Try double
# # # # # # # # # #                 try:
# # # # # # # # # #                     pd.to_numeric(series, errors='raise', downcast='float')
# # # # # # # # # #                     return "double"
# # # # # # # # # #                 except ValueError:
# # # # # # # # # #                     pass

# # # # # # # # # #                 # Default to string
# # # # # # # # # #                 return "string"

# # # # # # # # # #             # Infer schema
# # # # # # # # # #             schema = [
# # # # # # # # # #                 {
# # # # # # # # # #                     "column_name": col,
# # # # # # # # # #                     "data_type": infer_column_dtype(df[col])
# # # # # # # # # #                 }
# # # # # # # # # #                 for col in df.columns
# # # # # # # # # #             ]
# # # # # # # # # #             print(f"Inferred schema: {schema}")

# # # # # # # # # #             # **New Step: Convert Boolean Columns to 'true'/'false' Strings**
# # # # # # # # # #             boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # # # # # #             for col in boolean_columns:
# # # # # # # # # #                 df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # # # # # #                 df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # # # # # #             print(f"Boolean columns converted: {boolean_columns}")

# # # # # # # # # #             # Step 6: Handle Duplicate Files Dynamically
# # # # # # # # # #             file_name_base, file_extension = os.path.splitext(uploaded_file.name)
# # # # # # # # # #             file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # # # # # #             existing_file = UploadedFile.objects.filter(name=uploaded_file.name).first()
# # # # # # # # # #             if existing_file:
# # # # # # # # # #                 timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # # # # #                 new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # # # # # #                 uploaded_file.name = new_file_name
# # # # # # # # # #                 print(f"Duplicate file detected. Renaming file to: {new_file_name}")
# # # # # # # # # #             else:
# # # # # # # # # #                 print(f"File name is unique: {uploaded_file.name}")

# # # # # # # # # #             # Step 7: Save Metadata to Database
# # # # # # # # # #             uploaded_file.seek(0)  # Reset file pointer before saving
# # # # # # # # # #             file_serializer = UploadedFileSerializer(data={'name': uploaded_file.name, 'file': uploaded_file})
# # # # # # # # # #             if file_serializer.is_valid():
# # # # # # # # # #                 file_instance = file_serializer.save()

# # # # # # # # # #                 # Step 8: Convert DataFrame to CSV and Upload to S3
# # # # # # # # # #                 csv_buffer = BytesIO()
# # # # # # # # # #                 df.to_csv(csv_buffer, index=False)
# # # # # # # # # #                 csv_buffer.seek(0)
# # # # # # # # # #                 s3_file_name = os.path.splitext(uploaded_file.name)[0] + '.csv'
# # # # # # # # # #                 file_key = f"uploads/{s3_file_name}"

# # # # # # # # # #                 s3 = boto3.client(
# # # # # # # # # #                     's3',
# # # # # # # # # #                     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# # # # # # # # # #                     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
# # # # # # # # # #                     region_name=os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # #                 )

# # # # # # # # # #                 s3.upload_fileobj(csv_buffer, os.getenv('AWS_STORAGE_BUCKET_NAME'), file_key)

# # # # # # # # # #                 # Generate file URL
# # # # # # # # # #                 file_url = f"https://{os.getenv('AWS_STORAGE_BUCKET_NAME')}.s3.{os.getenv('AWS_S3_REGION_NAME')}.amazonaws.com/{file_key}"
# # # # # # # # # #                 file_instance.file_url = file_url
# # # # # # # # # #                 file_instance.save()

# # # # # # # # # #                 # Step 9: Save Schema to Database
# # # # # # # # # #                 FileSchema.objects.create(file=file_instance, schema=schema)

# # # # # # # # # #                 # Step 10: Trigger AWS Glue Table Update
# # # # # # # # # #                 self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # # # # # #                 # Step 11: Return Response
# # # # # # # # # #                 return Response({
# # # # # # # # # #                     'id': file_instance.id,
# # # # # # # # # #                     'name': file_instance.name,
# # # # # # # # # #                     'file_url': file_instance.file_url,
# # # # # # # # # #                     'uploaded_at': file_instance.uploaded_at,
# # # # # # # # # #                     'schema': schema
# # # # # # # # # #                 }, status=status.HTTP_201_CREATED)
# # # # # # # # # #             else:
# # # # # # # # # #                 return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         except pd.errors.EmptyDataError:
# # # # # # # # # #             return Response({'error': 'File content is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # # # # # #         except NoCredentialsError:
# # # # # # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # # # # #         except ClientError as e:
# # # # # # # # # #             return Response({'error': f'S3 upload failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # # # # #         except Exception as e:
# # # # # # # # # #             print(f"Unexpected error: {str(e)}")
# # # # # # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #     def trigger_glue_update(self, table_name, schema, file_key):
# # # # # # # # # #         """
# # # # # # # # # #         Dynamically update AWS Glue table schema.
# # # # # # # # # #         """
# # # # # # # # # #         glue = boto3.client(
# # # # # # # # # #             'glue',
# # # # # # # # # #             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# # # # # # # # # #             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
# # # # # # # # # #             region_name=os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # #         )
# # # # # # # # # #         s3_location = f"s3://{os.getenv('AWS_STORAGE_BUCKET_NAME')}/uploads/"
# # # # # # # # # #         storage_descriptor = {
# # # # # # # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # # # # # # #             'Location': s3_location,
# # # # # # # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # # # # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # # # # # # #             'SerdeInfo': {
# # # # # # # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # # # # # # #                 'Parameters': {
# # # # # # # # # #                     'field.delim': ',',
# # # # # # # # # #                     'skip.header.line.count': '1'
# # # # # # # # # #                 }
# # # # # # # # # #             }
# # # # # # # # # #         }
# # # # # # # # # #         try:
# # # # # # # # # #             glue.update_table(
# # # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # # #                 TableInput={
# # # # # # # # # #                     'Name': table_name,
# # # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # # #                 }
# # # # # # # # # #             )
# # # # # # # # # #             print(f"Glue table '{table_name}' updated successfully.")
# # # # # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # # # # #             print(f"Table '{table_name}' not found. Creating a new table...")
# # # # # # # # # #             glue.create_table(
# # # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # # #                 TableInput={
# # # # # # # # # #                     'Name': table_name,
# # # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # # #                 }
# # # # # # # # # #             )
# # # # # # # # # #             print(f"Glue table '{table_name}' created successfully.")
# # # # # # # # # #         except Exception as e:
# # # # # # # # # #             print(f"Glue operation failed: {str(e)}")



# # # # # # # # # # class FileDeleteView(APIView):
# # # # # # # # # #     def delete(self, request, pk, *args, **kwargs):
# # # # # # # # # #         try:
# # # # # # # # # #             # Fetch the file instance from the database using the primary key (pk)
# # # # # # # # # #             file_instance = UploadedFile.objects.get(pk=pk)
# # # # # # # # # #         except UploadedFile.DoesNotExist:
# # # # # # # # # #             raise Http404

# # # # # # # # # #         # Extract the exact S3 key used during the upload
# # # # # # # # # #         file_url = file_instance.file_url
# # # # # # # # # #         s3_bucket_url = f"https://{os.getenv('AWS_STORAGE_BUCKET_NAME')}.s3.{os.getenv('AWS_S3_REGION_NAME')}.amazonaws.com/"
# # # # # # # # # #         s3_key = file_url.replace(s3_bucket_url, '')

# # # # # # # # # #         # Initialize S3 client
# # # # # # # # # #         s3 = boto3.client(
# # # # # # # # # #             's3',
# # # # # # # # # #             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# # # # # # # # # #             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
# # # # # # # # # #             region_name=os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # #         )

# # # # # # # # # #         try:
# # # # # # # # # #             # Delete file from S3 bucket
# # # # # # # # # #             s3.delete_object(Bucket=os.getenv('AWS_STORAGE_BUCKET_NAME'), Key=s3_key)
# # # # # # # # # #             print(f"Deleted file from S3 bucket: {s3_key}")
# # # # # # # # # #         except ClientError as e:
# # # # # # # # # #             print(f"Error deleting file from S3: {e}")
# # # # # # # # # #             return Response({'error': 'Failed to delete file from S3'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # # # # # # # #         # Delete the associated AWS Glue table
# # # # # # # # # #         glue = boto3.client(
# # # # # # # # # #             'glue',
# # # # # # # # # #             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# # # # # # # # # #             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
# # # # # # # # # #             region_name=os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # #         )

# # # # # # # # # #         # The table name should match the one used during upload
# # # # # # # # # #         # Assuming table_name is derived from the file name base
# # # # # # # # # #         file_name_base = os.path.splitext(file_instance.name)[0].lower().replace(' ', '_')
# # # # # # # # # #         try:
# # # # # # # # # #             glue.delete_table(
# # # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # # #                 Name=file_name_base
# # # # # # # # # #             )
# # # # # # # # # #             print(f"Deleted Glue table: {file_name_base}")
# # # # # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # # # # #             print(f"Glue table '{file_name_base}' not found. Nothing to delete.")
# # # # # # # # # #         except Exception as e:
# # # # # # # # # #             print(f"Error deleting Glue table: {e}")
# # # # # # # # # #             return Response({'error': 'Failed to delete Glue table'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # # # # # # # #         # Delete file schema from the database
# # # # # # # # # #         try:
# # # # # # # # # #             FileSchema.objects.filter(file=file_instance).delete()
# # # # # # # # # #             print(f"Deleted file schema for file ID: {file_instance.id}")
# # # # # # # # # #         except Exception as e:
# # # # # # # # # #             print(f"Error deleting file schema from the database: {e}")
# # # # # # # # # #             return Response({'error': 'Failed to delete file schema from the database'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # # # # # # # #         # Delete file record from the database
# # # # # # # # # #         file_instance.delete()
# # # # # # # # # #         print(f"Deleted file record from the database: {file_instance.name}")

# # # # # # # # # #         return Response(status=status.HTTP_204_NO_CONTENT)



# # # # # # # ===========================
# # # # # # # Below Merge code.
# # # # # # # ===========================





# # # # Chat/views.py

# # # import os
# # # import datetime
# # # from io import BytesIO
# # # from typing import Any, Dict, List

# # # import boto3
# # # import pandas as pd
# # # import openai
# # # from botocore.exceptions import ClientError, NoCredentialsError
# # # from django.conf import settings
# # # from rest_framework import status
# # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser  # Import JSONParser
# # # from rest_framework.response import Response
# # # from rest_framework.views import APIView
# # # from langchain.chains import ConversationChain
# # # from langchain.chat_models import ChatOpenAI
# # # from langchain.memory import ConversationBufferMemory
# # # from langchain.prompts import PromptTemplate

# # # from .models import FileSchema, UploadedFile
# # # from .serializers import UploadedFileSerializer

# # # # ===========================
# # # # AWS Configuration
# # # # ===========================
# # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')

# # # # ===========================
# # # # OpenAI Configuration
# # # # ===========================
# # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Ensure this is set in your environment

# # # # ===========================
# # # # Initialize OpenAI LangChain model for ChatGPT
# # # # ===========================
# # # openai.api_key = OPENAI_API_KEY

# # # llm_chatgpt = ChatOpenAI(
# # #     model="gpt-4",
# # #     temperature=0.7,
# # #     openai_api_key=OPENAI_API_KEY,
# # # )

# # # # LangChain prompt with memory integration for ChatGPT
# # # prompt_chatgpt = PromptTemplate(
# # #     input_variables=["history", "user_input"],
# # #     template=(
# # #         "You are a helpful AI assistant. You guide users through defining predictive questions and refining goals.\n"
# # #         "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
# # #         "Steps:\n"
# # #         "1. Discuss the Subject they want to predict.\n"
# # #         "2. Confirm the Target Value they want to predict.\n"
# # #         "3. Check if there's a specific time frame for the prediction.\n"
# # #         "4. Reference the dataset schema if available.\n"
# # #         "5. Summarize inputs before proceeding to model creation.\n\n"
# # #         "Conversation history: {history}\n"
# # #         "User input: {user_input}\n"
# # #         "Assistant:"
# # #     ),
# # # )

# # # memory_chatgpt = ConversationBufferMemory(
# # #     memory_key="history",
# # #     input_key="user_input",
# # # )

# # # conversation_chain_chatgpt = ConversationChain(
# # #     llm=llm_chatgpt,
# # #     prompt=prompt_chatgpt,
# # #     memory=memory_chatgpt,
# # #     input_key="user_input",
# # # )

# # # # ===========================
# # # # State Management for ChatGPT
# # # # ===========================
# # # class ChatGPTStateMachine:
# # #     """
# # #     Manages the conversation state for ChatGPT-based interactions.
# # #     """
# # #     def __init__(self):
# # #         self.state = "INITIAL"
# # #         self.answers = {}
# # #         self.dataset_schema = None  # Store the dataset schema

# # #     def next_state(self, user_input: str) -> str:
# # #         """
# # #         Transition to the next state based on the current state and user input.
# # #         """
# # #         if self.state == "INITIAL":
# # #             self.state = "DEFINE_SUBJECT"
# # #         elif self.state == "DEFINE_SUBJECT":
# # #             self.state = "DEFINE_TARGET"
# # #         elif self.state == "DEFINE_TARGET":
# # #             self.state = "ASK_TIME_ELEMENT"
# # #         elif self.state == "ASK_TIME_ELEMENT":
# # #             self.state = "CONFIRMATION"
# # #         elif self.state == "CONFIRMATION":
# # #             self.state = "COMPLETE"
# # #         return self.state

# # #     def reset(self):
# # #         """
# # #         Reset the conversation state to initial.
# # #         """
# # #         self.state = "INITIAL"
# # #         self.answers = {}
# # #         self.dataset_schema = None

# # # # Initialize ChatGPT state machine
# # # chatgpt_state_machine = ChatGPTStateMachine()

# # # # ===========================
# # # # Utility Functions
# # # # ===========================
# # # def get_s3_client():
# # #     """
# # #     Creates and returns an AWS S3 client.
# # #     """
# # #     return boto3.client(
# # #         's3',
# # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # #         region_name=AWS_S3_REGION_NAME
# # #     )

# # # def get_glue_client():
# # #     """
# # #     Creates and returns an AWS Glue client.
# # #     """
# # #     return boto3.client(
# # #         'glue',
# # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # #         region_name=AWS_S3_REGION_NAME
# # #     )

# # # def infer_column_dtype(series: pd.Series) -> str:
# # #     """
# # #     Infers the correct data type for a column by handling mixed types.
# # #     """
# # #     series = series.dropna().astype(str).str.strip()  # Handle mixed types and strip whitespace

# # #     # Try boolean
# # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # #     unique_values = set(series.str.lower().unique())
# # #     if unique_values.issubset(boolean_values):
# # #         return "boolean"

# # #     # Try integer
# # #     try:
# # #         int_series = pd.to_numeric(series, errors='raise')
# # #         if (int_series % 1 == 0).all():
# # #             int_min = int_series.min()
# # #             int_max = int_series.max()
# # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # #                 return "int"
# # #             else:
# # #                 return "bigint"
# # #     except ValueError:
# # #         pass

# # #     # Try double
# # #     try:
# # #         pd.to_numeric(series, errors='raise', downcast='float')
# # #         return "double"
# # #     except ValueError:
# # #         pass

# # #     # Default to string
# # #     return "string"

# # # def suggest_target_column(df: pd.DataFrame) -> Any:
# # #     """
# # #     Suggests a target column based on numeric data types.
# # #     """
# # #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# # #     return numeric_cols[0] if len(numeric_cols) > 0 else None

# # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # #     """
# # #     Suggests an entity ID column based on uniqueness.
# # #     """
# # #     for col in df.columns:
# # #         if df[col].is_unique:
# # #             return col
# # #     return None

# # # # ===========================
# # # # Unified ChatGPT API
# # # # ===========================
# # # class UnifiedChatGPTAPI(APIView):
# # #     """
# # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # #     Endpoint: /api/chatgpt/
# # #     """
# # #     parser_classes = [MultiPartParser, FormParser, JSONParser]  # Include JSONParser
    
# # #     def post(self, request):
# # #         """
# # #         Handles POST requests for both chat messages and file uploads.
# # #         Differentiates based on the presence of files in the request.
# # #         """
# # #         if "file" in request.FILES:  # If files are present, handle file uploads
# # #             return self.handle_file_upload(request.FILES.getlist("file"))
        
# # #         # Else, handle chat message
# # #         return self.handle_chat(request)
    
# # #     def handle_file_upload(self, files: List[Any]):
# # #         """
# # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # #         After processing, appends schema details to the chat messages.
# # #         """
# # #         if not files:
# # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # #         try:
# # #             uploaded_files_info = []
# # #             s3 = get_s3_client()
# # #             glue = get_glue_client()

# # #             for file in files:
# # #                 # Validate file format
# # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # #                 # Read file into Pandas DataFrame
# # #                 if file.name.lower().endswith('.csv'):
# # #                     df = pd.read_csv(file)
# # #                 else:
# # #                     df = pd.read_excel(file)

# # #                 # Normalize column headers
# # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # #                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # #                 # Infer schema
# # #                 schema = [
# # #                     {
# # #                         "column_name": col,
# # #                         "data_type": infer_column_dtype(df[col])
# # #                     }
# # #                     for col in df.columns
# # #                 ]
# # #                 print(f"Inferred schema: {schema}")  # Debugging statement

# # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # #                 for col in boolean_columns:
# # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # #                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# # #                 # Handle Duplicate Files Dynamically
# # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # #                 if existing_file:
# # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # #                     file.name = new_file_name
# # #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # #                 else:
# # #                     print(f"File name is unique: {file.name}")  # Debugging statement

# # #                 # Save Metadata to Database
# # #                 file.seek(0)  # Reset file pointer before saving
# # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # #                 if file_serializer.is_valid():
# # #                     file_instance = file_serializer.save()

# # #                     # Convert DataFrame to CSV and Upload to S3
# # #                     csv_buffer = BytesIO()
# # #                     df.to_csv(csv_buffer, index=False)
# # #                     csv_buffer.seek(0)
# # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # #                     file_key = f"uploads/{s3_file_name}"

# # #                     # Upload to AWS S3
# # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)

# # #                     # Generate file URL
# # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # #                     file_instance.file_url = file_url
# # #                     file_instance.save()

# # #                     # Save Schema to Database
# # #                     FileSchema.objects.create(file=file_instance, schema=schema)

# # #                     # Trigger AWS Glue Table Update
# # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # #                     # Append file info to response
# # #                     uploaded_files_info.append({
# # #                         'id': file_instance.id,
# # #                         'name': file_instance.name,
# # #                         'file_url': file_instance.file_url,
# # #                         'schema': schema,
# # #                         'suggestions': {  # Add suggestions based on the data
# # #                             'target_column': suggest_target_column(df),
# # #                             'entity_id_column': suggest_entity_id_column(df),
# # #                         }
# # #                     })
# # #                 else:
# # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # #             # After processing all files, update the conversation with schema details
# # #             for uploaded_file in uploaded_files_info:
# # #                 schema_text = self.format_schema_message(uploaded_file)
# # #                 self.append_assistant_message(schema_text)

# # #             return Response({
# # #                 "message": "Files uploaded and processed successfully.",
# # #                 "uploaded_files": uploaded_files_info
# # #             }, status=status.HTTP_201_CREATED)

# # #         except pd.errors.EmptyDataError:
# # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # #         except NoCredentialsError:
# # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # #         except ClientError as e:
# # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # #         except Exception as e:
# # #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # #     def handle_chat(self, request):
# # #         """
# # #         Handles user chat messages using ChatGPT.
# # #         """
# # #         user_input = request.data.get("message", "").strip()
# # #         user_id = request.data.get("user_id", "default_user")  # Optional: Track user sessions

# # #         if not user_input:
# # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # #         # Initialize session if new user
# # #         if chatgpt_state_machine.state == "INITIAL" and not chatgpt_state_machine.answers:
# # #             # Already initialized in ChatGPTStateMachine constructor
# # #             pass

# # #         # Include schema in the conversation if available
# # #         if chatgpt_state_machine.dataset_schema:
# # #             schema_info = f"The dataset has the following columns: {', '.join(chatgpt_state_machine.dataset_schema['columns'])}."
# # #             conversation_chain_chatgpt.memory.save_context(
# # #                 {"user_input": user_input},
# # #                 {"output": f"Schema info: {schema_info}"}
# # #             )

# # #         # Process user input through the conversation chain
# # #         assistant_response = conversation_chain_chatgpt.run(user_input=user_input)

# # #         # Determine next state and generate structured responses
# # #         current_state = chatgpt_state_machine.state
# # #         response = ""

# # #         if current_state == "INITIAL":
# # #             response = (
# # #                 "Hi! 👋 I'm your AI assistant.\n"
# # #                 "I'll help you formulate a predictive question. Let's start with the Subject. What do you want to predict?"
# # #             )
# # #         elif current_state == "DEFINE_SUBJECT":
# # #             chatgpt_state_machine.answers["subject"] = user_input
# # #             response = (
# # #                 f"Got it! The Subject of your prediction is '{user_input}'.\n"
# # #                 "What is the Target Value you'd like to predict for this Subject? For example, price or churn probability."
# # #             )
# # #         elif current_state == "DEFINE_TARGET":
# # #             chatgpt_state_machine.answers["target"] = user_input
# # #             response = (
# # #                 f"Understood. You want to predict '{user_input}' for '{chatgpt_state_machine.answers['subject']}'.\n"
# # #                 "Does your prediction involve a specific time frame or conditions?"
# # #             )
# # #         elif current_state == "ASK_TIME_ELEMENT":
# # #             chatgpt_state_machine.answers["time_element"] = user_input
# # #             response = (
# # #                 f"Got it! The time frame is '{user_input}' if provided, or none.\n"
# # #                 "Here's a summary of what we have so far:\n"
# # #                 f"- Subject: {chatgpt_state_machine.answers.get('subject')}\n"
# # #                 f"- Target: {chatgpt_state_machine.answers.get('target')}\n"
# # #                 f"- Time Frame: {chatgpt_state_machine.answers.get('time_element', 'None')}\n\n"
# # #                 "Does that look correct?"
# # #             )
# # #         elif current_state == "CONFIRMATION":
# # #             if user_input.lower() in ["yes", "correct"]:
# # #                 response = "Great! We're ready to proceed with building your predictive model."
# # #             else:
# # #                 chatgpt_state_machine.reset()
# # #                 response = "Let's start over. What is the Subject of your prediction?"
# # #         elif current_state == "COMPLETE":
# # #             response = "Thank you for using the AI assistant!"

# # #         # Transition to the next state
# # #         chatgpt_state_machine.next_state(user_input)

# # #         # Combine assistant responses
# # #         final_response = f"{assistant_response}\n\n{response}" if response else assistant_response

# # #         # Append the assistant response to the conversation
# # #         self.append_assistant_message(final_response)

# # #         return Response({
# # #             # "response": final_response
# # #             "response": assistant_response
# # #         })

# # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # #         """
# # #         Formats the schema information to be appended as an assistant message in the chat.
# # #         """
# # #         schema = uploaded_file['schema']
# # #         schema_text = (
# # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # #             f"Target Column Suggestion: {uploaded_file['suggestions']['target_column']}\n"
# # #             f"Entity ID Column Suggestion: {uploaded_file['suggestions']['entity_id_column']}\n\n"
# # #             "Please confirm:\n\n"
# # #             "- Is the Target Column correct?\n"
# # #             "- Is the Entity ID Column correct?\n"
# # #             '(Reply "yes" or provide the correct column names.)'
# # #         )
# # #         return schema_text

# # #     def append_assistant_message(self, message_text: str):
# # #         """
# # #         Appends an assistant message to the conversation history.
# # #         Note: In this implementation, conversation history is managed via LangChain's memory.
# # #         If you have a separate mechanism for storing messages, implement it accordingly.
# # #         """
# # #         # Since LangChain manages the conversation history internally, we don't need to manually append messages.
# # #         # However, if you have a frontend that maintains its own chat history, ensure it displays the assistant messages accordingly.
# # #         pass  # Implement based on your frontend integration

# # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # #         """
# # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # #         """
# # #         glue = get_glue_client()
# # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # #         storage_descriptor = {
# # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # #             'Location': s3_location,
# # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # #             'SerdeInfo': {
# # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # #                 'Parameters': {
# # #                     'field.delim': ',',
# # #                     'skip.header.line.count': '1'
# # #                 }
# # #             }
# # #         }
# # #         try:
# # #             glue.update_table(
# # #                 DatabaseName='pa_user_datafiles_db',
# # #                 TableInput={
# # #                     'Name': table_name,
# # #                     'StorageDescriptor': storage_descriptor,
# # #                     'TableType': 'EXTERNAL_TABLE'
# # #                 }
# # #             )
# # #             print(f"Glue table '{table_name}' updated successfully.")
# # #         except glue.exceptions.EntityNotFoundException:
# # #             print(f"Table '{table_name}' not found. Creating a new table...")
# # #             glue.create_table(
# # #                 DatabaseName='pa_user_datafiles_db',
# # #                 TableInput={
# # #                     'Name': table_name,
# # #                     'StorageDescriptor': storage_descriptor,
# # #                     'TableType': 'EXTERNAL_TABLE'
# # #                 }
# # #             )
# # #             print(f"Glue table '{table_name}' created successfully.")
# # #         except Exception as e:
# # #             print(f"Glue operation failed: {str(e)}")




# # # # # # Chat/views.py

# # # # # import os
# # # # # import datetime
# # # # # from io import BytesIO
# # # # # from typing import Any, Dict, List

# # # # # import boto3
# # # # # import pandas as pd
# # # # # import openai
# # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # from django.conf import settings
# # # # # from rest_framework import status
# # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # from rest_framework.response import Response
# # # # # from rest_framework.views import APIView
# # # # # from langchain.chains import ConversationChain
# # # # # from langchain.chat_models import ChatOpenAI
# # # # # from langchain.memory import ConversationBufferMemory
# # # # # from langchain.prompts import PromptTemplate

# # # # # from .models import FileSchema, UploadedFile
# # # # # from .serializers import UploadedFileSerializer

# # # # # # ===========================
# # # # # # AWS Configuration
# # # # # # ===========================
# # # # # AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# # # # # AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# # # # # AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")
# # # # # AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")

# # # # # # ===========================
# # # # # # OpenAI Configuration
# # # # # # ===========================
# # # # # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment

# # # # # # ===========================
# # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # ===========================
# # # # # openai.api_key = OPENAI_API_KEY

# # # # # # Define available models
# # # # # available_models = [
# # # # #     "gpt-4",
# # # # #     "gpt-4o",
# # # # #     "gpt-4-turbo",
# # # # #     "gpt-3.5-turbo",
# # # # #     "gpt-3.5-turbo-16k",
# # # # # ]

# # # # # llm_chatgpt = ChatOpenAI(
# # # # #     # model="gpt-4",
# # # # #     model="gpt-3.5-turbo-16k",
# # # # #     temperature=0.7,
# # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # )

# # # # # prompt_chatgpt = PromptTemplate(
# # # # #     input_variables=["history", "user_input"],
# # # # #     template=(
# # # # #         "You are a helpful AI assistant. You guide users through defining predictive questions and refining goals.\n"
# # # # #         "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
# # # # #         "Steps:\n"
# # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # #         "4. Reference the dataset schema if available.\n"
# # # # #         "5. Summarize inputs before proceeding to model creation.\n\n"
# # # # #         "Conversation history: {history}\n"
# # # # #         "User input: {user_input}\n"
# # # # #         "Assistant:"
# # # # #     ),
# # # # # )

# # # # # memory_chatgpt = ConversationBufferMemory(
# # # # #     memory_key="history",
# # # # #     input_key="user_input",
# # # # # )

# # # # # conversation_chain_chatgpt = ConversationChain(
# # # # #     llm=llm_chatgpt,
# # # # #     prompt=prompt_chatgpt,
# # # # #     memory=memory_chatgpt,
# # # # #     input_key="user_input",
# # # # # )

# # # # # # ===========================
# # # # # # ChatGPT State Machine
# # # # # # ===========================
# # # # # class ChatGPTStateMachine:
# # # # #     """
# # # # #     Manages the conversation state for ChatGPT-based interactions.
# # # # #     """
# # # # #     def __init__(self):
# # # # #         self.state = "INITIAL"
# # # # #         self.answers = {}
# # # # #         self.dataset_schema = None  # Store the dataset schema

# # # # #     def update_dataset_schema(self, schema: List[Dict[str, Any]]):
# # # # #         """
# # # # #         Update the dataset schema in the state machine.
# # # # #         """
# # # # #         self.dataset_schema = {
# # # # #             "columns": [col["column_name"] for col in schema],
# # # # #             "data_types": {col["column_name"]: col["data_type"] for col in schema},
# # # # #         }

# # # # #     def reset(self):
# # # # #         """
# # # # #         Reset the conversation state to initial.
# # # # #         """
# # # # #         self.state = "INITIAL"
# # # # #         self.answers = {}
# # # # #         self.dataset_schema = None


# # # # # # Initialize ChatGPT state machine
# # # # # chatgpt_state_machine = ChatGPTStateMachine()

# # # # # # ===========================
# # # # # # Utility Functions
# # # # # # ===========================
# # # # # def get_s3_client():
# # # # #     return boto3.client(
# # # # #         "s3",
# # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # #         region_name=AWS_S3_REGION_NAME,
# # # # #     )


# # # # # def get_glue_client():
# # # # #     return boto3.client(
# # # # #         "glue",
# # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # #         region_name=AWS_S3_REGION_NAME,
# # # # #     )


# # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # #     series = series.dropna().astype(str).str.strip()
# # # # #     boolean_values = {"true", "false", "1", "0", "yes", "no"}
# # # # #     unique_values = set(series.str.lower().unique())
# # # # #     if unique_values.issubset(boolean_values):
# # # # #         return "boolean"
# # # # #     try:
# # # # #         int_series = pd.to_numeric(series, errors="raise")
# # # # #         if (int_series % 1 == 0).all():
# # # # #             return "int"
# # # # #     except ValueError:
# # # # #         pass
# # # # #     try:
# # # # #         pd.to_numeric(series, errors="raise", downcast="float")
# # # # #         return "double"
# # # # #     except ValueError:
# # # # #         pass
# # # # #     return "string"


# # # # # def suggest_target_column(df: pd.DataFrame) -> Any:
# # # # #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# # # # #     return numeric_cols[0] if len(numeric_cols) > 0 else None


# # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # #     for col in df.columns:
# # # # #         if df[col].is_unique:
# # # # #             return col
# # # # #     return None

# # # # # # ===========================
# # # # # # Unified ChatGPT API
# # # # # # ===========================
# # # # # class UnifiedChatGPTAPI(APIView):
# # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # #     def post(self, request):
# # # # #         if "file" in request.FILES:
# # # # #             return self.handle_file_upload(request.FILES.getlist("file"))
# # # # #         return self.handle_chat(request)

# # # # #     def handle_file_upload(self, files: List[Any]):
# # # # #         if not files:
# # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # #         try:
# # # # #             uploaded_files_info = []
# # # # #             s3 = get_s3_client()

# # # # #             for file in files:
# # # # #                 if file.name.lower().endswith(".csv"):
# # # # #                     df = pd.read_csv(file)
# # # # #                 else:
# # # # #                     df = pd.read_excel(file)
# # # # #                 df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# # # # #                 schema = [
# # # # #                     {"column_name": col, "data_type": infer_column_dtype(df[col])}
# # # # #                     for col in df.columns
# # # # #                 ]

# # # # #                 chatgpt_state_machine.update_dataset_schema(schema)

# # # # #                 file_name = file.name
# # # # #                 s3_key = f"uploads/{file_name}"
# # # # #                 s3.upload_fileobj(file, AWS_STORAGE_BUCKET_NAME, s3_key)

# # # # #                 file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"
# # # # #                 uploaded_files_info.append(
# # # # #                     {
# # # # #                         "name": file_name,
# # # # #                         "file_url": file_url,
# # # # #                         "schema": schema,
# # # # #                         "suggestions": {
# # # # #                             "target_column": suggest_target_column(df),
# # # # #                             "entity_id_column": suggest_entity_id_column(df),
# # # # #                         },
# # # # #                     }
# # # # #                 )

# # # # #             for uploaded_file in uploaded_files_info:
# # # # #                 schema_text = self.format_schema_message(uploaded_file)
# # # # #                 self.append_assistant_message(schema_text)

# # # # #             return Response(
# # # # #                 {
# # # # #                     "message": "Files uploaded and processed successfully.",
# # # # #                     "uploaded_files": uploaded_files_info,
# # # # #                 },
# # # # #                 status=status.HTTP_201_CREATED,
# # # # #             )

# # # # #         except Exception as e:
# # # # #             return Response({"error": f"File processing failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

# # # # #     def handle_chat(self, request):
# # # # #         user_input = request.data.get("message", "").strip()
# # # # #         if not user_input:
# # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # #         if chatgpt_state_machine.dataset_schema:
# # # # #             schema_context = f"The dataset has the following columns: {', '.join(chatgpt_state_machine.dataset_schema['columns'])}. " \
# # # # #                              f"Data types: {', '.join([f'{col}: {dtype}' for col, dtype in chatgpt_state_machine.dataset_schema['data_types'].items()])}."
# # # # #             conversation_chain_chatgpt.memory.save_context(
# # # # #                 {"user_input": user_input},
# # # # #                 {"output": schema_context},
# # # # #             )

# # # # #         assistant_response = conversation_chain_chatgpt.run(user_input=user_input)
# # # # #         return Response({"response": assistant_response})

# # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # #         schema = uploaded_file["schema"]
# # # # #         schema_text = (
# # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # #             "Columns:\n" + ", ".join([col["column_name"] for col in schema]) + "\n\n"
# # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # #             f"Target Column Suggestion: {uploaded_file['suggestions']['target_column']}\n"
# # # # #             f"Entity ID Column Suggestion: {uploaded_file['suggestions']['entity_id_column']}\n\n"
# # # # #             "Please confirm:\n\n"
# # # # #             "- Is the Target Column correct?\n"
# # # # #             "- Is the Entity ID Column correct?\n"
# # # # #             '(Reply "yes" or provide the correct column names.)'
# # # # #         )
# # # # #         return schema_text

# # # # #     def append_assistant_message(self, message_text: str):
# # # # #         """
# # # # #         Appends an assistant message to the conversation history.
# # # # #         This implementation uses LangChain's memory to manage the conversation.
# # # # #         """
# # # # #         # Retrieve the last user input from memory (if needed for context)
# # # # #         history = memory_chatgpt.load_memory_variables({})["history"]

# # # # #         # Append the message to LangChain's memory
# # # # #         memory_chatgpt.save_context(
# # # # #             {"user_input": history},
# # # # #             {"output": message_text},
# # # # #         )

# # # # #         # Optionally, print for debugging purposes
# # # # #         print(f"Assistant message appended: {message_text}")


# # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # #         glue = get_glue_client()
# # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # #         storage_descriptor = {
# # # # #             "Columns": [{"Name": col["column_name"], "Type": col["data_type"]} for col in schema],
# # # # #             "Location": s3_location,
# # # # #             "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
# # # # #             "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
# # # # #             "SerdeInfo": {
# # # # #                 "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
# # # # #                 "Parameters": {"field.delim": ",", "skip.header.line.count": "1"},
# # # # #             },
# # # # #         }
# # # # #         try:
# # # # #             glue.update_table(
# # # # #                 DatabaseName="pa_user_datafiles_db",
# # # # #                 TableInput={"Name": table_name, "StorageDescriptor": storage_descriptor, "TableType": "EXTERNAL_TABLE"},
# # # # #             )
# # # # #         except glue.exceptions.EntityNotFoundException:
# # # # #             glue.create_table(
# # # # #                 DatabaseName="pa_user_datafiles_db",
# # # # #                 TableInput={"Name": table_name, "StorageDescriptor": storage_descriptor, "TableType": "EXTERNAL_TABLE"},
# # # # #             )



# # # # import os
# # # # import pandas as pd
# # # # from io import BytesIO
# # # # import boto3
# # # # from botocore.exceptions import ClientError
# # # # from rest_framework.views import APIView
# # # # from rest_framework.response import Response
# # # # from rest_framework import status
# # # # from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

# # # # # LangChain Imports
# # # # from langchain.chains import ConversationChain
# # # # from langchain.chat_models import ChatOpenAI
# # # # from langchain.memory import ConversationBufferMemory
# # # # from langchain.prompts import PromptTemplate

# # # # # ============================
# # # # # AWS Configuration
# # # # # ============================
# # # # AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# # # # AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# # # # AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")
# # # # AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
# # # # DATABASE_NAME = "pa_user_datafiles_db"  # Glue Database name

# # # # # ============================
# # # # # OpenAI Configuration
# # # # # ============================
# # # # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # # # # ============================
# # # # # LangChain Setup for Chat
# # # # # ============================
# # # # # Initializes the chat model with a specific prompt and memory configuration
# # # # llm_chatgpt = ChatOpenAI(
# # # #     model="gpt-3.5-turbo-16k",
# # # #     temperature=0.7,
# # # #     openai_api_key=OPENAI_API_KEY,
# # # # )

# # # # # Chat prompt template guiding assistant behavior
# # # # prompt_chatgpt = PromptTemplate(
# # # #     input_variables=["history", "user_input"],
# # # #     template=(
# # # #         "You are a helpful AI assistant. Guide users through defining predictive questions and refining goals.\n"
# # # #         "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
# # # #         "Steps:\n"
# # # #         "1. Discuss the Subject they want to predict.\n"
# # # #         "2. Confirm the Target Value they want to predict.\n"
# # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # #         "4. Reference the dataset schema if available.\n"
# # # #         "5. Summarize inputs before proceeding to model creation.\n\n"
# # # #         "Conversation history: {history}\n"
# # # #         "User input: {user_input}\n"
# # # #         "Assistant:"
# # # #     ),
# # # # )

# # # # # Stores conversation memory for contextual replies
# # # # memory_chatgpt = ConversationBufferMemory(
# # # #     memory_key="history",
# # # #     input_key="user_input",
# # # # )

# # # # # Conversation chain linking LLM with the prompt and memory
# # # # conversation_chain_chatgpt = ConversationChain(
# # # #     llm=llm_chatgpt,
# # # #     prompt=prompt_chatgpt,
# # # #     memory=memory_chatgpt,
# # # #     input_key="user_input",
# # # # )

# # # # # ============================
# # # # # Utility Functions for AWS
# # # # # ============================
# # # # def get_s3_client():
# # # #     """
# # # #     Returns an S3 client for AWS operations.
# # # #     """
# # # #     return boto3.client(
# # # #         "s3",
# # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #         region_name=AWS_S3_REGION_NAME,
# # # #     )


# # # # def get_glue_client():
# # # #     """
# # # #     Returns a Glue client for AWS operations.
# # # #     """
# # # #     return boto3.client(
# # # #         "glue",
# # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #         region_name=AWS_S3_REGION_NAME,
# # # #     )


# # # # def infer_column_dtype(series):
# # # #     """
# # # #     Infers the data type of a column for Glue/Athena.

# # # #     - Boolean: Columns with values like "true/false", "yes/no", "1/0".
# # # #     - Integer: Numeric values with no decimal places.
# # # #     - Double: Numeric values with decimals.
# # # #     - String: Fallback type for all others.
# # # #     """
# # # #     series = series.dropna().astype(str).str.strip()
# # # #     boolean_values = {"true", "false", "1", "0", "yes", "no"}
# # # #     unique_values = set(series.str.lower().unique())

# # # #     if unique_values.issubset(boolean_values):
# # # #         return "boolean"

# # # #     try:
# # # #         int_series = pd.to_numeric(series, errors="raise")
# # # #         if (int_series % 1 == 0).all():
# # # #             return "int"
# # # #     except ValueError:
# # # #         pass

# # # #     try:
# # # #         pd.to_numeric(series, errors="raise", downcast="float")
# # # #         return "double"
# # # #     except ValueError:
# # # #         pass

# # # #     return "string"


# # # # def suggest_target_column(df):
# # # #     """
# # # #     Suggests a target column based on numeric data.
# # # #     Returns the first numeric column found or None.
# # # #     """
# # # #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# # # #     return numeric_cols[0] if len(numeric_cols) > 0 else None


# # # # def suggest_entity_id_column(df):
# # # #     """
# # # #     Suggests an entity ID column based on column uniqueness.
# # # #     Returns the first column with unique values or None.
# # # #     """
# # # #     for col in df.columns:
# # # #         if df[col].is_unique:
# # # #             return col
# # # #     return None


# # # # def create_glue_table(glue, table_name, schema, s3_key):
# # # #     """
# # # #     Creates or updates a Glue table dynamically for the uploaded dataset.
# # # #     """
# # # #     s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/{s3_key}"
# # # #     columns = [{"Name": col["column_name"], "Type": col["data_type"]} for col in schema]

# # # #     try:
# # # #         print(f"Creating Glue table for {table_name}...")
# # # #         glue.create_table(
# # # #             DatabaseName=DATABASE_NAME,
# # # #             TableInput={
# # # #                 "Name": table_name,
# # # #                 "StorageDescriptor": {
# # # #                     "Columns": columns,
# # # #                     "Location": s3_location,
# # # #                     "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
# # # #                     "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
# # # #                     "SerdeInfo": {
# # # #                         "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
# # # #                         "Parameters": {"field.delim": ",", "skip.header.line.count": "1"},
# # # #                     },
# # # #                 },
# # # #                 "TableType": "EXTERNAL_TABLE",
# # # #             },
# # # #         )
# # # #         print(f"Glue table {table_name} created successfully.")
# # # #     except glue.exceptions.AlreadyExistsException:
# # # #         print(f"Glue table {table_name} already exists. Updating table schema.")
# # # #         glue.update_table(
# # # #             DatabaseName=DATABASE_NAME,
# # # #             TableInput={
# # # #                 "Name": table_name,
# # # #                 "StorageDescriptor": {
# # # #                     "Columns": columns,
# # # #                     "Location": s3_location,
# # # #                 },
# # # #             },
# # # #         )
# # # #         print(f"Glue table {table_name} updated successfully.")

# # # # # ============================
# # # # # Unified ChatGPT and File Upload API
# # # # # ============================
# # # # class UnifiedChatGPTAPI(APIView):
# # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # #     def post(self, request):
# # # #         """
# # # #         Unified endpoint to handle both file upload and chat interactions.
# # # #         """
# # # #         if "file" in request.FILES:
# # # #             return self.handle_file_upload(request.FILES.getlist("file"))
# # # #         return self.handle_chat(request)

# # # #     def handle_file_upload(self, files):
# # # #         """
# # # #         Handles file upload:
# # # #         1. Normalizes columns and converts XLSX to CSV if needed.
# # # #         2. Infers schema, uploads file to S3, and registers Glue table.
# # # #         3. Passes schema details to frontend and integrates into chat.
# # # #         """
# # # #         s3 = get_s3_client()
# # # #         glue = get_glue_client()
# # # #         uploaded_files_info = []

# # # #         for file in files:
# # # #             print(f"Processing file: {file.name}")
# # # #             if not file.size:
# # # #                 print("File is empty. Skipping upload.")
# # # #                 return Response({"error": f"File '{file.name}' is empty and was not uploaded."}, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Read and normalize file
# # # #             if file.name.endswith(".csv"):
# # # #                 df = pd.read_csv(file)
# # # #             elif file.name.endswith((".xls", ".xlsx")):
# # # #                 df = pd.read_excel(file)  # Convert XLSX to CSV
# # # #             else:
# # # #                 print(f"Unsupported file format: {file.name}")
# # # #                 return Response({"error": "Unsupported file type."}, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Normalize columns for Athena compatibility
# # # #             df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# # # #             # Validate if the dataset contains data
# # # #             if df.empty:
# # # #                 print(f"The dataset in {file.name} is empty.")
# # # #                 return Response({"error": f"Dataset '{file.name}' is empty."}, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Save DataFrame to buffer for S3 upload
# # # #             buffer = BytesIO()
# # # #             df.to_csv(buffer, index=False)
# # # #             buffer.seek(0)

# # # #             # Upload file to S3
# # # #             file_name = file.name.split(".")[0].lower() + ".csv"
# # # #             s3_key = f"uploads/{file_name}"
# # # #             s3.upload_fileobj(buffer, AWS_STORAGE_BUCKET_NAME, s3_key)
# # # #             print(f"File uploaded to S3: {s3_key}")

# # # #             # Verify file in S3 (Optional Debugging)
# # # #             print(f"Verifying file in S3: {s3_key}")
# # # #             s3_response = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=s3_key)
# # # #             if s3_response['ContentLength'] == 0:
# # # #                 print(f"File {s3_key} is empty in S3!")
# # # #                 return Response({"error": f"File '{file.name}' uploaded to S3 but is empty."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # #             # Infer schema
# # # #             schema = [
# # # #                 {"column_name": col, "data_type": infer_column_dtype(df[col])}
# # # #                 for col in df.columns
# # # #             ]
# # # #             target_column = suggest_target_column(df)
# # # #             entity_id_column = suggest_entity_id_column(df)

# # # #             # Create Glue table
# # # #             table_name = file_name.replace(".", "_").replace("-", "_")
# # # #             create_glue_table(glue, table_name, schema, s3_key)

# # # #             # Append schema details to chat
# # # #             schema_text = self.format_schema_message(file_name, schema, target_column, entity_id_column)
# # # #             self.append_assistant_message(schema_text)

# # # #             # Append to response
# # # #             uploaded_files_info.append(
# # # #                 {
# # # #                     "file_name": file_name,
# # # #                     "s3_key": s3_key,
# # # #                     "table_name": table_name,
# # # #                     "schema": schema,
# # # #                     "suggestions": {
# # # #                         "target_column": target_column,
# # # #                         "entity_id_column": entity_id_column,
# # # #                     },
# # # #                 }
# # # #             )

# # # #         return Response(
# # # #             {
# # # #                 "message": "Files uploaded successfully.",
# # # #                 "files": uploaded_files_info,
# # # #                 "chat_message": schema_text,  # Include schema message for frontend
# # # #             },
# # # #             status=status.HTTP_201_CREATED,
# # # #         )

# # # #     def handle_chat(self, request):
# # # #         """
# # # #         Handles chat input by interacting with LangChain's conversation model.
# # # #         """
# # # #         user_input = request.data.get("message", "").strip()
# # # #         if not user_input:
# # # #             print("No user input received.")
# # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # #         print(f"User input: {user_input}")
# # # #         assistant_response = conversation_chain_chatgpt.run(user_input=user_input)
# # # #         print(f"Assistant response: {assistant_response}")
# # # #         return Response({"response": assistant_response})

# # # #     def format_schema_message(self, file_name, schema, target_column, entity_id_column):
# # # #         """
# # # #         Formats schema details for integration into the assistant conversation.
# # # #         """
# # # #         schema_text = (
# # # #             f"Dataset '{file_name}' uploaded successfully!\n\n"
# # # #             "Columns:\n" + ", ".join([col["column_name"] for col in schema]) + "\n\n"
# # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # #             f"Target Column Suggestion: {target_column}\n"
# # # #             f"Entity ID Column Suggestion: {entity_id_column}\n\n"
# # # #             "Please confirm if this schema matches your expectations."
# # # #         )
# # # #         return schema_text

# # # #     def append_assistant_message(self, message_text):
# # # #         """
# # # #         Appends assistant messages to conversation memory.
# # # #         """
# # # #         print(f"Appending assistant message to conversation: {message_text}")
# # # #         memory_chatgpt.save_context(
# # # #             {"user_input": memory_chatgpt.load_memory_variables({})["history"]},
# # # #             {"output": message_text},
# # # #         )




# # # # lean




# # # # # backend_code.py

# # # # import os
# # # # import pandas as pd
# # # # from io import BytesIO
# # # # import boto3
# # # # from botocore.exceptions import ClientError
# # # # from rest_framework.views import APIView
# # # # from rest_framework.response import Response
# # # # from rest_framework import status
# # # # from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

# # # # # LangChain Imports
# # # # from langchain.chains import ConversationChain
# # # # from langchain.chat_models import ChatOpenAI
# # # # from langchain.memory import ConversationBufferMemory
# # # # from langchain.prompts import PromptTemplate

# # # # # ============================
# # # # # AWS Configuration
# # # # # ============================
# # # # AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# # # # AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# # # # AWS_S3_REGION_NAME = os.getenv("AWS_S3_REGION_NAME")
# # # # AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
# # # # DATABASE_NAME = "pa_user_datafiles_db"  # Glue Database name

# # # # # ============================
# # # # # OpenAI Configuration
# # # # # ============================
# # # # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # # # # ============================
# # # # # LangChain Setup for Chat
# # # # # ============================
# # # # # Initializes the chat model with a specific prompt and memory configuration
# # # # llm_chatgpt = ChatOpenAI(
# # # #     model="gpt-3.5-turbo-16k",
# # # #     temperature=0.7,
# # # #     openai_api_key=OPENAI_API_KEY,
# # # # )

# # # # # Chat prompt template guiding assistant behavior
# # # # prompt_chatgpt = PromptTemplate(
# # # #     input_variables=["history", "user_input"],
# # # #     template=(
# # # #         "You are a helpful AI assistant. Guide users through defining predictive questions and refining goals.\n"
# # # #         "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
# # # #         "Steps:\n"
# # # #         "1. Discuss the Subject they want to predict.\n"
# # # #         "2. Confirm the Target Value they want to predict.\n"
# # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # #         "4. Reference the dataset schema if available.\n"
# # # #         "5. Summarize inputs before proceeding to model creation.\n\n"
# # # #         "Conversation history: {history}\n"
# # # #         "User input: {user_input}\n"
# # # #         "Assistant:"
# # # #     ),
# # # # )

# # # # # Stores conversation memory for contextual replies
# # # # memory_chatgpt = ConversationBufferMemory(
# # # #     memory_key="history",
# # # #     input_key="user_input",
# # # # )

# # # # # Conversation chain linking LLM with the prompt and memory
# # # # conversation_chain_chatgpt = ConversationChain(
# # # #     llm=llm_chatgpt,
# # # #     prompt=prompt_chatgpt,
# # # #     memory=memory_chatgpt,
# # # #     input_key="user_input",
# # # # )

# # # # # ============================
# # # # # Utility Functions for AWS
# # # # # ============================
# # # # def get_s3_client():
# # # #     """
# # # #     Returns an S3 client for AWS operations.
# # # #     """
# # # #     return boto3.client(
# # # #         "s3",
# # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #         region_name=AWS_S3_REGION_NAME,
# # # #     )


# # # # def get_glue_client():
# # # #     """
# # # #     Returns a Glue client for AWS operations.
# # # #     """
# # # #     return boto3.client(
# # # #         "glue",
# # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #         region_name=AWS_S3_REGION_NAME,
# # # #     )


# # # # def infer_column_dtype(series):
# # # #     """
# # # #     Infers the data type of a column for Glue/Athena.

# # # #     - Boolean: Columns with values like "true/false", "yes/no", "1/0".
# # # #     - Integer: Numeric values with no decimal places.
# # # #     - Double: Numeric values with decimals.
# # # #     - String: Fallback type for all others.
# # # #     """
# # # #     series = series.dropna().astype(str).str.strip()
# # # #     boolean_values = {"true", "false", "1", "0", "yes", "no"}
# # # #     unique_values = set(series.str.lower().unique())

# # # #     if unique_values.issubset(boolean_values):
# # # #         return "boolean"

# # # #     try:
# # # #         int_series = pd.to_numeric(series, errors="raise")
# # # #         if (int_series % 1 == 0).all():
# # # #             return "int"
# # # #     except ValueError:
# # # #         pass

# # # #     try:
# # # #         pd.to_numeric(series, errors="raise", downcast="float")
# # # #         return "double"
# # # #     except ValueError:
# # # #         pass

# # # #     return "string"


# # # # def suggest_target_column(df):
# # # #     """
# # # #     Suggests a target column based on numeric data.
# # # #     Returns the last numeric column found or None.
# # # #     """
# # # #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# # # #     return numeric_cols[-1] if len(numeric_cols) > 0 else None


# # # # def suggest_entity_id_column(df):
# # # #     """
# # # #     Suggests an entity ID column based on column uniqueness.
# # # #     Returns the first column with unique values or None.
# # # #     """
# # # #     for col in df.columns:
# # # #         if df[col].is_unique:
# # # #             return col
# # # #     return None


# # # # def create_glue_table(glue, table_name, schema, s3_key):
# # # #     """
# # # #     Creates or updates a Glue table dynamically for the uploaded dataset.
# # # #     """
# # # #     s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/{s3_key}"
# # # #     columns = [{"Name": col["column_name"], "Type": col["data_type"]} for col in schema]

# # # #     table_input = {
# # # #         "Name": table_name,
# # # #         "StorageDescriptor": {
# # # #             "Columns": columns,
# # # #             "Location": s3_location,
# # # #             "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
# # # #             "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
# # # #             "SerdeInfo": {
# # # #                 "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
# # # #                 "Parameters": {
# # # #                     "field.delim": ",",
# # # #                     "skip.header.line.count": "1",
# # # #                 },
# # # #             },
# # # #         },
# # # #         "TableType": "EXTERNAL_TABLE",
# # # #         "Parameters": {
# # # #             "skip.header.line.count": "1",
# # # #         },
# # # #     }

# # # #     try:
# # # #         print(f"Creating Glue table for {table_name}...")
# # # #         glue.create_table(
# # # #             DatabaseName=DATABASE_NAME,
# # # #             TableInput=table_input,
# # # #         )
# # # #         print(f"Glue table {table_name} created successfully.")
# # # #     except glue.exceptions.AlreadyExistsException:
# # # #         print(f"Glue table {table_name} already exists. Updating table schema.")
# # # #         glue.update_table(
# # # #             DatabaseName=DATABASE_NAME,
# # # #             TableInput=table_input,
# # # #         )
# # # #         print(f"Glue table {table_name} updated successfully.")


# # # # # ============================
# # # # # Unified ChatGPT and File Upload API
# # # # # ============================
# # # # class UnifiedChatGPTAPI(APIView):
# # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # #     def post(self, request):
# # # #         """
# # # #         Unified endpoint to handle both file upload and chat interactions.
# # # #         """
# # # #         if "file" in request.FILES:
# # # #             return self.handle_file_upload(request.FILES.getlist("file"))
# # # #         return self.handle_chat(request)

# # # #     def handle_file_upload(self, files):
# # # #         """
# # # #         Handles file upload:
# # # #         1. Normalizes columns and converts XLSX to CSV if needed.
# # # #         2. Infers schema, uploads file to S3, and registers Glue table.
# # # #         3. Passes schema details to frontend and integrates into chat.
# # # #         """
# # # #         s3 = get_s3_client()
# # # #         glue = get_glue_client()
# # # #         uploaded_files_info = []
# # # #         schema_texts = []

# # # #         for file in files:
# # # #             print(f"Processing file: {file.name}")
# # # #             if not file.size:
# # # #                 print("File is empty. Skipping upload.")
# # # #                 return Response({"error": f"File '{file.name}' is empty and was not uploaded."}, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Read and normalize file
# # # #             if file.name.endswith(".csv"):
# # # #                 df = pd.read_csv(file)
# # # #             elif file.name.endswith((".xls", ".xlsx")):
# # # #                 df = pd.read_excel(file)  # Convert XLSX to CSV
# # # #             else:
# # # #                 print(f"Unsupported file format: {file.name}")
# # # #                 return Response({"error": "Unsupported file type."}, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Normalize columns for Athena compatibility
# # # #             df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# # # #             # Validate if the dataset contains data
# # # #             if df.empty:
# # # #                 print(f"The dataset in {file.name} is empty.")
# # # #                 return Response({"error": f"Dataset '{file.name}' is empty."}, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Save DataFrame to buffer for S3 upload
# # # #             buffer = BytesIO()
# # # #             df.to_csv(buffer, index=False)
# # # #             buffer.seek(0)

# # # #             # Upload file to S3
# # # #             file_name = file.name.split(".")[0].lower() + ".csv"
# # # #             s3_key = f"uploads/{file_name}"
# # # #             s3.upload_fileobj(buffer, AWS_STORAGE_BUCKET_NAME, s3_key)
# # # #             # print(f"File uploaded to S3: {s3_key}")

# # # #             # Verify file in S3 (Optional Debugging)
# # # #             print(f"Verifying file in S3: {s3_key}")
# # # #             s3_response = s3.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=s3_key)
# # # #             if s3_response['ContentLength'] == 0:
# # # #                 print(f"File {s3_key} is empty in S3!")
# # # #                 return Response({"error": f"File '{file.name}' uploaded to S3 but is empty."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # #             # Infer schema
# # # #             schema = [
# # # #                 {"column_name": col, "data_type": infer_column_dtype(df[col])}
# # # #                 for col in df.columns
# # # #             ]
# # # #             target_column = suggest_target_column(df)
# # # #             entity_id_column = suggest_entity_id_column(df)

# # # #             # Create Glue table
# # # #             table_name = file_name.replace(".", "_").replace("-", "_")
# # # #             create_glue_table(glue, table_name, schema, s3_key)

# # # #             # Append schema details to chat
# # # #             schema_text = self.format_schema_message(file_name, schema, target_column, entity_id_column)
# # # #             schema_texts.append(schema_text)
# # # #             self.append_assistant_message(schema_text)

# # # #             # Append to response
# # # #             uploaded_files_info.append(
# # # #                 {
# # # #                     "file_name": file_name,  # Keeping 'file_name' to maintain chat services
# # # #                     "s3_key": s3_key,
# # # #                     "table_name": table_name,
# # # #                     "schema": schema,
# # # #                     "suggestions": {
# # # #                         "target_column": target_column,
# # # #                         "entity_id_column": entity_id_column,
# # # #                     },
# # # #                 }
# # # #             )

# # # #         # Combine all schema_texts into one if multiple files are uploaded
# # # #         combined_schema_text = "\n\n".join(schema_texts)

# # # #         return Response(
# # # #             {
# # # #                 "message": "Files uploaded successfully.",
# # # #                 "files": uploaded_files_info,  # Keeping 'files' for chat services
# # # #                 "chat_message": combined_schema_text,  # Separate key for schema display
# # # #             },
# # # #             status=status.HTTP_201_CREATED,
# # # #         )

# # # #     def handle_chat(self, request):
# # # #         """
# # # #         Handles chat input by interacting with LangChain's conversation model.
# # # #         """
# # # #         user_input = request.data.get("message", "").strip()
# # # #         if not user_input:
# # # #             print("No user input received.")
# # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # #         print(f"User input: {user_input}")
# # # #         assistant_response = conversation_chain_chatgpt.run(user_input=user_input)
# # # #         print(f"Assistant response: {assistant_response}")
# # # #         return Response({"response": assistant_response})

# # # #     def format_schema_message(self, file_name, schema, target_column, entity_id_column):
# # # #         """
# # # #         Formats schema details for integration into the assistant conversation.
# # # #         """
# # # #         schema_text = (
# # # #             f"Dataset '{file_name}' uploaded successfully!\n\n"
# # # #             "Columns:\n" + ", ".join([col["column_name"] for col in schema]) + "\n\n"
# # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # #             f"Target Column Suggestion: {target_column}\n"
# # # #             f"Entity ID Column Suggestion: {entity_id_column}\n\n"
# # # #             "Please confirm if this schema matches your expectations."
# # # #         )
# # # #         return schema_text

# # # #     def append_assistant_message(self, message_text):
# # # #         """
# # # #         Appends assistant messages to conversation memory.
# # # #         """
# # # #         print(f"Appending assistant message to conversation: {message_text}")
# # # #         memory_chatgpt.save_context(
# # # #             {"user_input": memory_chatgpt.load_memory_variables({})["history"]},
# # # #             {"output": message_text},
# # # #         )



# # import os
# # import datetime
# # from io import BytesIO
# # from typing import Any, Dict, List

# # import boto3
# # import pandas as pd
# # import openai
# # from botocore.exceptions import ClientError, NoCredentialsError
# # from django.conf import settings
# # from rest_framework import status
# # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # from rest_framework.response import Response
# # from rest_framework.views import APIView
# # from langchain.chains import ConversationChain
# # from langchain.chat_models import ChatOpenAI
# # from langchain.prompts import PromptTemplate

# # from .models import FileSchema, UploadedFile
# # from .serializers import UploadedFileSerializer

# # # ===========================
# # # AWS Configuration
# # # ===========================
# # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')

# # # ===========================
# # # OpenAI Configuration
# # # ===========================
# # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # openai.api_key = OPENAI_API_KEY

# # # ===========================
# # # Initialize OpenAI LangChain model for ChatGPT
# # # ===========================
# # llm_chatgpt = ChatOpenAI(
# #     model="gpt-3.5-turbo-16k",
# #     temperature=0.7,
# #     openai_api_key=OPENAI_API_KEY,
# # )

# # # LangChain prompt with memory integration for ChatGPT
# # prompt_chatgpt = PromptTemplate(
# #     input_variables=["history", "user_input"],
# #     template=(
# #         "You are a helpful AI assistant. You guide users through defining predictive questions and refining goals.\n"
# #         "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
# #         "Steps:\n"
# #         "1. Discuss the Subject they want to predict.\n"
# #         "2. Confirm the Target Value they want to predict.\n"
# #         "3. Check if there's a specific time frame for the prediction.\n"
# #         "4. Reference the dataset schema if available.\n"
# #         "5. Summarize inputs before proceeding to model creation.\n\n"
# #         "Conversation history: {history}\n"
# #         "User input: {user_input}\n"
# #         "Assistant:"
# #     ),
# # )

# # conversation_chain_chatgpt = ConversationChain(
# #     llm=llm_chatgpt,
# #     prompt=prompt_chatgpt,
# #     input_key="user_input",
# # )

# # # ===========================
# # # Utility Functions
# # # ===========================
# # def get_s3_client():
# #     """
# #     Creates and returns an AWS S3 client.
# #     """
# #     return boto3.client(
# #         's3',
# #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# #         region_name=AWS_S3_REGION_NAME
# #     )

# # def get_glue_client():
# #     """
# #     Creates and returns an AWS Glue client.
# #     """
# #     return boto3.client(
# #         'glue',
# #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# #         region_name=AWS_S3_REGION_NAME
# #     )

# # def infer_column_dtype(series: pd.Series) -> str:
# #     """
# #     Infers the correct data type for a column by handling mixed types.
# #     """
# #     series = series.dropna().astype(str).str.strip()  # Handle mixed types and strip whitespace

# #     # Try boolean
# #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# #     unique_values = set(series.str.lower().unique())
# #     if unique_values.issubset(boolean_values):
# #         return "boolean"

# #     # Try integer
# #     try:
# #         int_series = pd.to_numeric(series, errors='raise')
# #         if (int_series % 1 == 0).all():
# #             int_min = int_series.min()
# #             int_max = int_series.max()
# #             if int_min >= -2147483648 and int_max <= 2147483647:
# #                 return "int"
# #             else:
# #                 return "bigint"
# #     except ValueError:
# #         pass

# #     # Try double
# #     try:
# #         pd.to_numeric(series, errors='raise', downcast='float')
# #         return "double"
# #     except ValueError:
# #         pass

# #     # Default to string
# #     return "string"

# # def suggest_target_column(df: pd.DataFrame) -> Any:
# #     """
# #     Suggests a target column based on numeric data types.
# #     """
# #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# #     return numeric_cols[0] if len(numeric_cols) > 0 else None

# # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# #     """
# #     Suggests an entity ID column based on uniqueness.
# #     """
# #     for col in df.columns:
# #         if df[col].is_unique:
# #             return col
# #     return None

# # # ===========================
# # # Unified ChatGPT API
# # # ===========================
# # class UnifiedChatGPTAPI(APIView):
# #     """
# #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# #     Endpoint: /api/chatgpt/
# #     """
# #     parser_classes = [MultiPartParser, FormParser, JSONParser]  # Include JSONParser
    
# #     def post(self, request):
# #         """
# #         Handles POST requests for both chat messages and file uploads.
# #         Differentiates based on the presence of files in the request.
# #         """
# #         if "file" in request.FILES:  # If files are present, handle file uploads
# #             return self.handle_file_upload(request.FILES.getlist("file"))
        
# #         # Else, handle chat message
# #         return self.handle_chat(request)
    
# #     def handle_file_upload(self, files: List[Any]):
# #         """
# #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# #         After processing, appends schema details to the chat messages.
# #         """
# #         if not files:
# #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# #         try:
# #             uploaded_files_info = []
# #             s3 = get_s3_client()
# #             glue = get_glue_client()

# #             for file in files:
# #                 # Validate file format
# #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# #                 # Read file into Pandas DataFrame
# #                 if file.name.lower().endswith('.csv'):
# #                     df = pd.read_csv(file)
# #                 else:
# #                     df = pd.read_excel(file)

# #                 # Normalize column headers
# #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# #                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# #                 # Infer schema
# #                 schema = [
# #                     {
# #                         "column_name": col,
# #                         "data_type": infer_column_dtype(df[col])
# #                     }
# #                     for col in df.columns
# #                 ]
# #                 print(f"Inferred schema: {schema}")  # Debugging statement

# #                 # Convert Boolean Columns to 'true'/'false' Strings
# #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# #                 for col in boolean_columns:
# #                     df[col] = df[col].astype(str).str.strip().str.lower()
# #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# #                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# #                 # Handle Duplicate Files Dynamically
# #                 file_name_base, file_extension = os.path.splitext(file.name)
# #                 file_name_base = file_name_base.lower().replace(' ', '_')

# #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# #                 if existing_file:
# #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# #                     file.name = new_file_name
# #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# #                 else:
# #                     print(f"File name is unique: {file.name}")  # Debugging statement

# #                 # Save Metadata to Database
# #                 file.seek(0)  # Reset file pointer before saving
# #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# #                 if file_serializer.is_valid():
# #                     file_instance = file_serializer.save()

# #                     # Convert DataFrame to CSV and Upload to S3
# #                     csv_buffer = BytesIO()
# #                     df.to_csv(csv_buffer, index=False)
# #                     csv_buffer.seek(0)
# #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# #                     file_key = f"uploads/{s3_file_name}"

# #                     # Upload to AWS S3
# #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)

# #                     # Generate file URL
# #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# #                     file_instance.file_url = file_url
# #                     file_instance.save()

# #                     # Save Schema to Database
# #                     FileSchema.objects.create(file=file_instance, schema=schema)

# #                     # Trigger AWS Glue Table Update
# #                     self.trigger_glue_update(file_name_base, schema, file_key)

# #                     # Append file info to response
# #                     uploaded_files_info.append({
# #                         'id': file_instance.id,
# #                         'name': file_instance.name,
# #                         'file_url': file_instance.file_url,
# #                         'schema': schema,
# #                         'suggestions': {  # Add suggestions based on the data
# #                             'target_column': suggest_target_column(df),
# #                             'entity_id_column': suggest_entity_id_column(df),
# #                         }
# #                     })
# #                 else:
# #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# #             return Response({
# #                 "message": "Files uploaded and processed successfully.",
# #                 "uploaded_files": uploaded_files_info
# #             }, status=status.HTTP_201_CREATED)

# #         except pd.errors.EmptyDataError:
# #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# #         except NoCredentialsError:
# #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# #         except ClientError as e:
# #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# #         except Exception as e:
# #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# #     def handle_chat(self, request):
# #         """
# #         Handles user chat messages using ChatGPT.
# #         """
# #         user_input = request.data.get("message", "").strip()
# #         user_id = request.data.get("user_id", "default_user")  # Optional: Track user sessions

# #         if not user_input:
# #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# #         assistant_response = conversation_chain_chatgpt.run(user_input=user_input)

# #         return Response({
# #             "response": assistant_response
# #         })

# #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# #         """
# #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# #         """
# #         glue = get_glue_client()
# #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# #         storage_descriptor = {
# #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# #             'Location': s3_location,
# #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# #             'SerdeInfo': {
# #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# #                 'Parameters': {
# #                     'field.delim': ',',
# #                     'skip.header.line.count': '1'
# #                 }
# #             }
# #         }
# #         try:
# #             glue.update_table(
# #                 DatabaseName='pa_user_datafiles_db',
# #                 TableInput={
# #                     'Name': table_name,
# #                     'StorageDescriptor': storage_descriptor,
# #                     'TableType': 'EXTERNAL_TABLE'
# #                 }
# #             )
# #             print(f"Glue table '{table_name}' updated successfully.")
# #         except glue.exceptions.EntityNotFoundException:
# #             print(f"Table '{table_name}' not found. Creating a new table...")
# #             glue.create_table(
# #                 DatabaseName='pa_user_datafiles_db',
# #                 TableInput={
# #                     'Name': table_name,
# #                     'StorageDescriptor': storage_descriptor,
# #                     'TableType': 'EXTERNAL_TABLE'
# #                 }
# #             )
# #             print(f"Glue table '{table_name}' created successfully.")
# #         except Exception as e:
# #             print(f"Glue operation failed: {str(e)}")


# import os
# import datetime
# from io import BytesIO
# from typing import Any, Dict, List

# import boto3
# import pandas as pd
# import openai
# from botocore.exceptions import ClientError, NoCredentialsError
# from django.conf import settings
# from rest_framework import status
# from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from langchain.chains import ConversationChain
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate

# from .models import FileSchema, UploadedFile
# from .serializers import UploadedFileSerializer

# # ===========================
# # AWS Configuration
# # ===========================
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')

# # ===========================
# # OpenAI Configuration
# # ===========================
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY

# # ===========================
# # Initialize OpenAI LangChain model for ChatGPT
# # ===========================
# llm_chatgpt = ChatOpenAI(
#     model="gpt-3.5-turbo-16k",
#     temperature=0.7,
#     openai_api_key=OPENAI_API_KEY,
# )

# # LangChain prompt with memory integration for ChatGPT
# prompt_chatgpt = PromptTemplate(
#     input_variables=["history", "user_input"],
#     template=(
#         "You are a helpful AI assistant. You guide users through defining predictive questions and refining goals.\n"
#         "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
#         "Steps:\n"
#         "1. Discuss the Subject they want to predict.\n"
#         "2. Confirm the Target Value they want to predict.\n"
#         "3. Check if there's a specific time frame for the prediction.\n"
#         "4. Reference the dataset schema if available.\n"
#         "5. Summarize inputs before proceeding to model creation.\n\n"
#         "Conversation history: {history}\n"
#         "User input: {user_input}\n"
#         "Assistant:"
#     ),
# )

# conversation_chain_chatgpt = ConversationChain(
#     llm=llm_chatgpt,
#     prompt=prompt_chatgpt,
#     input_key="user_input",
# )

# # ===========================
# # Utility Functions
# # ===========================
# def get_s3_client():
#     """
#     Creates and returns an AWS S3 client.
#     """
#     return boto3.client(
#         's3',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )

# def get_glue_client():
#     """
#     Creates and returns an AWS Glue client.
#     """
#     return boto3.client(
#         'glue',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )

# def infer_column_dtype(series: pd.Series) -> str:
#     """
#     Infers the correct data type for a column by handling mixed types.
#     """
#     series = series.dropna().astype(str).str.strip()  # Handle mixed types and strip whitespace

#     # Try boolean
#     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
#     unique_values = set(series.str.lower().unique())
#     if unique_values.issubset(boolean_values):
#         return "boolean"

#     # Try integer
#     try:
#         int_series = pd.to_numeric(series, errors='raise')
#         if (int_series % 1 == 0).all():
#             int_min = int_series.min()
#             int_max = int_series.max()
#             if int_min >= -2147483648 and int_max <= 2147483647:
#                 return "int"
#             else:
#                 return "bigint"
#     except ValueError:
#         pass

#     # Try double
#     try:
#         pd.to_numeric(series, errors='raise', downcast='float')
#         return "double"
#     except ValueError:
#         pass

#     # Default to string
#     return "string"

# def suggest_target_column(df: pd.DataFrame) -> Any:
#     """
#     Suggests a target column based on numeric data types.
#     """
#     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
#     return numeric_cols[0] if len(numeric_cols) > 0 else None

# def suggest_entity_id_column(df: pd.DataFrame) -> Any:
#     """
#     Suggests an entity ID column based on uniqueness.
#     """
#     for col in df.columns:
#         if df[col].is_unique:
#             return col
#     return None

# # ===========================
# # Unified ChatGPT API
# # ===========================
# # class UnifiedChatGPTAPI(APIView):
# #     """
# #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# #     Endpoint: /api/chatgpt/
# #     """
# #     parser_classes = [MultiPartParser, FormParser, JSONParser]  # Include JSONParser
    
# #     def post(self, request):
# #         """
# #         Handles POST requests for both chat messages and file uploads.
# #         Differentiates based on the presence of files in the request.
# #         """
# #         if "file" in request.FILES:  # If files are present, handle file uploads
# #             return self.handle_file_upload(request.FILES.getlist("file"))
        
# #         # Else, handle chat message
# #         return self.handle_chat(request)
    
# #     def handle_file_upload(self, files: List[Any]):
# #         """
# #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# #         After processing, appends schema details to the chat messages.
# #         """
# #         if not files:
# #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# #         try:
# #             uploaded_files_info = []
# #             s3 = get_s3_client()
# #             glue = get_glue_client()

# #             for file in files:
# #                 # Validate file format
# #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# #                 # Read file into Pandas DataFrame
# #                 if file.name.lower().endswith('.csv'):
# #                     df = pd.read_csv(file)
# #                 else:
# #                     df = pd.read_excel(file)

# #                 # Normalize column headers
# #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# #                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# #                 # Infer schema
# #                 schema = [
# #                     {
# #                         "column_name": col,
# #                         "data_type": infer_column_dtype(df[col])
# #                     }
# #                     for col in df.columns
# #                 ]
# #                 print(f"Inferred schema: {schema}")  # Debugging statement

# #                 # Convert Boolean Columns to 'true'/'false' Strings
# #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# #                 for col in boolean_columns:
# #                     df[col] = df[col].astype(str).str.strip().str.lower()
# #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# #                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# #                 # Handle Duplicate Files Dynamically
# #                 file_name_base, file_extension = os.path.splitext(file.name)
# #                 file_name_base = file_name_base.lower().replace(' ', '_')

# #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# #                 if existing_file:
# #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# #                     file.name = new_file_name
# #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# #                 else:
# #                     print(f"File name is unique: {file.name}")  # Debugging statement

# #                 # Save Metadata to Database
# #                 file.seek(0)  # Reset file pointer before saving
# #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# #                 if file_serializer.is_valid():
# #                     file_instance = file_serializer.save()

# #                     # Convert DataFrame to CSV and Upload to S3
# #                     csv_buffer = BytesIO()
# #                     df.to_csv(csv_buffer, index=False)
# #                     csv_buffer.seek(0)
# #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# #                     file_key = f"uploads/{s3_file_name}"

# #                     # Upload to AWS S3
# #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)

# #                     # Generate file URL
# #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# #                     file_instance.file_url = file_url
# #                     file_instance.save()

# #                     # Save Schema to Database
# #                     FileSchema.objects.create(file=file_instance, schema=schema)

# #                     # Trigger AWS Glue Table Update
# #                     self.trigger_glue_update(file_name_base, schema, file_key)

# #                     # Append file info to response
# #                     uploaded_files_info.append({
# #                         'id': file_instance.id,
# #                         'name': file_instance.name,
# #                         'file_url': file_instance.file_url,
# #                         'schema': schema,
# #                         'suggestions': {  # Add suggestions based on the data
# #                             'target_column': suggest_target_column(df),
# #                             'entity_id_column': suggest_entity_id_column(df),
# #                         }
# #                     })
# #                 else:
# #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# #             # Format schema messages to append to assistant conversation
# #             for uploaded_file in uploaded_files_info:
# #                 schema_text = self.format_schema_message(uploaded_file)
# #                 print(f"Schema formatted for chat: {schema_text}")  # Debugging statement

# #             return Response({
# #                 "message": "Files uploaded and processed successfully.",
# #                 "uploaded_files": uploaded_files_info
# #             }, status=status.HTTP_201_CREATED)

# #         except pd.errors.EmptyDataError:
# #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# #         except NoCredentialsError:
# #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# #         except ClientError as e:
# #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# #         except Exception as e:
# #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# class UnifiedChatGPTAPI(APIView):
#     """
#     Unified API for handling ChatGPT-based chat interactions and file uploads.
#     Endpoint: /api/chatgpt/
#     """
#     parser_classes = [MultiPartParser, FormParser, JSONParser]  # Include JSONParser

#     def post(self, request):
#         """
#         Handles POST requests for both chat messages and file uploads.
#         Differentiates based on the presence of files in the request.
#         """
#         if "file" in request.FILES:  # If files are present, handle file uploads
#             return self.handle_file_upload(request.FILES.getlist("file"))

#         # Else, handle chat message
#         return self.handle_chat(request)

#     def handle_file_upload(self, files: List[Any]):
#         """
#         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
#         After processing, appends schema details to the chat messages.
#         """
#         if not files:
#             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             uploaded_files_info = []
#             s3 = get_s3_client()
#             glue = get_glue_client()

#             for file in files:
#                 # Validate file format
#                 if not file.name.lower().endswith(('.csv', '.xlsx')):
#                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

#                 # Read file into Pandas DataFrame
#                 if file.name.lower().endswith('.csv'):
#                     df = pd.read_csv(file)
#                 else:
#                     df = pd.read_excel(file)

#                 # Normalize column headers
#                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
#                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

#                 # Infer schema
#                 schema = [
#                     {
#                         "column_name": col,
#                         "data_type": infer_column_dtype(df[col])
#                     }
#                     for col in df.columns
#                 ]
#                 print(f"Inferred schema: {schema}")  # Debugging statement

#                 # Convert Boolean Columns to 'true'/'false' Strings
#                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
#                 for col in boolean_columns:
#                     df[col] = df[col].astype(str).str.strip().str.lower()
#                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
#                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

#                 # Handle Duplicate Files Dynamically
#                 file_name_base, file_extension = os.path.splitext(file.name)
#                 file_name_base = file_name_base.lower().replace(' ', '_')

#                 existing_file = UploadedFile.objects.filter(name=file.name).first()
#                 if existing_file:
#                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
#                     file.name = new_file_name
#                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
#                 else:
#                     print(f"File name is unique: {file.name}")  # Debugging statement

#                 # Save Metadata to Database
#                 file.seek(0)  # Reset file pointer before saving
#                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
#                 if file_serializer.is_valid():
#                     file_instance = file_serializer.save()

#                     # Convert DataFrame to CSV and Upload to S3
#                     csv_buffer = BytesIO()
#                     df.to_csv(csv_buffer, index=False)
#                     csv_buffer.seek(0)
#                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
#                     file_key = f"uploads/{s3_file_name}"

#                     # Upload to AWS S3
#                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)

#                     # Generate file URL
#                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
#                     file_instance.file_url = file_url
#                     file_instance.save()

#                     # Save Schema to Database
#                     FileSchema.objects.create(file=file_instance, schema=schema)

#                     # Trigger AWS Glue Table Update
#                     self.trigger_glue_update(file_name_base, schema, file_key)

#                     # Append file info to response
#                     uploaded_files_info.append({
#                         'id': file_instance.id,
#                         'name': file_instance.name,
#                         'file_url': file_instance.file_url,
#                         'schema': schema,
#                         'suggestions': {  # Add suggestions based on the data
#                             'target_column': suggest_target_column(df),
#                             'entity_id_column': suggest_entity_id_column(df),
#                         }
#                     })
#                 else:
#                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#             # Format schema messages to append to assistant conversation
#             schema_messages = [self.format_schema_message(uploaded_file) for uploaded_file in uploaded_files_info]
#             combined_schema_message = "\n\n".join(schema_messages)
#             print(f"Combined schema message for chat: {combined_schema_message}")  # Debugging statement

#             return Response({
#                 "message": "Files uploaded and processed successfully.",
#                 "uploaded_files": uploaded_files_info,
#                 "chat_message": combined_schema_message  # Include chat_message in the response
#             }, status=status.HTTP_201_CREATED)

#         except pd.errors.EmptyDataError:
#             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
#         except NoCredentialsError:
#             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
#         except ClientError as e:
#             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#         except Exception as e:
#             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
#             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

#     def handle_chat(self, request):
#         """
#         Handles user chat messages using ChatGPT.
#         """
#         user_input = request.data.get("message", "").strip()
#         user_id = request.data.get("user_id", "default_user")  # Optional: Track user sessions

#         if not user_input:
#             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

#         assistant_response = conversation_chain_chatgpt.run(user_input=user_input)

#         return Response({
#             "response": assistant_response
#         })

#     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
#         """
#         Formats the schema information to be appended as an assistant message in the chat.
#         """
#         schema = uploaded_file['schema']
#         schema_text = (
#             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
#             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
#             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
#             f"Target Column Suggestion: {uploaded_file['suggestions']['target_column']}\n"
#             f"Entity ID Column Suggestion: {uploaded_file['suggestions']['entity_id_column']}\n\n"
#             "Please confirm:\n\n"
#             "- Is the Target Column correct?\n"
#             "- Is the Entity ID Column correct?\n"
#             '(Reply "yes" or provide the correct column names.)'
#         )
#         return schema_text

#     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
#         """
#         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
#         """
#         glue = get_glue_client()
#         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
#         storage_descriptor = {
#             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
#             'Location': s3_location,
#             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
#             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
#             'SerdeInfo': {
#                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
#                 'Parameters': {
#                     'field.delim': ',',
#                     'skip.header.line.count': '1'
#                 }
#             }
#         }
#         try:
#             glue.update_table(
#                 DatabaseName='pa_user_datafiles_db',
#                 TableInput={
#                     'Name': table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             print(f"Glue table '{table_name}' updated successfully.")
#         except glue.exceptions.EntityNotFoundException:
#             print(f"Table '{table_name}' not found. Creating a new table...")
#             glue.create_table(
#                 DatabaseName='pa_user_datafiles_db',
#                 TableInput={
#                     'Name': table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             print(f"Glue table '{table_name}' created successfully.")
#         except Exception as e:
#             print(f"Glue operation failed: {str(e)}")


import os
import datetime
from io import BytesIO
from typing import Any, Dict, List

import boto3
import pandas as pd
import openai
from botocore.exceptions import ClientError, NoCredentialsError
from django.conf import settings
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from .models import FileSchema, UploadedFile
from .serializers import UploadedFileSerializer

# ===========================
# AWS Configuration
# ===========================
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')

# ===========================
# OpenAI Configuration
# ===========================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# ===========================
# Initialize OpenAI LangChain model for ChatGPT
# ===========================
llm_chatgpt = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY,
)

# LangChain prompt with memory integration for ChatGPT
prompt_chatgpt = PromptTemplate(
    input_variables=["history", "user_input"],
    template=(
        "You are a helpful AI assistant. You guide users through defining predictive questions and refining goals.\n"
        "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
        "Steps:\n"
        "1. Discuss the Subject they want to predict.\n"
        "2. Confirm the Target Value they want to predict.\n"
        "3. Check if there's a specific time frame for the prediction.\n"
        "4. Reference the dataset schema if available.\n"
        "5. Summarize inputs before proceeding to model creation.\n\n"
        "Conversation history: {history}\n"
        "User input: {user_input}\n"
        "Assistant:"
    ),
)

conversation_chain_chatgpt = ConversationChain(
    llm=llm_chatgpt,
    prompt=prompt_chatgpt,
    input_key="user_input",
)

# ===========================
# Utility Functions
# ===========================
def get_s3_client():
    """
    Creates and returns an AWS S3 client.
    """
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def get_glue_client():
    """
    Creates and returns an AWS Glue client.
    """
    return boto3.client(
        'glue',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def infer_column_dtype(series: pd.Series) -> str:
    """
    Infers the correct data type for a column by handling mixed types.
    """
    series = series.dropna().astype(str).str.strip()  # Handle mixed types and strip whitespace

    # Try boolean
    boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
    unique_values = set(series.str.lower().unique())
    if unique_values.issubset(boolean_values):
        return "boolean"

    # Try integer
    try:
        int_series = pd.to_numeric(series, errors='raise')
        if (int_series % 1 == 0).all():
            int_min = int_series.min()
            int_max = int_series.max()
            if int_min >= -2147483648 and int_max <= 2147483647:
                return "int"
            else:
                return "bigint"
    except ValueError:
        pass

    # Try double
    try:
        pd.to_numeric(series, errors='raise', downcast='float')
        return "double"
    except ValueError:
        pass

    # Default to string
    return "string"

def suggest_target_column(df: pd.DataFrame) -> Any:
    """
    Suggests a target column based on numeric data types.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    return numeric_cols[0] if len(numeric_cols) > 0 else None

def suggest_entity_id_column(df: pd.DataFrame) -> Any:
    """
    Suggests an entity ID column based on uniqueness.
    """
    for col in df.columns:
        if df[col].is_unique:
            return col
    return None

# ===========================
# Unified ChatGPT API
# ===========================
class UnifiedChatGPTAPI(APIView):
    """
    Unified API for handling ChatGPT-based chat interactions and file uploads.
    Endpoint: /api/chatgpt/
    """
    parser_classes = [MultiPartParser, FormParser, JSONParser]  # Include JSONParser

    def post(self, request):
        """
        Handles POST requests for both chat messages and file uploads.
        Differentiates based on the presence of files in the request.
        """
        if "file" in request.FILES:  # If files are present, handle file uploads
            return self.handle_file_upload(request.FILES.getlist("file"))

        # Else, handle chat message
        return self.handle_chat(request)

    def handle_file_upload(self, files: List[Any]):
        """
        Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
        After processing, appends schema details to the chat messages.
        """
        if not files:
            return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            uploaded_files_info = []
            s3 = get_s3_client()
            glue = get_glue_client()

            for file in files:
                # Validate file format
                if not file.name.lower().endswith(('.csv', '.xlsx')):
                    return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

                # Read file into Pandas DataFrame
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)

                # Normalize column headers
                df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                # print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

                # Infer schema
                schema = [
                    {
                        "column_name": col,
                        "data_type": infer_column_dtype(df[col])
                    }
                    for col in df.columns
                ]
                # print(f"Inferred schema: {schema}")  # Debugging statement

                # Convert Boolean Columns to 'true'/'false' Strings
                boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
                for col in boolean_columns:
                    df[col] = df[col].astype(str).str.strip().str.lower()
                    df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
                # print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

                # Handle Duplicate Files Dynamically
                file_name_base, file_extension = os.path.splitext(file.name)
                file_name_base = file_name_base.lower().replace(' ', '_')

                existing_file = UploadedFile.objects.filter(name=file.name).first()
                if existing_file:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
                    file.name = new_file_name
                    print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
                else:
                    print(f"File name is unique: {file.name}")  # Debugging statement

                # Save Metadata to Database
                file.seek(0)  # Reset file pointer before saving
                file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
                if file_serializer.is_valid():
                    file_instance = file_serializer.save()

                    # Convert DataFrame to CSV and Upload to S3
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    s3_file_name = os.path.splitext(file.name)[0] + '.csv'
                    file_key = f"uploads/{s3_file_name}"

                    # Upload to AWS S3
                    s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)

                    # Generate file URL
                    file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
                    file_instance.file_url = file_url
                    file_instance.save()

                    # Save Schema to Database
                    FileSchema.objects.create(file=file_instance, schema=schema)

                    # Trigger AWS Glue Table Update
                    self.trigger_glue_update(file_name_base, schema, file_key)

                    # Append file info to response
                    uploaded_files_info.append({
                        'id': file_instance.id,
                        'name': file_instance.name,
                        'file_url': file_instance.file_url,
                        'schema': schema,
                        'suggestions': {  # Add suggestions based on the data
                            'target_column': suggest_target_column(df),
                            'entity_id_column': suggest_entity_id_column(df),
                        }
                    })
                else:
                    return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            # Format schema messages to append to assistant conversation
            schema_messages = [self.format_schema_message(uploaded_file) for uploaded_file in uploaded_files_info]
            combined_schema_message = "\n\n".join(schema_messages)
            # print(f"Combined schema message for chat: {combined_schema_message}")  # Debugging statement

            return Response({
                "message": "Files uploaded and processed successfully.",
                "uploaded_files": uploaded_files_info,
                "chat_message": combined_schema_message  # Include chat_message in the response
            }, status=status.HTTP_201_CREATED)

        except pd.errors.EmptyDataError:
            return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
        except NoCredentialsError:
            return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
        except ClientError as e:
            return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
            return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

    def handle_chat(self, request):
        """
        Handles user chat messages using ChatGPT.
        """
        user_input = request.data.get("message", "").strip()
        user_id = request.data.get("user_id", "default_user")  # Optional: Track user sessions

        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

        assistant_response = conversation_chain_chatgpt.run(user_input=user_input)

        return Response({
            "response": assistant_response
        })

    def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
        """
        Formats the schema information to be appended as an assistant message in the chat.
        """
        schema = uploaded_file['schema']
        schema_text = (
            f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
            "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
            "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
            f"Target Column Suggestion: {uploaded_file['suggestions']['target_column']}\n"
            f"Entity ID Column Suggestion: {uploaded_file['suggestions']['entity_id_column']}\n\n"
            "Please confirm:\n\n"
            "- Is the Target Column correct?\n"
            "- Is the Entity ID Column correct?\n"
            '(Reply "yes" or provide the correct column names.)'
        )
        return schema_text

    def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
        """
        Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
        """
        glue = get_glue_client()
        s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
        storage_descriptor = {
            'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
            'Location': s3_location,
            'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
            'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
            'SerdeInfo': {
                'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                'Parameters': {
                    'field.delim': ',',
                    'skip.header.line.count': '1'
                }
            }
        }
        try:
            glue.update_table(
                DatabaseName='pa_user_datafiles_db',
                TableInput={
                    'Name': table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            print(f"Glue table '{table_name}' updated successfully.")
        except glue.exceptions.EntityNotFoundException:
            print(f"Table '{table_name}' not found. Creating a new table...")
            glue.create_table(
                DatabaseName='pa_user_datafiles_db',
                TableInput={
                    'Name': table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            print(f"Glue table '{table_name}' created successfully.")
        except Exception as e:
            print(f"Glue operation failed: {str(e)}")
