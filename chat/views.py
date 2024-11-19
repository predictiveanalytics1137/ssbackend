# from django.shortcuts import render

# # Create your views here.
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load LLaMA model
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from .models import ChatHistory

# # login("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") 

# # Load LLaMA model and tokenizer
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# # Load your Hugging Face access token
# # HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')  # Store token in environment variable
# # HUGGING_FACE_TOKEN = "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"  # Store token in environment variable

# # # Load LLaMA model and tokenizer with the token
# # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=HUGGING_FACE_TOKEN)
# # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=HUGGING_FACE_TOKEN)

# # @api_view(['POST'])
# # def chat_response(request):

# #     user_input = request.data.get('message', '')
# #     print(user_input)
# #     print("user_input")

# #     # Tokenize user input
# #     # inputs = tokenizer(user_input, return_tensors="pt")

# #     # # Generate response from LLaMA model
# #     # outputs = model.generate(inputs.input_ids, max_length=100)
# #     # response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# #     # return Response({"response": response_text})
# #     return Response({"response": "hi how are you"})


# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.parsers import MultiPartParser, FormParser
# from .models import UploadedFile
# from .serializers import UploadedFileSerializer
# from django.http import Http404
# import boto3
# from botocore.exceptions import NoCredentialsError, ClientError
# from django.conf import settings
# import boto3
# import pandas as pd
# from io import StringIO
# # api/views.py
# from rest_framework.decorators import api_view
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Load the model and tokenizer
# hf_token = "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"
# model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # def generate_response(prompt, max_length=150, temperature=0.6, top_p=0.8):
# #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
# #     output = model.generate(
# #         **inputs,
# #         max_length=max_length,
# #         temperature=temperature,
# #         top_p=top_p,
# #         do_sample=True
# #     )
# #     response = tokenizer.decode(output[0], skip_special_tokens=True)
# #     return response

# # @api_view(['POST'])
# # def chat_response(request):
# #     print("generating response")
# #     prompt = request.data.get("message", "")
# #     if not prompt:
# #         return Response({"error": "No message provided"}, status=400)

# #     # Generate response using the model
# #     response_text = generate_response(prompt)
# #     return Response({"response": response_text})


# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the model and tokenizer
# hf_token = "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"
# model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# question_flow = [
#     {
#         "id": 1,
#         "question": "What is the subject of your prediction (e.g., customers, users, products)?",
#         "key": "subject",
#         "next_question_id": 2
#     },
#     {
#         "id": 2,
#         "question": "What is the target value you want to predict for each customer?",
#         "key": "target_value",
#         "next_question_id": 3
#     },
#     {
#         "id": 3,
#         "question": "Is there a specific future time horizon relevant to the prediction (e.g., next 7 days)?",
#         "key": "time_horizon",
#         "next_question_id": 4
#     },
#     {
#         "id": 4,
#         "question": "Do you have a unique identifier column for each customer in your data?",
#         "key": "unique_identifier",
#         "next_question_id": None  # End of flow
#     }
# ]

# # In-memory storage for simplicity (use a database for production)
# user_sessions = {}

# def generate_response(prompt, max_length=150, temperature=0.6, top_p=0.8):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     output = model.generate(
#         **inputs,
#         max_length=max_length,
#         temperature=temperature,
#         top_p=top_p,
#         do_sample=True
#     )
#     response = tokenizer.decode(output[0], skip_special_tokens=True)
#     return response

# @api_view(['POST'])
# def chat_response(request):
#     user_id = request.data.get("user_id", "default_user")  # Track user session
#     message = request.data.get("message", "")

#     # Initialize session if new user
#     if user_id not in user_sessions:
#         user_sessions[user_id] = {
#             "current_question_id": 1,
#             "answers": {}
#         }

#     session = user_sessions[user_id]
#     current_question_id = session["current_question_id"]

#     # Handle mandatory question flow
#     if current_question_id:
#         current_question = next(q for q in question_flow if q["id"] == current_question_id)
#         key = current_question["key"]
#         session["answers"][key] = message  # Save the user's answer
#         session["current_question_id"] = current_question["next_question_id"]  # Move to the next question

#         # Check if there's a next question
#         if session["current_question_id"]:
#             next_question = next(q for q in question_flow if q["id"] == session["current_question_id"])
#             return Response({"response": next_question["question"], "mandatory": True})
#         else:
#             return Response({"response": "Thank you for answering all mandatory questions. Feel free to ask general questions now.", "mandatory": False})

#     # Handle general interaction with LLaMA
#     response_text = generate_response(message)
#     return Response({"response": response_text, "mandatory": False})






# class FileUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         if 'file' not in request.FILES:
#             return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

#         uploaded_file = request.FILES['file']

#         # Validate the file size
#         if uploaded_file.size == 0:
#             return Response({'error': 'File is empty'}, status=status.HTTP_400_BAD_REQUEST)

#         # Validate the file type
#         if not uploaded_file.name.endswith('.csv'):
#             return Response({'error': 'Only CSV files are allowed'}, status=status.HTTP_400_BAD_REQUEST)

#         # Create a serializer instance
#         file_serializer = UploadedFileSerializer(data={'name': uploaded_file.name, 'file': uploaded_file})

#         if file_serializer.is_valid():
#             try:
#                 # Save metadata to the database
#                 file_instance = file_serializer.save()

#                 # Upload file to S3
#                 s3 = boto3.client(
#                     's3',
#                     aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#                     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
#                     region_name=settings.AWS_S3_REGION_NAME
#                 )

#                 s3_key = f"uploads/{uploaded_file.name}"  # Define the S3 key
#                 uploaded_file.seek(0)  # Reset file pointer
#                 s3.upload_fileobj(
#                     uploaded_file,
#                     settings.AWS_STORAGE_BUCKET_NAME,
#                     s3_key
#                 )

#                 # Generate the full S3 URL
#                 full_file_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"

#                 print("full_file_url")
#                 # Update the file instance with the S3 URL
#                 file_instance.file_url = full_file_url
#                 file_instance.save()
#                 print("print saved file instance")

#                 # Validate the CSV and extract schema
#                 obj = s3.get_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_key)
#                 csv_content = obj['Body'].read().decode('utf-8')
#                 df = pd.read_csv(StringIO(csv_content))
#                 print(df.head(5))

#                 # Generate schema
#                 data_schema = pd.DataFrame({
#                     'Column Name': df.columns,
#                     'Data Type': [str(dtype) for dtype in df.dtypes]
#                 })

#                 # Map data types to user-friendly names
#                 data_schema['Data Type'] = data_schema['Data Type'].replace({
#                     'int64': 'Long',
#                     'float64': 'Double',
#                     'object': 'String',
#                     'bool': 'Boolean'
#                 })

#                 schema_result = data_schema.to_dict(orient='records')
#                 # print(schema_result)
#                 # print("schema_result")

#                 # Return success response
#                 return Response({
#                     'id': file_instance.id,
#                     'name': file_instance.name,
#                     'file_url': file_instance.file_url,
#                     'uploaded_at': file_instance.uploaded_at,
#                     'schema': schema_result
#                 }, status=status.HTTP_201_CREATED)

#             except NoCredentialsError:
#                 return Response({'error': 'AWS credentials not available'}, status=status.HTTP_403_FORBIDDEN)
#             except ClientError as e:
#                 return Response({'error': f'S3 upload failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#             except Exception as e:
#                 return Response({'error': f'File validation failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

#         return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




# # File Delete View
# # File delete view
# class FileDeleteView(APIView):
#     def delete(self, request, pk, *args, **kwargs):
#         try:
#             # Fetch the file instance from the database using the primary key (pk)
#             file_instance = UploadedFile.objects.get(pk=pk)
#         except UploadedFile.DoesNotExist:
#             raise Http404

#         # Extract the exact S3 key used during the upload
#         s3_key = file_instance.file_url.split('.amazonaws.com/')[1]  # Get key from the full URL

#         # Initialize S3 client
#         s3 = boto3.client(
#             's3',
#             aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#             aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
#             region_name=settings.AWS_S3_REGION_NAME
#         )
        
#         try:
#             # Delete file from S3 bucket
#             s3.delete_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_key)
#             print(f"Deleted file from S3 bucket: {s3_key}")
#         except ClientError as e:
#             print(f"Error deleting file from S3: {e}")
#             return Response({'error': 'Failed to delete file from S3'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         # Delete file record from the database
#         file_instance.delete()

#         return Response(status=status.HTTP_204_NO_CONTENT)

from rest_framework.decorators import api_view
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .models import UploadedFile, ChatHistory
from .serializers import UploadedFileSerializer
from django.http import Http404
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from django.conf import settings
import pandas as pd
from io import StringIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the LLaMA model and tokenizer
hf_token = "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"
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the question flow
question_flow = [
    {
        "id": 1,
        "question": "What is the subject of your prediction (e.g., customers, users, products)?",
        "key": "subject",
        "next_question_id": 2,
        "expected_type": str
    },
    {
        "id": 2,
        "question": "What is the target value you want to predict for each subject?",
        "key": "target_value",
        "next_question_id": 3,
        "expected_type": str
    },
    {
        "id": 3,
        "question": "Is there a specific future time horizon relevant to the prediction (e.g., next 7 days)?",
        "key": "time_horizon",
        "next_question_id": 4,
        "expected_type": str
    },
    {
        "id": 4,
        "question": "Do you have a unique identifier column for each subject in your data?",
        "key": "unique_identifier",
        "next_question_id": None,  # End of flow
        "expected_type": str
    }
]

# In-memory storage for simplicity (use a database for production)
user_sessions = {}

def generate_response(prompt, max_length=150, temperature=0.6, top_p=0.8):
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
    user_id = request.data.get("user_id", "default_user")  # Track user session
    message = request.data.get("message", "")

    # Initialize session if new user
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "current_question_id": 1,
            "answers": {}
        }

    session = user_sessions[user_id]
    current_question_id = session["current_question_id"]

    # Check if the user wants to proceed with the question flow or have a general conversation
    if current_question_id:
        current_question = next(q for q in question_flow if q["id"] == current_question_id)
        key = current_question["key"]
        expected_type = current_question["expected_type"]

        # Validate the user's answer
        try:
            value = expected_type(message)
            session["answers"][key] = value  # Save the user's answer
            session["current_question_id"] = current_question["next_question_id"]  # Move to the next question
        except ValueError:
            return Response({"response": f"Please provide a valid {expected_type.__name__} value for the question."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if there's a next question
        if session["current_question_id"]:
            next_question = next(q for q in question_flow if q["id"] == session["current_question_id"])
            return Response({"response": next_question["question"], "mandatory": True})
        else:
            # Process the user's answers and generate the predictive model
            subject = session["answers"]["subject"]
            target_value = session["answers"]["target_value"]
            time_horizon = session["answers"]["time_horizon"]
            unique_identifier = session["answers"]["unique_identifier"]

            # Use the gathered information to create a predictive model
            # and provide the user with the necessary next steps
            response_text = f"Thank you for answering all mandatory questions. Based on your inputs, we will create a predictive model to forecast {target_value} for your {subject} data. The next steps are: 1) Upload your CSV file containing the relevant data columns. 2) Review the data schema and make any necessary adjustments. 3) Train the model and evaluate its performance."
            return Response({"response": response_text, "mandatory": False})
    else:
        # Handle general interaction with LLaMA
        response_text = generate_response(message)
        return Response({"response": response_text, "mandatory": False})

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        if 'file' not in request.FILES:
            return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = request.FILES['file']

        # Validate the file size
        if uploaded_file.size == 0:
            return Response({'error': 'File is empty'}, status=status.HTTP_400_BAD_REQUEST)

        # Validate the file type
        if not uploaded_file.name.endswith('.csv'):
            return Response({'error': 'Only CSV files are allowed'}, status=status.HTTP_400_BAD_REQUEST)

        # Create a serializer instance
        file_serializer = UploadedFileSerializer(data={'name': uploaded_file.name, 'file': uploaded_file})

        if file_serializer.is_valid():
            try:
                # Save metadata to the database
                file_instance = file_serializer.save()

                # Upload file to S3
                s3 = boto3.client(
                    's3',
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                    region_name=settings.AWS_S3_REGION_NAME
                )

                s3_key = f"uploads/{uploaded_file.name}"  # Define the S3 key
                uploaded_file.seek(0)  # Reset file pointer
                s3.upload_fileobj(
                    uploaded_file,
                    settings.AWS_STORAGE_BUCKET_NAME,
                    s3_key
                )

                # Generate the full S3 URL
                full_file_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{s3_key}"

                # Update the file instance with the S3 URL
                file_instance.file_url = full_file_url
                file_instance.save()

                # Validate the CSV and extract schema
                obj = s3.get_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_key)
                csv_content = obj['Body'].read().decode('utf-8')
                df = pd.read_csv(StringIO(csv_content))
                print(df.head(5))

                # Generate schema
                data_schema = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': [str(dtype) for dtype in df.dtypes]
                })

                # Map data types to user-friendly names
                data_schema['Data Type'] = data_schema['Data Type'].replace({
                    'int64': 'Long',
                    'float64': 'Double',
                    'object': 'String',
                    'bool': 'Boolean'
                })

                schema_result = data_schema.to_dict(orient='records')

                # Return success response
                return Response({
                    'id': file_instance.id,
                    'name': file_instance.name,
                    'file_url': file_instance.file_url,
                    'uploaded_at': file_instance.uploaded_at,
                    'schema': schema_result
                }, status=status.HTTP_201_CREATED)

            except NoCredentialsError:
                return Response({'error': 'AWS credentials not available'}, status=status.HTTP_403_FORBIDDEN)
            except ClientError as e:
                return Response({'error': f'S3 upload failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                return Response({'error': f'File validation failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

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