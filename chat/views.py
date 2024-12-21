
# # # # # # # # # # # #chat/views.py

# # # # # # # # # # # import os
# # # # # # # # # # # import datetime
# # # # # # # # # # # from io import BytesIO
# # # # # # # # # # # from typing import Any, Dict, List
# # # # # # # # # # # from langchain.schema import AIMessage, HumanMessage
# # # # # # # # # # # import boto3
# # # # # # # # # # # import pandas as pd
# # # # # # # # # # # import openai
# # # # # # # # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # # # # # # # from django.conf import settings
# # # # # # # # # # # from rest_framework import status
# # # # # # # # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # # from rest_framework.views import APIView
# # # # # # # # # # # from langchain.chains import ConversationChain
# # # # # # # # # # # from langchain.chat_models import ChatOpenAI
# # # # # # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # # # # # from langchain.memory import ConversationBufferMemory
# # # # # # # # # # # from .models import FileSchema, UploadedFile
# # # # # # # # # # # from .serializers import UploadedFileSerializer

# # # # # # # # # # # # ===========================
# # # # # # # # # # # # AWS Configuration
# # # # # # # # # # # # ===========================
# # # # # # # # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # # # # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # # # # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')

# # # # # # # # # # # # ===========================
# # # # # # # # # # # # OpenAI Configuration
# # # # # # # # # # # # ===========================
# # # # # # # # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # # # # # # # openai.api_key = OPENAI_API_KEY

# # # # # # # # # # # # ===========================
# # # # # # # # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # # # # # # # ===========================
# # # # # # # # # # # llm_chatgpt = ChatOpenAI(
# # # # # # # # # # #     model="gpt-3.5-turbo-16k",
# # # # # # # # # # #     #  model="gpt-4",
    
# # # # # # # # # # #     # model = "gpt-4",

# # # # # # # # # # #     temperature=0.7,
# # # # # # # # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # # # # # # # )



# # # # # # # # # # # user_conversations = {}

# # # # # # # # # # # # Modify the prompt in the existing code
# # # # # # # # # # # # Adding system instructions to guide the model
# # # # # # # # # # # prompt_chatgpt = PromptTemplate(
# # # # # # # # # # #     input_variables=["history", "user_input"],  # Remove system_instructions from here
# # # # # # # # # # #     template=(
# # # # # # # # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # # # # # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # # # # # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # # # # # # # #         "Steps:\n"
# # # # # # # # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # # # # # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # # # # # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # # # # # # # #         "4. Reference the dataset schema if available.\n"
# # # # # # # # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # # # # # # # #         "Conversation history: {history}\n"
# # # # # # # # # # #         "User input: {user_input}\n"
# # # # # # # # # # #         "Assistant:"
# # # # # # # # # # #     ),
# # # # # # # # # # # )
# # # # # # # # # # # from langchain.schema import SystemMessage

# # # # # # # # # # # memory = ConversationBufferMemory()
# # # # # # # # # # # memory.chat_memory.add_message(SystemMessage(content="You are a helpful PACX AI assistant. Follow the steps strictly and assist users with predictive questions."))

# # # # # # # # # # # conversation_chain_chatgpt = ConversationChain(
# # # # # # # # # # #     llm=llm_chatgpt,
# # # # # # # # # # #     prompt=prompt_chatgpt,
# # # # # # # # # # #     input_key="user_input",
# # # # # # # # # # #     memory=ConversationBufferMemory(),  # Add memory instance
# # # # # # # # # # # )

# # # # # # # # # # # # ===========================
# # # # # # # # # # # # Utility Functions
# # # # # # # # # # # # ===========================
# # # # # # # # # # # def get_s3_client():
# # # # # # # # # # #     """
# # # # # # # # # # #     Creates and returns an AWS S3 client.
# # # # # # # # # # #     """
# # # # # # # # # # #     return boto3.client(
# # # # # # # # # # #         's3',
# # # # # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # # # # #     )

# # # # # # # # # # # def get_glue_client():
# # # # # # # # # # #     """
# # # # # # # # # # #     Creates and returns an AWS Glue client.
# # # # # # # # # # #     """
# # # # # # # # # # #     return boto3.client(
# # # # # # # # # # #         'glue',
# # # # # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # # # # #     )

# # # # # # # # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # # # # # # # #     """
# # # # # # # # # # #     Infers the correct data type for a column by handling mixed types.
# # # # # # # # # # #     """
# # # # # # # # # # #     series = series.dropna().astype(str).str.strip()  # Handle mixed types and strip whitespace

# # # # # # # # # # #     # Try boolean
# # # # # # # # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # # # # # # #     unique_values = set(series.str.lower().unique())
# # # # # # # # # # #     if unique_values.issubset(boolean_values):
# # # # # # # # # # #         return "boolean"

# # # # # # # # # # #     # Try integer
# # # # # # # # # # #     try:
# # # # # # # # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # # # # # # # #         if (int_series % 1 == 0).all():
# # # # # # # # # # #             int_min = int_series.min()
# # # # # # # # # # #             int_max = int_series.max()
# # # # # # # # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # # # # # # #                 return "int"
# # # # # # # # # # #             else:
# # # # # # # # # # #                 return "bigint"
# # # # # # # # # # #     except ValueError:
# # # # # # # # # # #         pass

# # # # # # # # # # #     # Try double
# # # # # # # # # # #     try:
# # # # # # # # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # # # # # # # #         return "double"
# # # # # # # # # # #     except ValueError:
# # # # # # # # # # #         pass

# # # # # # # # # # #     # Default to string
# # # # # # # # # # #     return "string"

# # # # # # # # # # # # def suggest_target_column(df: pd.DataFrame) -> Any:
# # # # # # # # # # # #     """
# # # # # # # # # # # #     Suggests a target column based on numeric data types.
# # # # # # # # # # # #     """
# # # # # # # # # # # #     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
# # # # # # # # # # # #     return numeric_cols[0] if len(numeric_cols) > 0 else None

# # # # # # # # # # # def get_user_specified_target(chat_history: List[Any]) -> str:
# # # # # # # # # # #     """
# # # # # # # # # # #     Extracts user-specified target column from chat history if available.
# # # # # # # # # # #     """
# # # # # # # # # # #     for message in reversed(chat_history):  # Traverse messages in reverse order to find the latest target
# # # # # # # # # # #         if isinstance(message, HumanMessage):  # Only check user messages
# # # # # # # # # # #             if "target column" in message.content.lower():
# # # # # # # # # # #                 # Extract the column name (customize regex as needed)
# # # # # # # # # # #                 import re
# # # # # # # # # # #                 match = re.search(r"target column: (\w+)", message.content, re.IGNORECASE)
# # # # # # # # # # #                 if match:
# # # # # # # # # # #                     return match.group(1)
# # # # # # # # # # #     return None



# # # # # # # # # # # from langchain.schema import HumanMessage

# # # # # # # # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # # # # # # # #     """
# # # # # # # # # # #     Suggests a target column based on user input or predictive question.
# # # # # # # # # # #     """
# # # # # # # # # # #     # Check if the user specified a target column
# # # # # # # # # # #     user_target = get_user_specified_target(chat_history)
# # # # # # # # # # #     if user_target and user_target in df.columns:
# # # # # # # # # # #         return user_target

# # # # # # # # # # #     # Use LLM to predict the target column based on a description (if available)
# # # # # # # # # # #     target_suggestion_prompt = (
# # # # # # # # # # #         f"The user uploaded a dataset with the following columns: {', '.join(df.columns)}.\n"
# # # # # # # # # # #         "Based on the context of their predictive question, suggest the best target column."
# # # # # # # # # # #     )
# # # # # # # # # # #     response = llm_chatgpt.invoke([HumanMessage(content=target_suggestion_prompt)])
# # # # # # # # # # #     suggested_column = response.content.strip()
# # # # # # # # # # #     return suggested_column if suggested_column in df.columns else None


# # # # # # # # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # # # # # # # #     """
# # # # # # # # # # # #     Suggests an entity ID column based on uniqueness.
# # # # # # # # # # # #     """
# # # # # # # # # # # #     for col in df.columns:
# # # # # # # # # # # #         if df[col].is_unique:
# # # # # # # # # # # #             return col
# # # # # # # # # # # #     return None

# # # # # # # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # # # # # # #     """
# # # # # # # # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # # # # # # # #     """
# # # # # # # # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # # # # # # # #     for col in likely_id_columns:
# # # # # # # # # # #         if df[col].nunique() / len(df) > 0.95:  # At least 95% unique values
# # # # # # # # # # #             return col

# # # # # # # # # # #     # Fallback: Find any column with >95% unique values
# # # # # # # # # # #     for col in df.columns:
# # # # # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # # # # #             return col
# # # # # # # # # # #     return None




# # # # # # # # # # # # ===========================
# # # # # # # # # # # # Unified ChatGPT API
# # # # # # # # # # # # ===========================
# # # # # # # # # # # class UnifiedChatGPTAPI(APIView):
# # # # # # # # # # #     """
# # # # # # # # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # # # # # # # #     Endpoint: /api/chatgpt/
# # # # # # # # # # #     """
# # # # # # # # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]  # Include JSONParser
# # # # # # # # # # #     uploaded_schema_by_user = {}

    

# # # # # # # # # # #     def post(self, request):
# # # # # # # # # # #         """
# # # # # # # # # # #         Handles POST requests for both chat messages and file uploads.
# # # # # # # # # # #         Differentiates based on the presence of files in the request.
# # # # # # # # # # #         """
# # # # # # # # # # #         action = request.data.get('action', '')
# # # # # # # # # # #         if action == 'reset':
# # # # # # # # # # #             return self.reset_conversation(request)
# # # # # # # # # # #         if "file" in request.FILES:  # If files are present, handle file uploads
# # # # # # # # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # # # # # # # #         # Else, handle chat message
# # # # # # # # # # #         return self.handle_chat(request)
    
# # # # # # # # # # #     # def post(self, request):
        

# # # # # # # # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # # # # # # # #         """
# # # # # # # # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # # # # # # # #         After processing, appends schema details to the chat messages.
# # # # # # # # # # #         """

# # # # # # # # # # #         files = request.FILES.getlist("file")
# # # # # # # # # # #         if not files:
# # # # # # # # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)
        
# # # # # # # # # # #         user_id = request.data.get("user_id", "default_user") 
        
        

# # # # # # # # # # #         try:
# # # # # # # # # # #             uploaded_files_info = []
# # # # # # # # # # #             s3 = get_s3_client()
# # # # # # # # # # #             glue = get_glue_client()

# # # # # # # # # # #             for file in files:
# # # # # # # # # # #                 # Validate file format
# # # # # # # # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # # # # # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)
                
            
# # # # # # # # # # #                 # Read file into Pandas DataFrame
# # # # # # # # # # #                 if file.name.lower().endswith('.csv'):
# # # # # # # # # # #                     df = pd.read_csv(file)
# # # # # # # # # # #                 else:
# # # # # # # # # # #                     df = pd.read_excel(file)

# # # # # # # # # # #                 # Normalize column headers
# # # # # # # # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # # # # # # #                 # print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # # # # # # # #                 # Infer schema
# # # # # # # # # # #                 schema = [
# # # # # # # # # # #                     {
# # # # # # # # # # #                         "column_name": col,
# # # # # # # # # # #                         "data_type": infer_column_dtype(df[col])
# # # # # # # # # # #                     }
# # # # # # # # # # #                     for col in df.columns
# # # # # # # # # # #                 ]
# # # # # # # # # # #                 # print(f"Inferred schema: {schema}")  # Debugging statement

# # # # # # # # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # # # # # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # # # # # # #                 for col in boolean_columns:
# # # # # # # # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # # # # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # # # # # # #                 # print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # # # # # # # #                 # Handle Duplicate Files Dynamically
# # # # # # # # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # # # # # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # # # # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # # # # # # # #                 if existing_file:
# # # # # # # # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # # # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # # # # # # #                     file.name = new_file_name
# # # # # # # # # # #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # # # # # # # #                 else:
# # # # # # # # # # #                     print(f"File name is unique: {file.name}")  # Debugging statement

# # # # # # # # # # #                 # Save Metadata to Database
# # # # # # # # # # #                 file.seek(0)  # Reset file pointer before saving
# # # # # # # # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # # # # # # # #                 if file_serializer.is_valid():
# # # # # # # # # # #                     file_instance = file_serializer.save()

# # # # # # # # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # # # # # # # #                     csv_buffer = BytesIO()
# # # # # # # # # # #                     df.to_csv(csv_buffer, index=False)
# # # # # # # # # # #                     csv_buffer.seek(0)
# # # # # # # # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # # # # # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # # # # # # # #                     # Upload to AWS S3
# # # # # # # # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)

# # # # # # # # # # #                     # Generate file URL
# # # # # # # # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # # # # # # # #                     file_instance.file_url = file_url
# # # # # # # # # # #                     file_instance.save()

# # # # # # # # # # #                     # Save Schema to Database
# # # # # # # # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)

# # # # # # # # # # #                     # Trigger AWS Glue Table Update
# # # # # # # # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # # # # # # #                     # Append file info to response
# # # # # # # # # # #                     uploaded_files_info.append({
# # # # # # # # # # #                         'id': file_instance.id,
# # # # # # # # # # #                         'name': file_instance.name,
# # # # # # # # # # #                         'file_url': file_instance.file_url,
# # # # # # # # # # #                         'schema': schema,
# # # # # # # # # # #                         'suggestions': {  # Add suggestions based on the data
# # # # # # # # # # #                             'target_column': suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages),
# # # # # # # # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # # # # # # # #                         }
# # # # # # # # # # #                     })
                    
# # # # # # # # # # #                 else:
# # # # # # # # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # # #             # Format schema messages to append to assistant conversation
# # # # # # # # # # #             schema_messages = [self.format_schema_message(uploaded_file) for uploaded_file in uploaded_files_info]
# # # # # # # # # # #             combined_schema_message = "\n\n".join(schema_messages)
# # # # # # # # # # #             # print(f"Combined schema message for chat: {combined_schema_message}")  # Debugging statement


# # # # # # # # # # #             # Store schema for user
# # # # # # # # # # #             UnifiedChatGPTAPI.uploaded_schema_by_user[user_id] = combined_schema_message

# # # # # # # # # # #             if hasattr(conversation_chain_chatgpt.memory, "chat_memory"):
# # # # # # # # # # #                 conversation_chain_chatgpt.memory.chat_memory.messages.append(
# # # # # # # # # # #                     HumanMessage(content=f"Schema for '{file.name}': {combined_schema_message}")
# # # # # # # # # # #                 )


# # # # # # # # # # #             return Response({
# # # # # # # # # # #                 "message": "Files uploaded and processed successfully.",
# # # # # # # # # # #                 "uploaded_files": uploaded_files_info,
# # # # # # # # # # #                 "chat_message": combined_schema_message  # Include chat_message in the response
# # # # # # # # # # #             }, status=status.HTTP_201_CREATED)

# # # # # # # # # # #         except pd.errors.EmptyDataError:
# # # # # # # # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # # # # # # #         except NoCredentialsError:
# # # # # # # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # # # # # #         except ClientError as e:
# # # # # # # # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # # # # # #         except Exception as e:
# # # # # # # # # # #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # # # # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
    


# # # # # # # # # # #     def handle_chat(self, request):
# # # # # # # # # # #         user_input = request.data.get("message", "").strip()
# # # # # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # # # # #         if not user_input:
# # # # # # # # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # # #         # Fetch schema for the user session
# # # # # # # # # # #         uploaded_schema = UnifiedChatGPTAPI.uploaded_schema_by_user.get(user_id, "")

# # # # # # # # # # #         # Get or create conversation chain for the user
# # # # # # # # # # #         if user_id not in user_conversations:
# # # # # # # # # # #             conversation_chain = ConversationChain(
# # # # # # # # # # #                 llm=llm_chatgpt,
# # # # # # # # # # #                 prompt=prompt_chatgpt,
# # # # # # # # # # #                 input_key="user_input",
# # # # # # # # # # #                 memory=ConversationBufferMemory()
# # # # # # # # # # #             )
# # # # # # # # # # #             user_conversations[user_id] = conversation_chain
# # # # # # # # # # #         else:
# # # # # # # # # # #             conversation_chain = user_conversations[user_id]

# # # # # # # # # # #         assistant_response = conversation_chain.run(
# # # # # # # # # # #             user_input=f"{user_input}\n\nUploaded Schema:\n{uploaded_schema}"
# # # # # # # # # # #         )

# # # # # # # # # # #         return Response({
# # # # # # # # # # #             "response": assistant_response
# # # # # # # # # # #         })

# # # # # # # # # # #     def reset_conversation(self, request):
# # # # # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # # # # #         # Remove user's conversation chain
# # # # # # # # # # #         if user_id in user_conversations:
# # # # # # # # # # #             del user_conversations[user_id]
# # # # # # # # # # #         # Remove user's uploaded schema
# # # # # # # # # # #         if user_id in UnifiedChatGPTAPI.uploaded_schema_by_user:
# # # # # # # # # # #             del UnifiedChatGPTAPI.uploaded_schema_by_user[user_id]
# # # # # # # # # # #         return Response({"message": "Conversation reset successful."})



    
    


    
# # # # # # # # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # # # # # # # #         """
# # # # # # # # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # # # # # # # #         """
# # # # # # # # # # #         schema = uploaded_file['schema']
# # # # # # # # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # # # # # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # # # # # # # #         schema_text = (
# # # # # # # # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # # # # # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # # # # # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # # # # # # # #             f"Target Column Suggestion: {target_column or 'None provided'}\n"
# # # # # # # # # # #             f"Entity ID Column Suggestion: {entity_id_column or 'None provided'}\n\n"
# # # # # # # # # # #             "Please confirm:\n\n"
# # # # # # # # # # #             "- Is the Target Column correct?\n"
# # # # # # # # # # #             "- Is the Entity ID Column correct?\n"
# # # # # # # # # # #             '(Reply "yes" or provide the correct column names.)'
# # # # # # # # # # #         )
# # # # # # # # # # #         return schema_text


# # # # # # # # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # # # # # # # #         """
# # # # # # # # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # # # # # # # #         """
# # # # # # # # # # #         glue = get_glue_client()
# # # # # # # # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # # # # # # # #         storage_descriptor = {
# # # # # # # # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # # # # # # # #             'Location': s3_location,
# # # # # # # # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # # # # # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # # # # # # # #             'SerdeInfo': {
# # # # # # # # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # # # # # # # #                 'Parameters': {
# # # # # # # # # # #                     'field.delim': ',',
# # # # # # # # # # #                     'skip.header.line.count': '1'
# # # # # # # # # # #                 }
# # # # # # # # # # #             }
# # # # # # # # # # #         }
# # # # # # # # # # #         try:
# # # # # # # # # # #             glue.update_table(
# # # # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # # # #                 TableInput={
# # # # # # # # # # #                     'Name': table_name,
# # # # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # # # #                 }
# # # # # # # # # # #             )
# # # # # # # # # # #             print(f"Glue table '{table_name}' updated successfully.")
# # # # # # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # # # # # #             print(f"Table '{table_name}' not found. Creating a new table...")
# # # # # # # # # # #             glue.create_table(
# # # # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # # # #                 TableInput={
# # # # # # # # # # #                     'Name': table_name,
# # # # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # # # #                 }
# # # # # # # # # # #             )
# # # # # # # # # # #             print(f"Glue table '{table_name}' created successfully.")
# # # # # # # # # # #         except Exception as e:
# # # # # # # # # # #             print(f"Glue operation failed: {str(e)}")





# # # # # # # # # # # chat/views.py

# # # # # # # # # # import os
# # # # # # # # # # import datetime
# # # # # # # # # # from io import BytesIO
# # # # # # # # # # from typing import Any, Dict, List
# # # # # # # # # # import boto3
# # # # # # # # # # import pandas as pd
# # # # # # # # # # import openai
# # # # # # # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # # # # # # from django.conf import settings
# # # # # # # # # # from rest_framework import status
# # # # # # # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # # # # # # from rest_framework.response import Response
# # # # # # # # # # from rest_framework.views import APIView
# # # # # # # # # # from langchain.chains import ConversationChain
# # # # # # # # # # from langchain.chat_models import ChatOpenAI
# # # # # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # # # # from langchain.memory import ConversationBufferMemory
# # # # # # # # # # from langchain.schema import AIMessage, HumanMessage, SystemMessage
# # # # # # # # # # from .models import FileSchema, UploadedFile
# # # # # # # # # # from .serializers import UploadedFileSerializer

# # # # # # # # # # # ===========================
# # # # # # # # # # # AWS Configuration
# # # # # # # # # # # ===========================
# # # # # # # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # # # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # # # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')

# # # # # # # # # # # ===========================
# # # # # # # # # # # OpenAI Configuration
# # # # # # # # # # # ===========================
# # # # # # # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # # # # # # openai.api_key = OPENAI_API_KEY

# # # # # # # # # # # ===========================
# # # # # # # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # # # # # # ===========================
# # # # # # # # # # llm_chatgpt = ChatOpenAI(
# # # # # # # # # #     model="gpt-3.5-turbo-16k",
# # # # # # # # # #     temperature=0.7,
# # # # # # # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # # # # # # )

# # # # # # # # # # # Global dictionaries to store user-specific data
# # # # # # # # # # user_conversations = {}
# # # # # # # # # # user_schemas = {}
# # # # # # # # # # user_confirmations = {}
# # # # # # # # # # user_notebook_flags = {}

# # # # # # # # # # # Modify the prompt in the existing code
# # # # # # # # # # prompt_chatgpt = PromptTemplate(
# # # # # # # # # #     input_variables=["history", "user_input"],
# # # # # # # # # #     template=(
# # # # # # # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # # # # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # # # # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # # # # # # #         "Steps:\n"
# # # # # # # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # # # # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # # # # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # # # # # # #         "4. Reference the dataset schema if available.\n"
# # # # # # # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # # # # # # #         "Conversation history: {history}\n"
# # # # # # # # # #         "User input: {user_input}\n"
# # # # # # # # # #         "Assistant:"
# # # # # # # # # #     ),
# # # # # # # # # # )

# # # # # # # # # # memory = ConversationBufferMemory()
# # # # # # # # # # memory.chat_memory.add_message(SystemMessage(content="You are a helpful PACX AI assistant. Follow the steps strictly and assist users with predictive questions."))

# # # # # # # # # # conversation_chain_chatgpt = ConversationChain(
# # # # # # # # # #     llm=llm_chatgpt,
# # # # # # # # # #     prompt=prompt_chatgpt,
# # # # # # # # # #     input_key="user_input",
# # # # # # # # # #     memory=ConversationBufferMemory(),
# # # # # # # # # # )

# # # # # # # # # # # ===========================
# # # # # # # # # # # Utility Functions
# # # # # # # # # # # ===========================

# # # # # # # # # # def get_s3_client():
# # # # # # # # # #     """
# # # # # # # # # #     Creates and returns an AWS S3 client.
# # # # # # # # # #     """
# # # # # # # # # #     return boto3.client(
# # # # # # # # # #         's3',
# # # # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # # # #     )

# # # # # # # # # # def get_glue_client():
# # # # # # # # # #     """
# # # # # # # # # #     Creates and returns an AWS Glue client.
# # # # # # # # # #     """
# # # # # # # # # #     return boto3.client(
# # # # # # # # # #         'glue',
# # # # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # # # #     )

# # # # # # # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # # # # # # #     """
# # # # # # # # # #     Infers the correct data type for a column by handling mixed types.
# # # # # # # # # #     """
# # # # # # # # # #     series = series.dropna().astype(str).str.strip()

# # # # # # # # # #     # Try datetime
# # # # # # # # # #     try:
# # # # # # # # # #         pd.to_datetime(series, errors='raise', infer_datetime_format=True)
# # # # # # # # # #         return "timestamp"
# # # # # # # # # #     except ValueError:
# # # # # # # # # #         pass

# # # # # # # # # #     # Try boolean
# # # # # # # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # # # # # #     unique_values = set(series.str.lower().unique())
# # # # # # # # # #     if unique_values.issubset(boolean_values):
# # # # # # # # # #         return "boolean"

# # # # # # # # # #     # Try integer
# # # # # # # # # #     try:
# # # # # # # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # # # # # # #         if (int_series % 1 == 0).all():
# # # # # # # # # #             int_min = int_series.min()
# # # # # # # # # #             int_max = int_series.max()
# # # # # # # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # # # # # #                 return "int"
# # # # # # # # # #             else:
# # # # # # # # # #                 return "bigint"
# # # # # # # # # #     except ValueError:
# # # # # # # # # #         pass

# # # # # # # # # #     # Try double
# # # # # # # # # #     try:
# # # # # # # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # # # # # # #         return "double"
# # # # # # # # # #     except ValueError:
# # # # # # # # # #         pass

# # # # # # # # # #     # Default to string
# # # # # # # # # #     return "string"

# # # # # # # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # # # # # # #     """
# # # # # # # # # #     Suggests a target column based on user input or predictive question.
# # # # # # # # # #     """
# # # # # # # # # #     # Use the last column as a default suggestion
# # # # # # # # # #     return df.columns[-1]

# # # # # # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # # # # # #     """
# # # # # # # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # # # # # # #     """
# # # # # # # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # # # # # # #     for col in likely_id_columns:
# # # # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # # # #             return col

# # # # # # # # # #     # Fallback: Find any column with >95% unique values
# # # # # # # # # #     for col in df.columns:
# # # # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # # # #             return col
# # # # # # # # # #     return None

# # # # # # # # # # # ===========================
# # # # # # # # # # # Unified ChatGPT API
# # # # # # # # # # # ===========================
# # # # # # # # # # class UnifiedChatGPTAPI(APIView):
# # # # # # # # # #     """
# # # # # # # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # # # # # # #     Endpoint: /api/chatgpt/
# # # # # # # # # #     """
# # # # # # # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # # # # # # #     def post(self, request):
# # # # # # # # # #         """
# # # # # # # # # #         Handles POST requests for chat messages and file uploads.
# # # # # # # # # #         Differentiates based on the presence of files in the request.
# # # # # # # # # #         """
# # # # # # # # # #         action = request.data.get('action', '')
# # # # # # # # # #         if action == 'reset':
# # # # # # # # # #             return self.reset_conversation(request)
# # # # # # # # # #         if "file" in request.FILES:
# # # # # # # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # # # # # # #         # Handle 'Generate Notebook' action
# # # # # # # # # #         if action == 'generate_notebook':
# # # # # # # # # #             return self.generate_notebook(request)

# # # # # # # # # #         # Else, handle chat message
# # # # # # # # # #         return self.handle_chat(request)

# # # # # # # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # # # # # # #         """
# # # # # # # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # # # # # # #         After processing, appends schema details to the chat messages.
# # # # # # # # # #         """
# # # # # # # # # #         files = request.FILES.getlist("file")
# # # # # # # # # #         if not files:
# # # # # # # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # # # #         try:
# # # # # # # # # #             uploaded_files_info = []
# # # # # # # # # #             s3 = get_s3_client()
# # # # # # # # # #             glue = get_glue_client()

# # # # # # # # # #             for file in files:
# # # # # # # # # #                 # Validate file format
# # # # # # # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # # # # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #                 # Read file into Pandas DataFrame
# # # # # # # # # #                 if file.name.lower().endswith('.csv'):
# # # # # # # # # #                     df = pd.read_csv(file)
# # # # # # # # # #                 else:
# # # # # # # # # #                     df = pd.read_excel(file)

# # # # # # # # # #                 # Normalize column headers
# # # # # # # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # # # # # #                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # # # # # # #                 # Infer schema with precision
# # # # # # # # # #                 schema = [
# # # # # # # # # #                     {
# # # # # # # # # #                         "column_name": col,
# # # # # # # # # #                         "data_type": infer_column_dtype(df[col])
# # # # # # # # # #                     }
# # # # # # # # # #                     for col in df.columns
# # # # # # # # # #                 ]
# # # # # # # # # #                 print(f"Inferred schema: {schema}")  # Debugging statement

# # # # # # # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # # # # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # # # # # #                 for col in boolean_columns:
# # # # # # # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # # # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # # # # # #                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # # # # # # #                 # Handle Duplicate Files Dynamically
# # # # # # # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # # # # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # # # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # # # # # # #                 if existing_file:
# # # # # # # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # # # # # #                     file.name = new_file_name
# # # # # # # # # #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # # # # # # #                 else:
# # # # # # # # # #                     print(f"File name is unique: {file.name}")  # Debugging statement

# # # # # # # # # #                 # Save Metadata to Database
# # # # # # # # # #                 file.seek(0)
# # # # # # # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # # # # # # #                 if file_serializer.is_valid():
# # # # # # # # # #                     file_instance = file_serializer.save()

# # # # # # # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # # # # # # #                     csv_buffer = BytesIO()
# # # # # # # # # #                     df.to_csv(csv_buffer, index=False)
# # # # # # # # # #                     csv_buffer.seek(0)
# # # # # # # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # # # # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # # # # # # #                     # Upload to AWS S3
# # # # # # # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # # # # # # # #                     print(f"File uploaded to S3: {file_key}")  # Debugging statement

# # # # # # # # # #                     # Generate file URL
# # # # # # # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # # # # # # #                     file_instance.file_url = file_url
# # # # # # # # # #                     file_instance.save()

# # # # # # # # # #                     # Save Schema to Database
# # # # # # # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # # # # # # # #                     print(f"Schema saved to database for file: {file.name}")  # Debugging statement

# # # # # # # # # #                     # Trigger AWS Glue Table Update
# # # # # # # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # # # # # #                     # Append file info to response
# # # # # # # # # #                     uploaded_files_info.append({
# # # # # # # # # #                         'id': file_instance.id,
# # # # # # # # # #                         'name': file_instance.name,
# # # # # # # # # #                         'file_url': file_instance.file_url,
# # # # # # # # # #                         'schema': schema,
# # # # # # # # # #                         'suggestions': {
# # # # # # # # # #                             'target_column': suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages),
# # # # # # # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # # # # # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages)]]
# # # # # # # # # #                         }
# # # # # # # # # #                     })

# # # # # # # # # #                 else:
# # # # # # # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #             # Store schema for user
# # # # # # # # # #             user_schemas[user_id] = uploaded_files_info

# # # # # # # # # #             # Initiate schema discussion with the user
# # # # # # # # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # # # # # # # #             if hasattr(conversation_chain_chatgpt.memory, "chat_memory"):
# # # # # # # # # #                 conversation_chain_chatgpt.memory.chat_memory.messages.append(
# # # # # # # # # #                     AIMessage(content=schema_discussion)
# # # # # # # # # #                 )
# # # # # # # # # #             print(f"Schema discussion initiated: {schema_discussion}")  # Debugging statement

# # # # # # # # # #             return Response({
# # # # # # # # # #                 "message": "Files uploaded and processed successfully.",
# # # # # # # # # #                 "uploaded_files": uploaded_files_info,
# # # # # # # # # #                 "chat_message": schema_discussion
# # # # # # # # # #             }, status=status.HTTP_201_CREATED)

# # # # # # # # # #         except pd.errors.EmptyDataError:
# # # # # # # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # # # # # #         except NoCredentialsError:
# # # # # # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # # # # #         except ClientError as e:
# # # # # # # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # # # # #         except Exception as e:
# # # # # # # # # #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # # # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #     def handle_chat(self, request):
# # # # # # # # # #         user_input = request.data.get("message", "").strip()
# # # # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # # # #         if not user_input:
# # # # # # # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         # Get or create conversation chain for the user
# # # # # # # # # #         if user_id not in user_conversations:
# # # # # # # # # #             conversation_chain = ConversationChain(
# # # # # # # # # #                 llm=llm_chatgpt,
# # # # # # # # # #                 prompt=prompt_chatgpt,
# # # # # # # # # #                 input_key="user_input",
# # # # # # # # # #                 memory=ConversationBufferMemory()
# # # # # # # # # #             )
# # # # # # # # # #             user_conversations[user_id] = conversation_chain
# # # # # # # # # #         else:
# # # # # # # # # #             conversation_chain = user_conversations[user_id]

# # # # # # # # # #         # Check if user is confirming schema
# # # # # # # # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # # # # # # # #             # Process user confirmation
# # # # # # # # # #             confirmation_response = self.process_schema_confirmation(user_input, user_id)
# # # # # # # # # #             if confirmation_response:
# # # # # # # # # #                 return Response({"response": confirmation_response})

# # # # # # # # # #         # Generate assistant response
# # # # # # # # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # # # # # # # #         print(f"Assistant response: {assistant_response}")  # Debugging statement

# # # # # # # # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # # # # # # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # # # # # # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # # # # # # # #             user_notebook_flags[user_id] = True  # Flag to show 'Generate Notebook' button
# # # # # # # # # #             print("GENERATE_NOTEBOOK_PROMPT detected. Flagging to show 'Generate Notebook' button.")  # Debugging statement

# # # # # # # # # #         return Response({
# # # # # # # # # #             "response": assistant_response,
# # # # # # # # # #             "show_generate_notebook": user_notebook_flags.get(user_id, False)
# # # # # # # # # #         })

# # # # # # # # # #     def process_schema_confirmation(self, user_input, user_id):
# # # # # # # # # #         """
# # # # # # # # # #         Processes user confirmation or adjustment of the schema.
# # # # # # # # # #         """
# # # # # # # # # #         uploaded_file_info = user_schemas[user_id][0]
# # # # # # # # # #         suggestions = uploaded_file_info['suggestions']

# # # # # # # # # #         # Assume user confirms or provides adjustments
# # # # # # # # # #         if 'yes' in user_input.lower():
# # # # # # # # # #             user_confirmations[user_id] = suggestions
# # # # # # # # # #             return "Schema confirmed. You can now click 'Generate Notebook' to proceed."
# # # # # # # # # #         else:
# # # # # # # # # #             # Parse user adjustments
# # # # # # # # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # # # # # # # #             if adjusted_columns:
# # # # # # # # # #                 user_confirmations[user_id] = adjusted_columns
# # # # # # # # # #                 return "Schema updated based on your inputs. You can now click 'Generate Notebook' to proceed."
# # # # # # # # # #             else:
# # # # # # # # # #                 return "Could not understand your adjustments. Please specify the correct 'Entity ID' and 'Target' column names."

# # # # # # # # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # # # # # # # #         """
# # # # # # # # # #         Parses user input for schema adjustments.
# # # # # # # # # #         """
# # # # # # # # # #         import re
# # # # # # # # # #         entity_id_match = re.search(r"Entity ID Column: (\w+)", user_input, re.IGNORECASE)
# # # # # # # # # #         target_column_match = re.search(r"Target Column: (\w+)", user_input, re.IGNORECASE)

# # # # # # # # # #         suggestions = uploaded_file_info['suggestions']
# # # # # # # # # #         entity_id_column = suggestions['entity_id_column']
# # # # # # # # # #         target_column = suggestions['target_column']

# # # # # # # # # #         if entity_id_match:
# # # # # # # # # #             entity_id_column = entity_id_match.group(1)
# # # # # # # # # #         if target_column_match:
# # # # # # # # # #             target_column = target_column_match.group(1)

# # # # # # # # # #         if entity_id_column and target_column:
# # # # # # # # # #             return {
# # # # # # # # # #                 'entity_id_column': entity_id_column,
# # # # # # # # # #                 'target_column': target_column,
# # # # # # # # # #                 'feature_columns': [col for col in uploaded_file_info['schema'] if col['column_name'] not in [entity_id_column, target_column]]
# # # # # # # # # #             }
# # # # # # # # # #         else:
# # # # # # # # # #             return None

# # # # # # # # # #     def reset_conversation(self, request):
# # # # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # # # #         # Remove user's conversation chain
# # # # # # # # # #         if user_id in user_conversations:
# # # # # # # # # #             del user_conversations[user_id]
# # # # # # # # # #         # Remove user's uploaded schema and confirmations
# # # # # # # # # #         if user_id in user_schemas:
# # # # # # # # # #             del user_schemas[user_id]
# # # # # # # # # #         if user_id in user_confirmations:
# # # # # # # # # #             del user_confirmations[user_id]
# # # # # # # # # #         if user_id in user_notebook_flags:
# # # # # # # # # #             del user_notebook_flags[user_id]
# # # # # # # # # #         print(f"Conversation reset for user: {user_id}")  # Debugging statement
# # # # # # # # # #         return Response({"message": "Conversation reset successful."})

# # # # # # # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # # # # # # #         """
# # # # # # # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # # # # # # #         """
# # # # # # # # # #         schema = uploaded_file['schema']
# # # # # # # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # # # # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # # # # # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # # # # # # # #         schema_text = (
# # # # # # # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # # # # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # # # # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # # # # # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # # # # # # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # # # # # # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # # # # # #             "Please confirm:\n"
# # # # # # # # # #             "- Is the Target Column correct?\n"
# # # # # # # # # #             "- Is the Entity ID Column correct?\n"
# # # # # # # # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # # # # # # # #         )
# # # # # # # # # #         return schema_text

# # # # # # # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # # # # # # #         """
# # # # # # # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # # # # # # #         """
# # # # # # # # # #         glue = get_glue_client()
# # # # # # # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
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
# # # # # # # # # #             print(f"Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # # # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # # # # #             print(f"Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # # # # # # # #             glue.create_table(
# # # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # # #                 TableInput={
# # # # # # # # # #                     'Name': table_name,
# # # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # # #                 }
# # # # # # # # # #             )
# # # # # # # # # #             print(f"Glue table '{table_name}' created successfully.")  # Debugging statement
# # # # # # # # # #         except Exception as e:
# # # # # # # # # #             print(f"Glue operation failed: {str(e)}")  # Debugging statement

# # # # # # # # # #     def generate_notebook(self, request):
# # # # # # # # # #         """
# # # # # # # # # #         Generates a Jupyter Notebook with pre-filled SQL queries based on the confirmed schema.
# # # # # # # # # #         """
# # # # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # # # #         if user_id not in user_confirmations:
# # # # # # # # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # # #         confirmation = user_confirmations[user_id]
# # # # # # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # # # # # #         target_column = confirmation['target_column']
# # # # # # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # # # # # #         # Create Jupyter Notebook
# # # # # # # # # #         notebook = self.create_notebook(entity_id_column, target_column, feature_columns)
# # # # # # # # # #         print("Notebook generated successfully.")  # Debugging statement

# # # # # # # # # #         # Return the notebook content
# # # # # # # # # #         return Response({
# # # # # # # # # #             "message": "Notebook generated successfully.",
# # # # # # # # # #             "notebook": notebook
# # # # # # # # # #         }, status=status.HTTP_200_OK)

# # # # # # # # # #     def create_notebook(self, entity_id_column, target_column, feature_columns):
# # # # # # # # # #         """
# # # # # # # # # #         Creates a Jupyter Notebook with SQL queries.
# # # # # # # # # #         """
# # # # # # # # # #         import nbformat
# # # # # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # # # # #         nb = new_notebook()
# # # # # # # # # #         cells = []

# # # # # # # # # #         # Introduction cell
# # # # # # # # # #         cells.append(new_markdown_cell("# Exploratory Data Analysis Notebook"))

# # # # # # # # # #         # Entity ID and Target analysis
# # # # # # # # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM your_table_name LIMIT 100;"
# # # # # # # # # #         cells.append(new_markdown_cell("## Entity ID and Target Column Analysis"))
# # # # # # # # # #         cells.append(new_code_cell(f"%%sql\n{sql_query_entity_target}"))

# # # # # # # # # #         # Feature columns analysis
# # # # # # # # # #         for feature in feature_columns:
# # # # # # # # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) FROM your_table_name GROUP BY {feature} LIMIT 100;"
# # # # # # # # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))
# # # # # # # # # #             cells.append(new_code_cell(f"%%sql\n{sql_query_feature}"))

# # # # # # # # # #         nb['cells'] = cells

# # # # # # # # # #         # Convert notebook to JSON
# # # # # # # # # #         notebook_json = nbformat.writes(nb)
# # # # # # # # # #         return notebook_json



# # # # # # # # # # chat/views.py

# # # # # # # # # import os
# # # # # # # # # import datetime
# # # # # # # # # from io import BytesIO
# # # # # # # # # from typing import Any, Dict, List
# # # # # # # # # import boto3
# # # # # # # # # import pandas as pd
# # # # # # # # # import openai
# # # # # # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # # # # # from django.conf import settings
# # # # # # # # # from rest_framework import status
# # # # # # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # # # # # from rest_framework.response import Response
# # # # # # # # # from rest_framework.views import APIView
# # # # # # # # # from langchain.chains import ConversationChain
# # # # # # # # # from langchain.chat_models import ChatOpenAI
# # # # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # # # from langchain.memory import ConversationBufferMemory
# # # # # # # # # from langchain.schema import AIMessage, HumanMessage, SystemMessage
# # # # # # # # # from .models import FileSchema, UploadedFile
# # # # # # # # # from .serializers import UploadedFileSerializer

# # # # # # # # # # ===========================
# # # # # # # # # # AWS Configuration
# # # # # # # # # # ===========================
# # # # # # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # # # # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # e.g., 's3://your-athena-query-results-bucket/'
# # # # # # # # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # # # # # # # ===========================
# # # # # # # # # # OpenAI Configuration
# # # # # # # # # # ===========================
# # # # # # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # # # # # openai.api_key = OPENAI_API_KEY

# # # # # # # # # # ===========================
# # # # # # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # # # # # ===========================
# # # # # # # # # llm_chatgpt = ChatOpenAI(
# # # # # # # # #     model="gpt-3.5-turbo-16k",
# # # # # # # # #     temperature=0.7,
# # # # # # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # # # # # )

# # # # # # # # # # Global dictionaries to store user-specific data
# # # # # # # # # user_conversations = {}
# # # # # # # # # user_schemas = {}
# # # # # # # # # user_confirmations = {}
# # # # # # # # # user_notebook_flags = {}
# # # # # # # # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # # # # # # # Modify the prompt in the existing code
# # # # # # # # # prompt_chatgpt = PromptTemplate(
# # # # # # # # #     input_variables=["history", "user_input"],
# # # # # # # # #     template=(
# # # # # # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # # # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # # # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # # # # # #         "Steps:\n"
# # # # # # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # # # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # # # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # # # # # #         "4. Reference the dataset schema if available.\n"
# # # # # # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # # # # # #         "Conversation history: {history}\n"
# # # # # # # # #         "User input: {user_input}\n"
# # # # # # # # #         "Assistant:"
# # # # # # # # #     ),
# # # # # # # # # )

# # # # # # # # # memory = ConversationBufferMemory()
# # # # # # # # # memory.chat_memory.add_message(SystemMessage(content="You are a helpful PACX AI assistant. Follow the steps strictly and assist users with predictive questions."))

# # # # # # # # # conversation_chain_chatgpt = ConversationChain(
# # # # # # # # #     llm=llm_chatgpt,
# # # # # # # # #     prompt=prompt_chatgpt,
# # # # # # # # #     input_key="user_input",
# # # # # # # # #     memory=ConversationBufferMemory(),
# # # # # # # # # )

# # # # # # # # # # ===========================
# # # # # # # # # # Utility Functions
# # # # # # # # # # ===========================

# # # # # # # # # def get_s3_client():
# # # # # # # # #     """
# # # # # # # # #     Creates and returns an AWS S3 client.
# # # # # # # # #     """
# # # # # # # # #     return boto3.client(
# # # # # # # # #         's3',
# # # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # # #     )

# # # # # # # # # def get_glue_client():
# # # # # # # # #     """
# # # # # # # # #     Creates and returns an AWS Glue client.
# # # # # # # # #     """
# # # # # # # # #     return boto3.client(
# # # # # # # # #         'glue',
# # # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # # #     )

# # # # # # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # # # # # #     """
# # # # # # # # #     Infers the correct data type for a column by handling mixed types.
# # # # # # # # #     """
# # # # # # # # #     series = series.dropna().astype(str).str.strip()

# # # # # # # # #     # Try datetime
# # # # # # # # #     try:
# # # # # # # # #         pd.to_datetime(series, errors='raise', infer_datetime_format=True)
# # # # # # # # #         return "timestamp"
# # # # # # # # #     except ValueError:
# # # # # # # # #         pass

# # # # # # # # #     # Try boolean
# # # # # # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # # # # #     unique_values = set(series.str.lower().unique())
# # # # # # # # #     if unique_values.issubset(boolean_values):
# # # # # # # # #         return "boolean"

# # # # # # # # #     # Try integer
# # # # # # # # #     try:
# # # # # # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # # # # # #         if (int_series % 1 == 0).all():
# # # # # # # # #             int_min = int_series.min()
# # # # # # # # #             int_max = int_series.max()
# # # # # # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # # # # #                 return "int"
# # # # # # # # #             else:
# # # # # # # # #                 return "bigint"
# # # # # # # # #     except ValueError:
# # # # # # # # #         pass

# # # # # # # # #     # Try double
# # # # # # # # #     try:
# # # # # # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # # # # # #         return "double"
# # # # # # # # #     except ValueError:
# # # # # # # # #         pass

# # # # # # # # #     # Default to string
# # # # # # # # #     return "string"

# # # # # # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # # # # # #     """
# # # # # # # # #     Suggests a target column based on user input or predictive question.
# # # # # # # # #     """
# # # # # # # # #     # Use the last column as a default suggestion
# # # # # # # # #     return df.columns[-1]

# # # # # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # # # # #     """
# # # # # # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # # # # # #     """
# # # # # # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # # # # # #     for col in likely_id_columns:
# # # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # # #             return col

# # # # # # # # #     # Fallback: Find any column with >95% unique values
# # # # # # # # #     for col in df.columns:
# # # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # # #             return col
# # # # # # # # #     return None

# # # # # # # # # # ===========================
# # # # # # # # # # Unified ChatGPT API
# # # # # # # # # # ===========================
# # # # # # # # # class UnifiedChatGPTAPI(APIView):
# # # # # # # # #     """
# # # # # # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # # # # # #     Endpoint: /api/chatgpt/
# # # # # # # # #     """
# # # # # # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # # # # # #     def post(self, request):
# # # # # # # # #         """
# # # # # # # # #         Handles POST requests for chat messages and file uploads.
# # # # # # # # #         Differentiates based on the presence of files in the request.
# # # # # # # # #         """
# # # # # # # # #         action = request.data.get('action', '')
# # # # # # # # #         if action == 'reset':
# # # # # # # # #             return self.reset_conversation(request)
# # # # # # # # #         if "file" in request.FILES:
# # # # # # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # # # # # #         # Handle 'Generate Notebook' action
# # # # # # # # #         if action == 'generate_notebook':
# # # # # # # # #             return self.generate_notebook(request)

# # # # # # # # #         # Else, handle chat message
# # # # # # # # #         return self.handle_chat(request)

# # # # # # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # # # # # #         """
# # # # # # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # # # # # #         After processing, appends schema details to the chat messages.
# # # # # # # # #         """
# # # # # # # # #         files = request.FILES.getlist("file")
# # # # # # # # #         if not files:
# # # # # # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # # #         try:
# # # # # # # # #             uploaded_files_info = []
# # # # # # # # #             s3 = get_s3_client()
# # # # # # # # #             glue = get_glue_client()

# # # # # # # # #             for file in files:
# # # # # # # # #                 # Validate file format
# # # # # # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # # # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #                 # Read file into Pandas DataFrame
# # # # # # # # #                 if file.name.lower().endswith('.csv'):
# # # # # # # # #                     df = pd.read_csv(file)
# # # # # # # # #                 else:
# # # # # # # # #                     df = pd.read_excel(file)

# # # # # # # # #                 # Normalize column headers
# # # # # # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # # # # #                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # # # # # #                 # Infer schema with precision
# # # # # # # # #                 schema = [
# # # # # # # # #                     {
# # # # # # # # #                         "column_name": col,
# # # # # # # # #                         "data_type": infer_column_dtype(df[col])
# # # # # # # # #                     }
# # # # # # # # #                     for col in df.columns
# # # # # # # # #                 ]
# # # # # # # # #                 print(f"Inferred schema: {schema}")  # Debugging statement

# # # # # # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # # # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # # # # #                 for col in boolean_columns:
# # # # # # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # # # # #                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # # # # # #                 # Handle Duplicate Files Dynamically
# # # # # # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # # # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # # # # # #                 if existing_file:
# # # # # # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # # # # #                     file.name = new_file_name
# # # # # # # # #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # # # # # #                 else:
# # # # # # # # #                     print(f"File name is unique: {file.name}")  # Debugging statement

# # # # # # # # #                 # Save Metadata to Database
# # # # # # # # #                 file.seek(0)
# # # # # # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # # # # # #                 if file_serializer.is_valid():
# # # # # # # # #                     file_instance = file_serializer.save()

# # # # # # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # # # # # #                     csv_buffer = BytesIO()
# # # # # # # # #                     df.to_csv(csv_buffer, index=False)
# # # # # # # # #                     csv_buffer.seek(0)
# # # # # # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # # # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # # # # # #                     # Upload to AWS S3
# # # # # # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # # # # # # #                     print(f"File uploaded to S3: {file_key}")  # Debugging statement

# # # # # # # # #                     # Generate file URL
# # # # # # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # # # # # #                     file_instance.file_url = file_url
# # # # # # # # #                     file_instance.save()

# # # # # # # # #                     # Save Schema to Database
# # # # # # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # # # # # # #                     print(f"Schema saved to database for file: {file.name}")  # Debugging statement

# # # # # # # # #                     # Trigger AWS Glue Table Update
# # # # # # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # # # # #                     # Append file info to response
# # # # # # # # #                     uploaded_files_info.append({
# # # # # # # # #                         'id': file_instance.id,
# # # # # # # # #                         'name': file_instance.name,
# # # # # # # # #                         'file_url': file_instance.file_url,
# # # # # # # # #                         'schema': schema,
# # # # # # # # #                         'suggestions': {
# # # # # # # # #                             'target_column': suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages),
# # # # # # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # # # # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages)]]
# # # # # # # # #                         }
# # # # # # # # #                     })

# # # # # # # # #                 else:
# # # # # # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #             # Store schema for user
# # # # # # # # #             user_schemas[user_id] = uploaded_files_info

# # # # # # # # #             # Initiate schema discussion with the user
# # # # # # # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # # # # # # #             if hasattr(conversation_chain_chatgpt.memory, "chat_memory"):
# # # # # # # # #                 conversation_chain_chatgpt.memory.chat_memory.messages.append(
# # # # # # # # #                     AIMessage(content=schema_discussion)
# # # # # # # # #                 )
# # # # # # # # #             print(f"Schema discussion initiated: {schema_discussion}")  # Debugging statement

# # # # # # # # #             return Response({
# # # # # # # # #                 "message": "Files uploaded and processed successfully.",
# # # # # # # # #                 "uploaded_files": uploaded_files_info,
# # # # # # # # #                 "chat_message": schema_discussion
# # # # # # # # #             }, status=status.HTTP_201_CREATED)

# # # # # # # # #         except pd.errors.EmptyDataError:
# # # # # # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # # # # #         except NoCredentialsError:
# # # # # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # # # #         except ClientError as e:
# # # # # # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #     def handle_chat(self, request):
# # # # # # # # #         user_input = request.data.get("message", "").strip()
# # # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # # #         if not user_input:
# # # # # # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #         # Get or create conversation chain for the user
# # # # # # # # #         if user_id not in user_conversations:
# # # # # # # # #             conversation_chain = ConversationChain(
# # # # # # # # #                 llm=llm_chatgpt,
# # # # # # # # #                 prompt=prompt_chatgpt,
# # # # # # # # #                 input_key="user_input",
# # # # # # # # #                 memory=ConversationBufferMemory()
# # # # # # # # #             )
# # # # # # # # #             user_conversations[user_id] = conversation_chain
# # # # # # # # #         else:
# # # # # # # # #             conversation_chain = user_conversations[user_id]

# # # # # # # # #         # Check if user is confirming schema
# # # # # # # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # # # # # # #             # Process user confirmation
# # # # # # # # #             confirmation_response = self.process_schema_confirmation(user_input, user_id)
# # # # # # # # #             if confirmation_response:
# # # # # # # # #                 return Response({"response": confirmation_response})

# # # # # # # # #         # Generate assistant response
# # # # # # # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # # # # # # #         print(f"Assistant response: {assistant_response}")  # Debugging statement

# # # # # # # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # # # # # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # # # # # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # # # # # # #             user_notebook_flags[user_id] = True  # Flag to show 'Generate Notebook' button
# # # # # # # # #             print("GENERATE_NOTEBOOK_PROMPT detected. Flagging to show 'Generate Notebook' button.")  # Debugging statement

# # # # # # # # #         return Response({
# # # # # # # # #             "response": assistant_response,
# # # # # # # # #             "show_generate_notebook": user_notebook_flags.get(user_id, False)
# # # # # # # # #         })

# # # # # # # # #     def process_schema_confirmation(self, user_input, user_id):
# # # # # # # # #         """
# # # # # # # # #         Processes user confirmation or adjustment of the schema.
# # # # # # # # #         """
# # # # # # # # #         uploaded_file_info = user_schemas[user_id][0]
# # # # # # # # #         suggestions = uploaded_file_info['suggestions']

# # # # # # # # #         # Assume user confirms or provides adjustments
# # # # # # # # #         if 'yes' in user_input.lower():
# # # # # # # # #             user_confirmations[user_id] = suggestions
# # # # # # # # #             return "Schema confirmed. You can now click 'Generate Notebook' to proceed."
# # # # # # # # #         else:
# # # # # # # # #             # Parse user adjustments
# # # # # # # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # # # # # # #             if adjusted_columns:
# # # # # # # # #                 user_confirmations[user_id] = adjusted_columns
# # # # # # # # #                 return "Schema updated based on your inputs. You can now click 'Generate Notebook' to proceed."
# # # # # # # # #             else:
# # # # # # # # #                 return "Could not understand your adjustments. Please specify the correct 'Entity ID' and 'Target' column names."

# # # # # # # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # # # # # # #         """
# # # # # # # # #         Parses user input for schema adjustments.
# # # # # # # # #         """
# # # # # # # # #         import re
# # # # # # # # #         entity_id_match = re.search(r"Entity ID Column: (\w+)", user_input, re.IGNORECASE)
# # # # # # # # #         target_column_match = re.search(r"Target Column: (\w+)", user_input, re.IGNORECASE)

# # # # # # # # #         suggestions = uploaded_file_info['suggestions']
# # # # # # # # #         entity_id_column = suggestions['entity_id_column']
# # # # # # # # #         target_column = suggestions['target_column']

# # # # # # # # #         if entity_id_match:
# # # # # # # # #             entity_id_column = entity_id_match.group(1)
# # # # # # # # #         if target_column_match:
# # # # # # # # #             target_column = target_column_match.group(1)

# # # # # # # # #         if entity_id_column and target_column:
# # # # # # # # #             return {
# # # # # # # # #                 'entity_id_column': entity_id_column,
# # # # # # # # #                 'target_column': target_column,
# # # # # # # # #                 'feature_columns': [col for col in uploaded_file_info['schema'] if col['column_name'] not in [entity_id_column, target_column]]
# # # # # # # # #             }
# # # # # # # # #         else:
# # # # # # # # #             return None

# # # # # # # # #     def reset_conversation(self, request):
# # # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # # #         # Remove user's conversation chain
# # # # # # # # #         if user_id in user_conversations:
# # # # # # # # #             del user_conversations[user_id]
# # # # # # # # #         # Remove user's uploaded schema and confirmations
# # # # # # # # #         if user_id in user_schemas:
# # # # # # # # #             del user_schemas[user_id]
# # # # # # # # #         if user_id in user_confirmations:
# # # # # # # # #             del user_confirmations[user_id]
# # # # # # # # #         if user_id in user_notebook_flags:
# # # # # # # # #             del user_notebook_flags[user_id]
# # # # # # # # #         if user_id in user_notebooks:
# # # # # # # # #             del user_notebooks[user_id]
# # # # # # # # #         print(f"Conversation reset for user: {user_id}")  # Debugging statement
# # # # # # # # #         return Response({"message": "Conversation reset successful."})

# # # # # # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # # # # # #         """
# # # # # # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # # # # # #         """
# # # # # # # # #         schema = uploaded_file['schema']
# # # # # # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # # # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # # # # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # # # # # # #         schema_text = (
# # # # # # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # # # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # # # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # # # # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # # # # # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # # # # # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # # # # #             "Please confirm:\n"
# # # # # # # # #             "- Is the Target Column correct?\n"
# # # # # # # # #             "- Is the Entity ID Column correct?\n"
# # # # # # # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # # # # # # #         )
# # # # # # # # #         return schema_text

# # # # # # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # # # # # #         """
# # # # # # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # # # # # #         """
# # # # # # # # #         glue = get_glue_client()
# # # # # # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # # # # # #         storage_descriptor = {
# # # # # # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # # # # # #             'Location': s3_location,
# # # # # # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # # # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # # # # # #             'SerdeInfo': {
# # # # # # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # # # # # #                 'Parameters': {
# # # # # # # # #                     'field.delim': ',',
# # # # # # # # #                     'skip.header.line.count': '1'
# # # # # # # # #                 }
# # # # # # # # #             }
# # # # # # # # #         }
# # # # # # # # #         try:
# # # # # # # # #             glue.update_table(
# # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # #                 TableInput={
# # # # # # # # #                     'Name': table_name,
# # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # #                 }
# # # # # # # # #             )
# # # # # # # # #             print(f"Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # # # #             print(f"Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # # # # # # #             glue.create_table(
# # # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # # #                 TableInput={
# # # # # # # # #                     'Name': table_name,
# # # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # # #                 }
# # # # # # # # #             )
# # # # # # # # #             print(f"Glue table '{table_name}' created successfully.")  # Debugging statement
# # # # # # # # #         except Exception as e:
# # # # # # # # #             print(f"Glue operation failed: {str(e)}")  # Debugging statement

# # # # # # # # #     def generate_notebook(self, request):
# # # # # # # # #         """
# # # # # # # # #         Generates two Jupyter Notebooks with pre-filled SQL queries based on the confirmed schema.
# # # # # # # # #         One for "Entity ID & Target", and another for "Features".
# # # # # # # # #         """
# # # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # # #         if user_id not in user_confirmations:
# # # # # # # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #         confirmation = user_confirmations[user_id]
# # # # # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # # # # #         target_column = confirmation['target_column']
# # # # # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # # # # #         # Get the table name from the uploaded file info
# # # # # # # # #         if user_id in user_schemas:
# # # # # # # # #             uploaded_file_info = user_schemas[user_id][0]
# # # # # # # # #             table_name = os.path.splitext(uploaded_file_info['name'])[0].lower().replace(' ', '_')
# # # # # # # # #         else:
# # # # # # # # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # # #         # Create Jupyter Notebooks
# # # # # # # # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name)
# # # # # # # # #         notebook_features = self.create_features_notebook(feature_columns, table_name)

# # # # # # # # #         # Store notebooks in user_notebooks dictionary
# # # # # # # # #         user_notebooks[user_id] = {
# # # # # # # # #             'entity_target_notebook': nbformat.writes(notebook_entity_target),
# # # # # # # # #             'features_notebook': nbformat.writes(notebook_features)
# # # # # # # # #         }

# # # # # # # # #         print("Notebooks generated and stored successfully.")  # Debugging statement

# # # # # # # # #         # Indicate success and that the "OPEN Notebook" button should be shown
# # # # # # # # #         return Response({
# # # # # # # # #             "message": "Notebooks generated successfully.",
# # # # # # # # #             "show_open_notebook": True
# # # # # # # # #         }, status=status.HTTP_200_OK)

# # # # # # # # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name):
# # # # # # # # #         """
# # # # # # # # #         Creates a Jupyter Notebook for Entity ID and Target analysis.
# # # # # # # # #         """
# # # # # # # # #         import nbformat
# # # # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # # # #         nb = new_notebook()
# # # # # # # # #         cells = []

# # # # # # # # #         # Introduction cell
# # # # # # # # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # # # # # # # #         # Athena connection setup cell
# # # # # # # # #         athena_s3_staging_dir = AWS_ATHENA_S3_STAGING_DIR
# # # # # # # # #         aws_region = AWS_REGION_NAME

# # # # # # # # #         cells.append(new_code_cell(
# # # # # # # # #             f"""
# # # # # # # # # import pandas as pd
# # # # # # # # # from pyathena import connect

# # # # # # # # # conn = connect(s3_staging_dir='{athena_s3_staging_dir}',
# # # # # # # # #                region_name='{aws_region}')
# # # # # # # # # """
# # # # # # # # #         ))

# # # # # # # # #         # Entity ID and Target analysis
# # # # # # # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM {table_name} LIMIT 100;"
# # # # # # # # #         cells.append(new_markdown_cell("## Entity ID and Target Column Analysis"))
# # # # # # # # #         cells.append(new_code_cell(f"df = pd.read_sql(\"\"\"{sql_query_entity_target}\"\"\", conn)\ndf.head()"))

# # # # # # # # #         nb['cells'] = cells

# # # # # # # # #         return nb

# # # # # # # # #     def create_features_notebook(self, feature_columns, table_name):
# # # # # # # # #         """
# # # # # # # # #         Creates a Jupyter Notebook for Features analysis.
# # # # # # # # #         """
# # # # # # # # #         import nbformat
# # # # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # # # #         nb = new_notebook()
# # # # # # # # #         cells = []

# # # # # # # # #         # Introduction cell
# # # # # # # # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # # # # # # # #         # Athena connection setup cell
# # # # # # # # #         athena_s3_staging_dir = AWS_ATHENA_S3_STAGING_DIR
# # # # # # # # #         aws_region = AWS_REGION_NAME

# # # # # # # # #         cells.append(new_code_cell(
# # # # # # # # #             f"""
# # # # # # # # # import pandas as pd
# # # # # # # # # from pyathena import connect

# # # # # # # # # conn = connect(s3_staging_dir='{athena_s3_staging_dir}',
# # # # # # # # #                region_name='{aws_region}')
# # # # # # # # # """
# # # # # # # # #         ))

# # # # # # # # #         # Feature columns analysis
# # # # # # # # #         for feature in feature_columns:
# # # # # # # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) FROM {table_name} GROUP BY {feature} LIMIT 100;"
# # # # # # # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))
# # # # # # # # #             cells.append(new_code_cell(f"df = pd.read_sql(\"\"\"{sql_query_feature}\"\"\", conn)\ndf.head()"))

# # # # # # # # #         nb['cells'] = cells

# # # # # # # # #         return nb


# # # # # # # # # chat/views.py

# # # # # # # # import os
# # # # # # # # import datetime
# # # # # # # # from io import BytesIO
# # # # # # # # from typing import Any, Dict, List
# # # # # # # # import boto3
# # # # # # # # import pandas as pd
# # # # # # # # import openai
# # # # # # # # import json
# # # # # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # # # # from django.conf import settings
# # # # # # # # from rest_framework import status
# # # # # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # # # # from rest_framework.response import Response
# # # # # # # # from rest_framework.views import APIView
# # # # # # # # from langchain.chains import ConversationChain
# # # # # # # # from langchain.chat_models import ChatOpenAI
# # # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # # from langchain.memory import ConversationBufferMemory
# # # # # # # # from langchain.schema import AIMessage, HumanMessage, SystemMessage
# # # # # # # # from .models import FileSchema, UploadedFile
# # # # # # # # from .serializers import UploadedFileSerializer

# # # # # # # # # ===========================
# # # # # # # # # AWS Configuration
# # # # # # # # # ===========================
# # # # # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # # # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # e.g., 's3://your-athena-query-results-bucket/'
# # # # # # # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # # # # # # ===========================
# # # # # # # # # OpenAI Configuration
# # # # # # # # # ===========================
# # # # # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # # # # openai.api_key = OPENAI_API_KEY

# # # # # # # # # ===========================
# # # # # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # # # # ===========================
# # # # # # # # llm_chatgpt = ChatOpenAI(
# # # # # # # #     model="gpt-3.5-turbo-16k",
# # # # # # # #     temperature=0.7,
# # # # # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # # # # )

# # # # # # # # # Global dictionaries to store user-specific data
# # # # # # # # user_conversations = {}
# # # # # # # # user_schemas = {}
# # # # # # # # user_confirmations = {}
# # # # # # # # user_notebook_flags = {}
# # # # # # # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # # # # # # Modify the prompt in the existing code
# # # # # # # # prompt_chatgpt = PromptTemplate(
# # # # # # # #     input_variables=["history", "user_input"],
# # # # # # # #     template=(
# # # # # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # # # # #         "Steps:\n"
# # # # # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # # # # #         "4. Reference the dataset schema if available.\n"
# # # # # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # # # # #         "Conversation history: {history}\n"
# # # # # # # #         "User input: {user_input}\n"
# # # # # # # #         "Assistant:"
# # # # # # # #     ),
# # # # # # # # )

# # # # # # # # memory = ConversationBufferMemory()
# # # # # # # # memory.chat_memory.add_message(SystemMessage(content="You are a helpful PACX AI assistant. Follow the steps strictly and assist users with predictive questions."))

# # # # # # # # conversation_chain_chatgpt = ConversationChain(
# # # # # # # #     llm=llm_chatgpt,
# # # # # # # #     prompt=prompt_chatgpt,
# # # # # # # #     input_key="user_input",
# # # # # # # #     memory=ConversationBufferMemory(),
# # # # # # # # )

# # # # # # # # # ===========================
# # # # # # # # # Utility Functions
# # # # # # # # # ===========================

# # # # # # # # def get_s3_client():
# # # # # # # #     """
# # # # # # # #     Creates and returns an AWS S3 client.
# # # # # # # #     """
# # # # # # # #     return boto3.client(
# # # # # # # #         's3',
# # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # #     )

# # # # # # # # def get_glue_client():
# # # # # # # #     """
# # # # # # # #     Creates and returns an AWS Glue client.
# # # # # # # #     """
# # # # # # # #     return boto3.client(
# # # # # # # #         'glue',
# # # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # # #     )

# # # # # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # # # # #     """
# # # # # # # #     Infers the correct data type for a column by handling mixed types.
# # # # # # # #     """
# # # # # # # #     series = series.dropna().astype(str).str.strip()

# # # # # # # #     # Try datetime
# # # # # # # #     try:
# # # # # # # #         pd.to_datetime(series, errors='raise', infer_datetime_format=True)
# # # # # # # #         return "timestamp"
# # # # # # # #     except ValueError:
# # # # # # # #         pass

# # # # # # # #     # Try boolean
# # # # # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # # # #     unique_values = set(series.str.lower().unique())
# # # # # # # #     if unique_values.issubset(boolean_values):
# # # # # # # #         return "boolean"

# # # # # # # #     # Try integer
# # # # # # # #     try:
# # # # # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # # # # #         if (int_series % 1 == 0).all():
# # # # # # # #             int_min = int_series.min()
# # # # # # # #             int_max = int_series.max()
# # # # # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # # # #                 return "int"
# # # # # # # #             else:
# # # # # # # #                 return "bigint"
# # # # # # # #     except ValueError:
# # # # # # # #         pass

# # # # # # # #     # Try double
# # # # # # # #     try:
# # # # # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # # # # #         return "double"
# # # # # # # #     except ValueError:
# # # # # # # #         pass

# # # # # # # #     # Default to string
# # # # # # # #     return "string"

# # # # # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # # # # #     """
# # # # # # # #     Suggests a target column based on user input or predictive question.
# # # # # # # #     """
# # # # # # # #     # Use the last column as a default suggestion
# # # # # # # #     return df.columns[-1]

# # # # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # # # #     """
# # # # # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # # # # #     """
# # # # # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # # # # #     for col in likely_id_columns:
# # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # #             return col

# # # # # # # #     # Fallback: Find any column with >95% unique values
# # # # # # # #     for col in df.columns:
# # # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # # #             return col
# # # # # # # #     return None

# # # # # # # # # ===========================
# # # # # # # # # Unified ChatGPT API
# # # # # # # # # ===========================
# # # # # # # # class UnifiedChatGPTAPI(APIView):
# # # # # # # #     """
# # # # # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # # # # #     Endpoint: /api/chatgpt/
# # # # # # # #     """
# # # # # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # # # # #     def post(self, request):
# # # # # # # #         """
# # # # # # # #         Handles POST requests for chat messages and file uploads.
# # # # # # # #         Differentiates based on the presence of files in the request.
# # # # # # # #         """
# # # # # # # #         action = request.data.get('action', '')
# # # # # # # #         if action == 'reset':
# # # # # # # #             return self.reset_conversation(request)
# # # # # # # #         if "file" in request.FILES:
# # # # # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # # # # #         # Handle 'Generate Notebook' action
# # # # # # # #         if action == 'generate_notebook':
# # # # # # # #             return self.generate_notebook(request)

# # # # # # # #         # Else, handle chat message
# # # # # # # #         return self.handle_chat(request)

# # # # # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # # # # #         """
# # # # # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # # # # #         After processing, appends schema details to the chat messages.
# # # # # # # #         """
# # # # # # # #         files = request.FILES.getlist("file")
# # # # # # # #         if not files:
# # # # # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # #         try:
# # # # # # # #             uploaded_files_info = []
# # # # # # # #             s3 = get_s3_client()
# # # # # # # #             glue = get_glue_client()

# # # # # # # #             for file in files:
# # # # # # # #                 # Validate file format
# # # # # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #                 # Read file into Pandas DataFrame
# # # # # # # #                 if file.name.lower().endswith('.csv'):
# # # # # # # #                     df = pd.read_csv(file)
# # # # # # # #                 else:
# # # # # # # #                     df = pd.read_excel(file)

# # # # # # # #                 # Normalize column headers
# # # # # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # # # #                 print(f"DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # # # # #                 # Infer schema with precision
# # # # # # # #                 schema = [
# # # # # # # #                     {
# # # # # # # #                         "column_name": col,
# # # # # # # #                         "data_type": infer_column_dtype(df[col])
# # # # # # # #                     }
# # # # # # # #                     for col in df.columns
# # # # # # # #                 ]
# # # # # # # #                 print(f"Inferred schema: {schema}")  # Debugging statement

# # # # # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # # # #                 for col in boolean_columns:
# # # # # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # # # #                 print(f"Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # # # # #                 # Handle Duplicate Files Dynamically
# # # # # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # # # # #                 if existing_file:
# # # # # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # # # #                     file.name = new_file_name
# # # # # # # #                     print(f"Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # # # # #                 else:
# # # # # # # #                     print(f"File name is unique: {file.name}")  # Debugging statement

# # # # # # # #                 # Save Metadata to Database
# # # # # # # #                 file.seek(0)
# # # # # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # # # # #                 if file_serializer.is_valid():
# # # # # # # #                     file_instance = file_serializer.save()

# # # # # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # # # # #                     csv_buffer = BytesIO()
# # # # # # # #                     df.to_csv(csv_buffer, index=False)
# # # # # # # #                     csv_buffer.seek(0)
# # # # # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # # # # #                     # Upload to AWS S3
# # # # # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # # # # # #                     print(f"File uploaded to S3: {file_key}")  # Debugging statement

# # # # # # # #                     # Generate file URL
# # # # # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # # # # #                     file_instance.file_url = file_url
# # # # # # # #                     file_instance.save()

# # # # # # # #                     # Save Schema to Database
# # # # # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # # # # # #                     print(f"Schema saved to database for file: {file.name}")  # Debugging statement

# # # # # # # #                     # Trigger AWS Glue Table Update
# # # # # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # # # #                     # Append file info to response
# # # # # # # #                     uploaded_files_info.append({
# # # # # # # #                         'id': file_instance.id,
# # # # # # # #                         'name': file_instance.name,
# # # # # # # #                         'file_url': file_instance.file_url,
# # # # # # # #                         'schema': schema,
# # # # # # # #                         'suggestions': {
# # # # # # # #                             'target_column': suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages),
# # # # # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # # # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages)]]
# # # # # # # #                         }
# # # # # # # #                     })

# # # # # # # #                 else:
# # # # # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #             # Store schema for user
# # # # # # # #             user_schemas[user_id] = uploaded_files_info

# # # # # # # #             # Initiate schema discussion with the user
# # # # # # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # # # # # #             if hasattr(conversation_chain_chatgpt.memory, "chat_memory"):
# # # # # # # #                 conversation_chain_chatgpt.memory.chat_memory.messages.append(
# # # # # # # #                     AIMessage(content=schema_discussion)
# # # # # # # #                 )
# # # # # # # #             print(f"Schema discussion initiated: {schema_discussion}")  # Debugging statement

# # # # # # # #             return Response({
# # # # # # # #                 "message": "Files uploaded and processed successfully.",
# # # # # # # #                 "uploaded_files": uploaded_files_info,
# # # # # # # #                 "chat_message": schema_discussion
# # # # # # # #             }, status=status.HTTP_201_CREATED)

# # # # # # # #         except pd.errors.EmptyDataError:
# # # # # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # # # #         except NoCredentialsError:
# # # # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # # #         except ClientError as e:
# # # # # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #     def handle_chat(self, request):
# # # # # # # #         user_input = request.data.get("message", "").strip()
# # # # # # # #         user_id = request.data.get("user_id", "default_user")

# # # # # # # #         if not user_input:
# # # # # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #         # Get or create conversation chain for the user
# # # # # # # #         if user_id not in user_conversations:
# # # # # # # #             conversation_chain = ConversationChain(
# # # # # # # #                 llm=llm_chatgpt,
# # # # # # # #                 prompt=prompt_chatgpt,
# # # # # # # #                 input_key="user_input",
# # # # # # # #                 memory=ConversationBufferMemory()
# # # # # # # #             )
# # # # # # # #             user_conversations[user_id] = conversation_chain
# # # # # # # #         else:
# # # # # # # #             conversation_chain = user_conversations[user_id]

# # # # # # # #         # Check if user is confirming schema
# # # # # # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # # # # # #             # Process user confirmation
# # # # # # # #             confirmation_response = self.process_schema_confirmation(user_input, user_id)
# # # # # # # #             if confirmation_response:
# # # # # # # #                 return Response({"response": confirmation_response})

# # # # # # # #         # Generate assistant response
# # # # # # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # # # # # #         print(f"Assistant response: {assistant_response}")  # Debugging statement

# # # # # # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # # # # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # # # # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # # # # # #             user_notebook_flags[user_id] = True  # Flag to show 'Generate Notebook' button
# # # # # # # #             print("GENERATE_NOTEBOOK_PROMPT detected. Flagging to show 'Generate Notebook' button.")  # Debugging statement

# # # # # # # #         return Response({
# # # # # # # #             "response": assistant_response,
# # # # # # # #             "show_generate_notebook": user_notebook_flags.get(user_id, False)
# # # # # # # #         })

# # # # # # # #     def process_schema_confirmation(self, user_input, user_id):
# # # # # # # #         """
# # # # # # # #         Processes user confirmation or adjustment of the schema.
# # # # # # # #         """
# # # # # # # #         uploaded_file_info = user_schemas[user_id][0]
# # # # # # # #         suggestions = uploaded_file_info['suggestions']

# # # # # # # #         # Assume user confirms or provides adjustments
# # # # # # # #         if 'yes' in user_input.lower():
# # # # # # # #             user_confirmations[user_id] = suggestions
# # # # # # # #             return "Schema confirmed. You can now click 'Generate Notebook' to proceed."
# # # # # # # #         else:
# # # # # # # #             # Parse user adjustments
# # # # # # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # # # # # #             if adjusted_columns:
# # # # # # # #                 user_confirmations[user_id] = adjusted_columns
# # # # # # # #                 return "Schema updated based on your inputs. You can now click 'Generate Notebook' to proceed."
# # # # # # # #             else:
# # # # # # # #                 return "Could not understand your adjustments. Please specify the correct 'Entity ID' and 'Target' column names."

# # # # # # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # # # # # #         """
# # # # # # # #         Parses user input for schema adjustments.
# # # # # # # #         """
# # # # # # # #         import re
# # # # # # # #         entity_id_match = re.search(r"Entity ID Column: (\w+)", user_input, re.IGNORECASE)
# # # # # # # #         target_column_match = re.search(r"Target Column: (\w+)", user_input, re.IGNORECASE)

# # # # # # # #         suggestions = uploaded_file_info['suggestions']
# # # # # # # #         entity_id_column = suggestions['entity_id_column']
# # # # # # # #         target_column = suggestions['target_column']

# # # # # # # #         if entity_id_match:
# # # # # # # #             entity_id_column = entity_id_match.group(1)
# # # # # # # #         if target_column_match:
# # # # # # # #             target_column = target_column_match.group(1)

# # # # # # # #         if entity_id_column and target_column:
# # # # # # # #             return {
# # # # # # # #                 'entity_id_column': entity_id_column,
# # # # # # # #                 'target_column': target_column,
# # # # # # # #                 'feature_columns': [col for col in uploaded_file_info['schema'] if col['column_name'] not in [entity_id_column, target_column]]
# # # # # # # #             }
# # # # # # # #         else:
# # # # # # # #             return None

# # # # # # # #     def reset_conversation(self, request):
# # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # #         # Remove user's conversation chain
# # # # # # # #         if user_id in user_conversations:
# # # # # # # #             del user_conversations[user_id]
# # # # # # # #         # Remove user's uploaded schema and confirmations
# # # # # # # #         if user_id in user_schemas:
# # # # # # # #             del user_schemas[user_id]
# # # # # # # #         if user_id in user_confirmations:
# # # # # # # #             del user_confirmations[user_id]
# # # # # # # #         if user_id in user_notebook_flags:
# # # # # # # #             del user_notebook_flags[user_id]
# # # # # # # #         if user_id in user_notebooks:
# # # # # # # #             del user_notebooks[user_id]
# # # # # # # #         print(f"Conversation reset for user: {user_id}")  # Debugging statement
# # # # # # # #         return Response({"message": "Conversation reset successful."})

# # # # # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # # # # #         """
# # # # # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # # # # #         """
# # # # # # # #         schema = uploaded_file['schema']
# # # # # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # # # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # # # # # #         schema_text = (
# # # # # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # # # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # # # # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # # # # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # # # #             "Please confirm:\n"
# # # # # # # #             "- Is the Target Column correct?\n"
# # # # # # # #             "- Is the Entity ID Column correct?\n"
# # # # # # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # # # # # #         )
# # # # # # # #         return schema_text

# # # # # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # # # # #         """
# # # # # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # # # # #         """
# # # # # # # #         glue = get_glue_client()
# # # # # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # # # # #         storage_descriptor = {
# # # # # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # # # # #             'Location': s3_location,
# # # # # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # # # # #             'SerdeInfo': {
# # # # # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # # # # #                 'Parameters': {
# # # # # # # #                     'field.delim': ',',
# # # # # # # #                     'skip.header.line.count': '1'
# # # # # # # #                 }
# # # # # # # #             }
# # # # # # # #         }
# # # # # # # #         try:
# # # # # # # #             glue.update_table(
# # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # #                 TableInput={
# # # # # # # #                     'Name': table_name,
# # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # #                 }
# # # # # # # #             )
# # # # # # # #             print(f"Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # # #             print(f"Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # # # # # #             glue.create_table(
# # # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # # #                 TableInput={
# # # # # # # #                     'Name': table_name,
# # # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # # #                 }
# # # # # # # #             )
# # # # # # # #             print(f"Glue table '{table_name}' created successfully.")  # Debugging statement
# # # # # # # #         except Exception as e:
# # # # # # # #             print(f"Glue operation failed: {str(e)}")  # Debugging statement

# # # # # # # #     def generate_notebook(self, request):
# # # # # # # #         """
# # # # # # # #         Generates two Jupyter Notebooks with pre-filled SQL queries based on the confirmed schema.
# # # # # # # #         One for "Entity ID & Target", and another for "Features".
# # # # # # # #         """
# # # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # # #         if user_id not in user_confirmations:
# # # # # # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #         confirmation = user_confirmations[user_id]
# # # # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # # # #         target_column = confirmation['target_column']
# # # # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # # # #         # Get the table name from the uploaded file info
# # # # # # # #         if user_id in user_schemas:
# # # # # # # #             uploaded_file_info = user_schemas[user_id][0]
# # # # # # # #             table_name = os.path.splitext(uploaded_file_info['name'])[0].lower().replace(' ', '_')
# # # # # # # #         else:
# # # # # # # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # # #         # Create Jupyter Notebooks with SQL queries
# # # # # # # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name)
# # # # # # # #         notebook_features = self.create_features_notebook(feature_columns, table_name)

# # # # # # # #         # Store notebooks in user_notebooks dictionary
# # # # # # # #         user_notebooks[user_id] = {
# # # # # # # #             'entity_target_notebook': json.dumps(notebook_entity_target),
# # # # # # # #             'features_notebook': json.dumps(notebook_features)
# # # # # # # #         }

# # # # # # # #         print("Notebooks generated and stored successfully.")  # Debugging statement

# # # # # # # #         # Include the notebooks in the response
# # # # # # # #         return Response({
# # # # # # # #             "message": "Notebooks generated successfully.",
# # # # # # # #             "show_open_notebook": True,
# # # # # # # #             "notebooks": user_notebooks[user_id]
# # # # # # # #         }, status=status.HTTP_200_OK)

# # # # # # # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name):
# # # # # # # #         """
# # # # # # # #         Creates a Jupyter Notebook for Entity ID and Target analysis with SQL queries.
# # # # # # # #         """
# # # # # # # #         import nbformat
# # # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # # #         nb = new_notebook()
# # # # # # # #         cells = []

# # # # # # # #         # Introduction cell
# # # # # # # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # # # # # # #         # SQL query cell
# # # # # # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM {table_name} LIMIT 100;"

# # # # # # # #         # Add the SQL query to the cell
# # # # # # # #         cells.append(new_code_cell(sql_query_entity_target))

# # # # # # # #         nb['cells'] = cells

# # # # # # # #         return nb

# # # # # # # #     def create_features_notebook(self, feature_columns, table_name):
# # # # # # # #         """
# # # # # # # #         Creates a Jupyter Notebook for Features analysis with SQL queries.
# # # # # # # #         """
# # # # # # # #         import nbformat
# # # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # # #         nb = new_notebook()
# # # # # # # #         cells = []

# # # # # # # #         # Introduction cell
# # # # # # # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # # # # # # #         # Feature columns analysis
# # # # # # # #         for feature in feature_columns:
# # # # # # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) FROM {table_name} GROUP BY {feature} LIMIT 100;"
# # # # # # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))
# # # # # # # #             cells.append(new_code_cell(sql_query_feature))

# # # # # # # #         nb['cells'] = cells

# # # # # # # #         return nb



# # # # # # # # chat/views.py

# # # # # # # import os
# # # # # # # import datetime
# # # # # # # from io import BytesIO
# # # # # # # from typing import Any, Dict, List
# # # # # # # import boto3
# # # # # # # import pandas as pd
# # # # # # # import openai
# # # # # # # import json
# # # # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # # # from django.conf import settings
# # # # # # # from rest_framework import status
# # # # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # # # from rest_framework.response import Response
# # # # # # # from rest_framework.views import APIView
# # # # # # # from langchain.chains import ConversationChain
# # # # # # # from langchain.chat_models import ChatOpenAI
# # # # # # # from langchain.prompts import PromptTemplate
# # # # # # # from langchain.memory import ConversationBufferMemory
# # # # # # # from langchain.schema import AIMessage, HumanMessage, SystemMessage
# # # # # # # from .models import FileSchema, UploadedFile
# # # # # # # from .serializers import UploadedFileSerializer

# # # # # # # # ===========================
# # # # # # # # AWS Configuration
# # # # # # # # ===========================
# # # # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # e.g., 's3://your-athena-query-results-bucket/'
# # # # # # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # # # # # ===========================
# # # # # # # # OpenAI Configuration
# # # # # # # # ===========================
# # # # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # # # openai.api_key = OPENAI_API_KEY

# # # # # # # # ===========================
# # # # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # # # ===========================
# # # # # # # llm_chatgpt = ChatOpenAI(
# # # # # # #     model="gpt-3.5-turbo-16k",
# # # # # # #     temperature=0.7,
# # # # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # # # )

# # # # # # # # Global dictionaries to store user-specific data
# # # # # # # user_conversations = {}
# # # # # # # user_schemas = {}
# # # # # # # user_confirmations = {}
# # # # # # # user_notebook_flags = {}
# # # # # # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # # # # # Modify the prompt in the existing code
# # # # # # # prompt_chatgpt = PromptTemplate(
# # # # # # #     input_variables=["history", "user_input"],
# # # # # # #     template=(
# # # # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # # # #         "Steps:\n"
# # # # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # # # #         "4. Reference the dataset schema if available.\n"
# # # # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # # # #         "Conversation history: {history}\n"
# # # # # # #         "User input: {user_input}\n"
# # # # # # #         "Assistant:"
# # # # # # #     ),
# # # # # # # )

# # # # # # # # ===========================
# # # # # # # # Utility Functions
# # # # # # # # ===========================

# # # # # # # def get_s3_client():
# # # # # # #     """
# # # # # # #     Creates and returns an AWS S3 client.
# # # # # # #     """
# # # # # # #     return boto3.client(
# # # # # # #         's3',
# # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # #     )

# # # # # # # def get_glue_client():
# # # # # # #     """
# # # # # # #     Creates and returns an AWS Glue client.
# # # # # # #     """
# # # # # # #     return boto3.client(
# # # # # # #         'glue',
# # # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # # #     )

# # # # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # # # #     """
# # # # # # #     Infers the correct data type for a column by handling mixed types.
# # # # # # #     """
# # # # # # #     series = series.dropna().astype(str).str.strip()

# # # # # # #     # Try datetime
# # # # # # #     try:
# # # # # # #         pd.to_datetime(series, errors='raise')
# # # # # # #         return "timestamp"
# # # # # # #     except ValueError:
# # # # # # #         pass

# # # # # # #     # Try boolean
# # # # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # # #     unique_values = set(series.str.lower().unique())
# # # # # # #     if unique_values.issubset(boolean_values):
# # # # # # #         return "boolean"

# # # # # # #     # Try integer
# # # # # # #     try:
# # # # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # # # #         if (int_series % 1 == 0).all():
# # # # # # #             int_min = int_series.min()
# # # # # # #             int_max = int_series.max()
# # # # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # # #                 return "int"
# # # # # # #             else:
# # # # # # #                 return "bigint"
# # # # # # #     except ValueError:
# # # # # # #         pass

# # # # # # #     # Try double
# # # # # # #     try:
# # # # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # # # #         return "double"
# # # # # # #     except ValueError:
# # # # # # #         pass

# # # # # # #     # Default to string
# # # # # # #     return "string"

# # # # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # # # #     """
# # # # # # #     Suggests a target column based on user input or predictive question.
# # # # # # #     """
# # # # # # #     # Use the last column as a default suggestion
# # # # # # #     return df.columns[-1]

# # # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # # #     """
# # # # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # # # #     """
# # # # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # # # #     for col in likely_id_columns:
# # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # #             return col

# # # # # # #     # Fallback: Find any column with >95% unique values
# # # # # # #     for col in df.columns:
# # # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # # #             return col
# # # # # # #     return None

# # # # # # # # ===========================
# # # # # # # # Unified ChatGPT API
# # # # # # # # ===========================
# # # # # # # class UnifiedChatGPTAPI(APIView):
# # # # # # #     """
# # # # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # # # #     Endpoint: /api/chatgpt/
# # # # # # #     """
# # # # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # # # #     def post(self, request):
# # # # # # #         """
# # # # # # #         Handles POST requests for chat messages and file uploads.
# # # # # # #         Differentiates based on the presence of files in the request.
# # # # # # #         """
# # # # # # #         action = request.data.get('action', '')
# # # # # # #         if action == 'reset':
# # # # # # #             return self.reset_conversation(request)
# # # # # # #         if action == 'generate_notebook':
# # # # # # #             return self.generate_notebook(request)
# # # # # # #         if "file" in request.FILES:
# # # # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # # # #         # Else, handle chat message
# # # # # # #         return self.handle_chat(request)

# # # # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # # # #         """
# # # # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # # # #         After processing, appends schema details to the chat messages.
# # # # # # #         """
# # # # # # #         files = request.FILES.getlist("file")
# # # # # # #         if not files:
# # # # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # #         print(f"[DEBUG] Handling file upload for user: {user_id}")

# # # # # # #         try:
# # # # # # #             uploaded_files_info = []
# # # # # # #             s3 = get_s3_client()
# # # # # # #             glue = get_glue_client()

# # # # # # #             for file in files:
# # # # # # #                 print(f"[DEBUG] Processing file: {file.name}")
# # # # # # #                 # Validate file format
# # # # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #                 # Read file into Pandas DataFrame
# # # # # # #                 if file.name.lower().endswith('.csv'):
# # # # # # #                     df = pd.read_csv(file)
# # # # # # #                 else:
# # # # # # #                     df = pd.read_excel(file)

# # # # # # #                 # Normalize column headers
# # # # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # # #                 print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # # # #                 # Infer schema with precision
# # # # # # #                 schema = [
# # # # # # #                     {
# # # # # # #                         "column_name": col,
# # # # # # #                         "data_type": infer_column_dtype(df[col])
# # # # # # #                     }
# # # # # # #                     for col in df.columns
# # # # # # #                 ]
# # # # # # #                 print(f"[DEBUG] Inferred schema: {schema}")  # Debugging statement

# # # # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # # #                 for col in boolean_columns:
# # # # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # # #                 print(f"[DEBUG] Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # # # #                 # Handle Duplicate Files Dynamically
# # # # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # # # #                 if existing_file:
# # # # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # # #                     file.name = new_file_name
# # # # # # #                     print(f"[DEBUG] Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # # # #                 else:
# # # # # # #                     print(f"[DEBUG] File name is unique: {file.name}")  # Debugging statement

# # # # # # #                 # Save Metadata to Database
# # # # # # #                 file.seek(0)
# # # # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # # # #                 if file_serializer.is_valid():
# # # # # # #                     file_instance = file_serializer.save()

# # # # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # # # #                     csv_buffer = BytesIO()
# # # # # # #                     df.to_csv(csv_buffer, index=False)
# # # # # # #                     csv_buffer.seek(0)
# # # # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # # # #                     # Upload to AWS S3
# # # # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # # # # #                     print(f"[DEBUG] File uploaded to S3: {file_key}")  # Debugging statement

# # # # # # #                     # Generate file URL
# # # # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # # # #                     file_instance.file_url = file_url
# # # # # # #                     file_instance.save()

# # # # # # #                     # Save Schema to Database
# # # # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # # # # #                     print(f"[DEBUG] Schema saved to database for file: {file.name}")  # Debugging statement

# # # # # # #                     # Trigger AWS Glue Table Update
# # # # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # # #                     # Append file info to response
# # # # # # #                     uploaded_files_info.append({
# # # # # # #                         'id': file_instance.id,
# # # # # # #                         'name': file_instance.name,
# # # # # # #                         'file_url': file_instance.file_url,
# # # # # # #                         'schema': schema,
# # # # # # #                         'suggestions': {
# # # # # # #                             'target_column': suggest_target_column(df, []),
# # # # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
# # # # # # #                         }
# # # # # # #                     })

# # # # # # #                 else:
# # # # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #             # Store schema for user
# # # # # # #             user_schemas[user_id] = uploaded_files_info
# # # # # # #             print(f"[DEBUG] Stored schema for user: {user_id}")

# # # # # # #             # Initiate schema discussion with the user
# # # # # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # # # # #             print(f"[DEBUG] Schema discussion message: {schema_discussion}")  # Debugging statement

# # # # # # #             # Create or get user's conversation chain
# # # # # # #             if user_id not in user_conversations:
# # # # # # #                 conversation_chain = ConversationChain(
# # # # # # #                     llm=llm_chatgpt,
# # # # # # #                     prompt=prompt_chatgpt,
# # # # # # #                     input_key="user_input",
# # # # # # #                     memory=ConversationBufferMemory()
# # # # # # #                 )
# # # # # # #                 user_conversations[user_id] = conversation_chain
# # # # # # #             else:
# # # # # # #                 conversation_chain = user_conversations[user_id]

# # # # # # #             # Add the schema discussion to the assistant's messages
# # # # # # #             conversation_chain.memory.chat_memory.messages.append(
# # # # # # #                 AIMessage(content=schema_discussion)
# # # # # # #             )

# # # # # # #             return Response({
# # # # # # #                 "message": "Files uploaded and processed successfully.",
# # # # # # #                 "uploaded_files": uploaded_files_info,
# # # # # # #                 "chat_message": schema_discussion
# # # # # # #             }, status=status.HTTP_201_CREATED)

# # # # # # #         except pd.errors.EmptyDataError:
# # # # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # # #         except NoCredentialsError:
# # # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # # #         except ClientError as e:
# # # # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # # #         except Exception as e:
# # # # # # #             print(f"[ERROR] Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #     def handle_chat(self, request):
# # # # # # #         user_input = request.data.get("message", "").strip()
# # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # #         print(f"[DEBUG] Handling chat for user: {user_id}, message: {user_input}")

# # # # # # #         if not user_input:
# # # # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #         # Get or create conversation chain for the user
# # # # # # #         if user_id not in user_conversations:
# # # # # # #             conversation_chain = ConversationChain(
# # # # # # #                 llm=llm_chatgpt,
# # # # # # #                 prompt=prompt_chatgpt,
# # # # # # #                 input_key="user_input",
# # # # # # #                 memory=ConversationBufferMemory()
# # # # # # #             )
# # # # # # #             user_conversations[user_id] = conversation_chain
# # # # # # #         else:
# # # # # # #             conversation_chain = user_conversations[user_id]

# # # # # # #         # Check if user is confirming schema
# # # # # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # # # # #             # Process user confirmation
# # # # # # #             assistant_response = self.process_schema_confirmation(user_input, user_id)
# # # # # # #             return Response({"response": assistant_response, "show_generate_notebook": True})

# # # # # # #         # Generate assistant response
# # # # # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # # # # #         print(f"[DEBUG] Assistant response: {assistant_response}")  # Debugging statement

# # # # # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # # # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # # # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # # # # #             print(f"[DEBUG] GENERATE_NOTEBOOK_PROMPT detected for user: {user_id}")
# # # # # # #             return Response({
# # # # # # #                 "response": assistant_response,
# # # # # # #                 "show_generate_notebook": True
# # # # # # #             })

# # # # # # #         return Response({
# # # # # # #             "response": assistant_response
# # # # # # #         })

# # # # # # #     def process_schema_confirmation(self, user_input, user_id):
# # # # # # #         """
# # # # # # #         Processes user confirmation or adjustment of the schema.
# # # # # # #         """
# # # # # # #         uploaded_file_info = user_schemas[user_id][0]
# # # # # # #         suggestions = uploaded_file_info['suggestions']

# # # # # # #         # Assume user confirms or provides adjustments
# # # # # # #         if 'yes' in user_input.lower():
# # # # # # #             user_confirmations[user_id] = suggestions
# # # # # # #             # Provide confirmed details and prompt to generate notebook
# # # # # # #             assistant_response = self.format_confirmation_message(suggestions)
# # # # # # #             return assistant_response
# # # # # # #         else:
# # # # # # #             # Parse user adjustments
# # # # # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # # # # #             if adjusted_columns:
# # # # # # #                 user_confirmations[user_id] = adjusted_columns
# # # # # # #                 # Provide confirmed details and prompt to generate notebook
# # # # # # #                 assistant_response = self.format_confirmation_message(adjusted_columns)
# # # # # # #                 return assistant_response
# # # # # # #             else:
# # # # # # #                 return "Could not understand your adjustments. Please specify the correct 'Entity ID' and 'Target' column names."

# # # # # # #     def format_confirmation_message(self, confirmation):
# # # # # # #         """
# # # # # # #         Formats the confirmation message with confirmed details and includes 'GENERATE_NOTEBOOK_PROMPT'.
# # # # # # #         """
# # # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # # #         target_column = confirmation['target_column']
# # # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # # #         confirmation_text = (
# # # # # # #             f"Great! You've confirmed the following details:\n\n"
# # # # # # #             f"Entity ID Column: {entity_id_column}\n"
# # # # # # #             f"Target Column: {target_column}\n"
# # # # # # #             f"Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # # #             "You can now generate the notebook to proceed with your analysis."
# # # # # # #             "\n\nGENERATE_NOTEBOOK_PROMPT"
# # # # # # #         )
# # # # # # #         return confirmation_text

# # # # # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # # # # #         """
# # # # # # #         Parses user input for schema adjustments.
# # # # # # #         """
# # # # # # #         import re
# # # # # # #         entity_id_match = re.search(r"Entity ID Column: (\w+)", user_input, re.IGNORECASE)
# # # # # # #         target_column_match = re.search(r"Target Column: (\w+)", user_input, re.IGNORECASE)

# # # # # # #         suggestions = uploaded_file_info['suggestions']
# # # # # # #         entity_id_column = suggestions['entity_id_column']
# # # # # # #         target_column = suggestions['target_column']

# # # # # # #         if entity_id_match:
# # # # # # #             entity_id_column = entity_id_match.group(1)
# # # # # # #         if target_column_match:
# # # # # # #             target_column = target_column_match.group(1)

# # # # # # #         if entity_id_column and target_column:
# # # # # # #             return {
# # # # # # #                 'entity_id_column': entity_id_column,
# # # # # # #                 'target_column': target_column,
# # # # # # #                 'feature_columns': [col for col in uploaded_file_info['schema'] if col['column_name'] not in [entity_id_column, target_column]]
# # # # # # #             }
# # # # # # #         else:
# # # # # # #             return None

# # # # # # #     def reset_conversation(self, request):
# # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # #         # Remove user's conversation chain
# # # # # # #         if user_id in user_conversations:
# # # # # # #             del user_conversations[user_id]
# # # # # # #         # Remove user's uploaded schema and confirmations
# # # # # # #         if user_id in user_schemas:
# # # # # # #             del user_schemas[user_id]
# # # # # # #         if user_id in user_confirmations:
# # # # # # #             del user_confirmations[user_id]
# # # # # # #         if user_id in user_notebook_flags:
# # # # # # #             del user_notebook_flags[user_id]
# # # # # # #         if user_id in user_notebooks:
# # # # # # #             del user_notebooks[user_id]
# # # # # # #         print(f"[DEBUG] Conversation reset for user: {user_id}")  # Debugging statement
# # # # # # #         return Response({"message": "Conversation reset successful."})

# # # # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # # # #         """
# # # # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # # # #         """
# # # # # # #         schema = uploaded_file['schema']
# # # # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # # # # #         schema_text = (
# # # # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # # # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # # # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # # #             "Please confirm:\n"
# # # # # # #             "- Is the Target Column correct?\n"
# # # # # # #             "- Is the Entity ID Column correct?\n"
# # # # # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # # # # #         )
# # # # # # #         return schema_text

# # # # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # # # #         """
# # # # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # # # #         """
# # # # # # #         glue = get_glue_client()
# # # # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # # # #         storage_descriptor = {
# # # # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # # # #             'Location': s3_location,
# # # # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # # # #             'SerdeInfo': {
# # # # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # # # #                 'Parameters': {
# # # # # # #                     'field.delim': ',',
# # # # # # #                     'skip.header.line.count': '1'
# # # # # # #                 }
# # # # # # #             }
# # # # # # #         }
# # # # # # #         try:
# # # # # # #             glue.update_table(
# # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # #                 TableInput={
# # # # # # #                     'Name': table_name,
# # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # #                 }
# # # # # # #             )
# # # # # # #             print(f"[DEBUG] Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # # #             print(f"[DEBUG] Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # # # # #             glue.create_table(
# # # # # # #                 DatabaseName='pa_user_datafiles_db',
# # # # # # #                 TableInput={
# # # # # # #                     'Name': table_name,
# # # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # # #                 }
# # # # # # #             )
# # # # # # #             print(f"[DEBUG] Glue table '{table_name}' created successfully.")  # Debugging statement
# # # # # # #         except Exception as e:
# # # # # # #             print(f"[ERROR] Glue operation failed: {str(e)}")  # Debugging statement

# # # # # # #     def generate_notebook(self, request):
# # # # # # #         """
# # # # # # #         Generates notebooks with pre-filled SQL queries and executed results.
# # # # # # #         """
# # # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # # #         print(f"[DEBUG] Generating notebook for user: {user_id}")

# # # # # # #         if user_id not in user_confirmations:
# # # # # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #         confirmation = user_confirmations[user_id]
# # # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # # #         target_column = confirmation['target_column']
# # # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # # #         # Get the table name from the uploaded file info
# # # # # # #         if user_id in user_schemas:
# # # # # # #             uploaded_file_info = user_schemas[user_id][0]
# # # # # # #             table_name = os.path.splitext(uploaded_file_info['name'])[0].lower().replace(' ', '_')
# # # # # # #         else:
# # # # # # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # # #         # Create notebooks with SQL queries and execute them to get results
# # # # # # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name)
# # # # # # #         notebook_features = self.create_features_notebook(feature_columns, table_name)

# # # # # # #         # Store notebooks in user_notebooks dictionary
# # # # # # #         user_notebooks[user_id] = {
# # # # # # #             'entity_target_notebook': notebook_entity_target,
# # # # # # #             'features_notebook': notebook_features
# # # # # # #         }

# # # # # # #         print("[DEBUG] Notebooks generated and stored successfully.")  # Debugging statement

# # # # # # #         return Response({
# # # # # # #             "message": "Notebooks generated successfully.",
# # # # # # #             "notebooks": user_notebooks[user_id]
# # # # # # #         }, status=status.HTTP_200_OK)

# # # # # # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name):
# # # # # # #         """
# # # # # # #         Creates a notebook for Entity ID and Target analysis with SQL queries and executed results.
# # # # # # #         """
# # # # # # #         import nbformat
# # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # #         nb = new_notebook()
# # # # # # #         cells = []

# # # # # # #         # Introduction cell
# # # # # # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # # # # # #         # SQL query cell
# # # # # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM {table_name} LIMIT 100;"

# # # # # # #         # Add the SQL query to the cell
# # # # # # #         cells.append(new_code_cell(sql_query_entity_target))

# # # # # # #         # Execute the query and get results
# # # # # # #         result = self.execute_sql_query(sql_query_entity_target)

# # # # # # #         # Add the results to the notebook
# # # # # # #         cells.append(new_code_cell(f"Result:\n{result}"))

# # # # # # #         nb['cells'] = cells

# # # # # # #         return nbformat.writes(nb)

# # # # # # #     def create_features_notebook(self, feature_columns, table_name):
# # # # # # #         """
# # # # # # #         Creates a notebook for Features analysis with SQL queries and executed results.
# # # # # # #         """
# # # # # # #         import nbformat
# # # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # # #         nb = new_notebook()
# # # # # # #         cells = []

# # # # # # #         # Introduction cell
# # # # # # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # # # # # #         # Feature columns analysis
# # # # # # #         for feature in feature_columns:
# # # # # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) FROM {table_name} GROUP BY {feature} LIMIT 100;"
# # # # # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))
# # # # # # #             cells.append(new_code_cell(sql_query_feature))

# # # # # # #             # Execute the query and get results
# # # # # # #             result = self.execute_sql_query(sql_query_feature)
# # # # # # #             cells.append(new_code_cell(f"Result:\n{result}"))

# # # # # # #         nb['cells'] = cells

# # # # # # #         return nbformat.writes(nb)

# # # # # # #     def execute_sql_query(self, query):
# # # # # # #         """
# # # # # # #         Executes a SQL query using AWS Athena and returns the results.
# # # # # # #         """
# # # # # # #         from pyathena import connect
# # # # # # #         try:
# # # # # # #             conn = connect(s3_staging_dir=AWS_ATHENA_S3_STAGING_DIR, region_name=AWS_REGION_NAME)
# # # # # # #             df = pd.read_sql(query, conn)
# # # # # # #             result = df.to_csv(index=False)
# # # # # # #             print(f"[DEBUG] Query executed successfully: {query}")
# # # # # # #             return result
# # # # # # #         except Exception as e:
# # # # # # #             print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
# # # # # # #             return f"Error executing query: {str(e)}"




# # # # # # # chat/views.py

# # # # # # import os
# # # # # # import datetime
# # # # # # from io import BytesIO
# # # # # # from typing import Any, Dict, List
# # # # # # import boto3
# # # # # # import pandas as pd
# # # # # # import openai
# # # # # # import json
# # # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # # from django.conf import settings
# # # # # # from rest_framework import status
# # # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # # from rest_framework.response import Response
# # # # # # from rest_framework.views import APIView
# # # # # # from langchain.chains import ConversationChain
# # # # # # from langchain.chat_models import ChatOpenAI
# # # # # # from langchain.prompts import PromptTemplate
# # # # # # from langchain.memory import ConversationBufferMemory
# # # # # # from langchain.schema import AIMessage
# # # # # # from .models import FileSchema, UploadedFile
# # # # # # from .serializers import UploadedFileSerializer

# # # # # # # ===========================
# # # # # # # AWS Configuration
# # # # # # # ===========================
# # # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # e.g., 's3://your-athena-query-results-bucket/'
# # # # # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # # # # Set the Athena database (schema) name
# # # # # # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Replace with your actual Athena database name

# # # # # # # ===========================
# # # # # # # OpenAI Configuration
# # # # # # # ===========================
# # # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # # openai.api_key = OPENAI_API_KEY

# # # # # # # ===========================
# # # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # # ===========================
# # # # # # llm_chatgpt = ChatOpenAI(
# # # # # #     model="gpt-3.5-turbo-16k",
# # # # # #     temperature=0.7,
# # # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # # )

# # # # # # # Global dictionaries to store user-specific data
# # # # # # user_conversations = {}
# # # # # # user_schemas = {}
# # # # # # user_confirmations = {}
# # # # # # user_notebook_flags = {}
# # # # # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # # # # Modify the prompt in the existing code
# # # # # # prompt_chatgpt = PromptTemplate(
# # # # # #     input_variables=["history", "user_input"],
# # # # # #     template=(
# # # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # # #         "Steps:\n"
# # # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # # #         "4. Reference the dataset schema if available.\n"
# # # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # # #         "Conversation history: {history}\n"
# # # # # #         "User input: {user_input}\n"
# # # # # #         "Assistant:"
# # # # # #     ),
# # # # # # )

# # # # # # # ===========================
# # # # # # # Utility Functions
# # # # # # # ===========================

# # # # # # def get_s3_client():
# # # # # #     """
# # # # # #     Creates and returns an AWS S3 client.
# # # # # #     """
# # # # # #     return boto3.client(
# # # # # #         's3',
# # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # #     )

# # # # # # def get_glue_client():
# # # # # #     """
# # # # # #     Creates and returns an AWS Glue client.
# # # # # #     """
# # # # # #     return boto3.client(
# # # # # #         'glue',
# # # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # #         region_name=AWS_S3_REGION_NAME
# # # # # #     )

# # # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # # #     """
# # # # # #     Infers the correct data type for a column by handling mixed types.
# # # # # #     """
# # # # # #     series = series.dropna().astype(str).str.strip()

# # # # # #     # Try datetime
# # # # # #     try:
# # # # # #         pd.to_datetime(series, errors='raise')
# # # # # #         return "timestamp"
# # # # # #     except ValueError:
# # # # # #         pass

# # # # # #     # Try boolean
# # # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # # #     unique_values = set(series.str.lower().unique())
# # # # # #     if unique_values.issubset(boolean_values):
# # # # # #         return "boolean"

# # # # # #     # Try integer
# # # # # #     try:
# # # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # # #         if (int_series % 1 == 0).all():
# # # # # #             int_min = int_series.min()
# # # # # #             int_max = int_series.max()
# # # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # # #                 return "int"
# # # # # #             else:
# # # # # #                 return "bigint"
# # # # # #     except ValueError:
# # # # # #         pass

# # # # # #     # Try double
# # # # # #     try:
# # # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # # #         return "double"
# # # # # #     except ValueError:
# # # # # #         pass

# # # # # #     # Default to string
# # # # # #     return "string"

# # # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # # #     """
# # # # # #     Suggests a target column based on user input or predictive question.
# # # # # #     """
# # # # # #     # Use the last column as a default suggestion
# # # # # #     return df.columns[-1]

# # # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # # #     """
# # # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # # #     """
# # # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # # #     for col in likely_id_columns:
# # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # #             return col

# # # # # #     # Fallback: Find any column with >95% unique values
# # # # # #     for col in df.columns:
# # # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # # #             return col
# # # # # #     return None

# # # # # # def execute_sql_query(query: str):
# # # # # #     """
# # # # # #     Executes a SQL query using AWS Athena and returns the results as a Pandas DataFrame.
# # # # # #     """
# # # # # #     from pyathena import connect
# # # # # #     try:
# # # # # #         conn = connect(
# # # # # #             aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # # #             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # # #             s3_staging_dir=AWS_ATHENA_S3_STAGING_DIR,
# # # # # #             region_name=AWS_REGION_NAME,
# # # # # #             schema_name=ATHENA_SCHEMA_NAME  # Specify the Athena database (schema)
# # # # # #         )
# # # # # #         df = pd.read_sql(query, conn)
# # # # # #         print(f"[DEBUG] Query executed successfully: {query}")
# # # # # #         return df
# # # # # #     except Exception as e:
# # # # # #         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
# # # # # #         return pd.DataFrame()  # Return an empty DataFrame on error

# # # # # # # ===========================
# # # # # # # Unified ChatGPT API
# # # # # # # ===========================
# # # # # # class UnifiedChatGPTAPI(APIView):
# # # # # #     """
# # # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # # #     Endpoint: /api/chatgpt/
# # # # # #     """
# # # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # # #     def post(self, request):
# # # # # #         """
# # # # # #         Handles POST requests for chat messages and file uploads.
# # # # # #         Differentiates based on the presence of files in the request.
# # # # # #         """
# # # # # #         action = request.data.get('action', '')
# # # # # #         if action == 'reset':
# # # # # #             return self.reset_conversation(request)
# # # # # #         if action == 'generate_notebook':
# # # # # #             return self.generate_notebook(request)
# # # # # #         if "file" in request.FILES:
# # # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # # #         # Else, handle chat message
# # # # # #         return self.handle_chat(request)

# # # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # # #         """
# # # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # # #         After processing, appends schema details to the chat messages.
# # # # # #         """
# # # # # #         files = request.FILES.getlist("file")
# # # # # #         if not files:
# # # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # #         print(f"[DEBUG] Handling file upload for user: {user_id}")

# # # # # #         try:
# # # # # #             uploaded_files_info = []
# # # # # #             s3 = get_s3_client()
# # # # # #             glue = get_glue_client()

# # # # # #             for file in files:
# # # # # #                 print(f"[DEBUG] Processing file: {file.name}")
# # # # # #                 # Validate file format
# # # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # #                 # Read file into Pandas DataFrame
# # # # # #                 if file.name.lower().endswith('.csv'):
# # # # # #                     df = pd.read_csv(file)
# # # # # #                 else:
# # # # # #                     df = pd.read_excel(file)

# # # # # #                 # Normalize column headers
# # # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # # #                 print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # # #                 # Infer schema with precision
# # # # # #                 schema = [
# # # # # #                     {
# # # # # #                         "column_name": col,
# # # # # #                         "data_type": infer_column_dtype(df[col])
# # # # # #                     }
# # # # # #                     for col in df.columns
# # # # # #                 ]
# # # # # #                 print(f"[DEBUG] Inferred schema: {schema}")  # Debugging statement

# # # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # # #                 for col in boolean_columns:
# # # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # # #                 print(f"[DEBUG] Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # # #                 # Handle Duplicate Files Dynamically
# # # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # # #                 if existing_file:
# # # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # # #                     file.name = new_file_name
# # # # # #                     print(f"[DEBUG] Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # # #                 else:
# # # # # #                     print(f"[DEBUG] File name is unique: {file.name}")  # Debugging statement

# # # # # #                 # Save Metadata to Database
# # # # # #                 file.seek(0)
# # # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # # #                 if file_serializer.is_valid():
# # # # # #                     file_instance = file_serializer.save()

# # # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # # #                     csv_buffer = BytesIO()
# # # # # #                     df.to_csv(csv_buffer, index=False)
# # # # # #                     csv_buffer.seek(0)
# # # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # # #                     # Upload to AWS S3
# # # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # # # #                     print(f"[DEBUG] File uploaded to S3: {file_key}")  # Debugging statement

# # # # # #                     # Generate file URL
# # # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # # #                     file_instance.file_url = file_url
# # # # # #                     file_instance.save()

# # # # # #                     # Save Schema to Database
# # # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # # # #                     print(f"[DEBUG] Schema saved to database for file: {file.name}")  # Debugging statement

# # # # # #                     # Trigger AWS Glue Table Update
# # # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # # #                     # Append file info to response
# # # # # #                     uploaded_files_info.append({
# # # # # #                         'id': file_instance.id,
# # # # # #                         'name': file_instance.name,
# # # # # #                         'file_url': file_instance.file_url,
# # # # # #                         'schema': schema,
# # # # # #                         'suggestions': {
# # # # # #                             'target_column': suggest_target_column(df, []),
# # # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
# # # # # #                         }
# # # # # #                     })

# # # # # #                 else:
# # # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # # #             # Store schema for user
# # # # # #             user_schemas[user_id] = uploaded_files_info
# # # # # #             print(f"[DEBUG] Stored schema for user: {user_id}")

# # # # # #             # Initiate schema discussion with the user
# # # # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # # # #             print(f"[DEBUG] Schema discussion message: {schema_discussion}")  # Debugging statement

# # # # # #             # Create or get user's conversation chain
# # # # # #             if user_id not in user_conversations:
# # # # # #                 conversation_chain = ConversationChain(
# # # # # #                     llm=llm_chatgpt,
# # # # # #                     prompt=prompt_chatgpt,
# # # # # #                     input_key="user_input",
# # # # # #                     memory=ConversationBufferMemory()
# # # # # #                 )
# # # # # #                 user_conversations[user_id] = conversation_chain
# # # # # #             else:
# # # # # #                 conversation_chain = user_conversations[user_id]

# # # # # #             # Add the schema discussion to the assistant's messages
# # # # # #             conversation_chain.memory.chat_memory.messages.append(
# # # # # #                 AIMessage(content=schema_discussion)
# # # # # #             )

# # # # # #             return Response({
# # # # # #                 "message": "Files uploaded and processed successfully.",
# # # # # #                 "uploaded_files": uploaded_files_info,
# # # # # #                 "chat_message": schema_discussion
# # # # # #             }, status=status.HTTP_201_CREATED)

# # # # # #         except pd.errors.EmptyDataError:
# # # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # # #         except NoCredentialsError:
# # # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # # #         except ClientError as e:
# # # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # # #         except Exception as e:
# # # # # #             print(f"[ERROR] Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # # #     def handle_chat(self, request):
# # # # # #         user_input = request.data.get("message", "").strip()
# # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # #         print(f"[DEBUG] Handling chat for user: {user_id}, message: {user_input}")

# # # # # #         if not user_input:
# # # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # # #         # Get or create conversation chain for the user
# # # # # #         if user_id not in user_conversations:
# # # # # #             conversation_chain = ConversationChain(
# # # # # #                 llm=llm_chatgpt,
# # # # # #                 prompt=prompt_chatgpt,
# # # # # #                 input_key="user_input",
# # # # # #                 memory=ConversationBufferMemory()
# # # # # #             )
# # # # # #             user_conversations[user_id] = conversation_chain
# # # # # #         else:
# # # # # #             conversation_chain = user_conversations[user_id]

# # # # # #         # Check if user is confirming schema
# # # # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # # # #             # Process user confirmation
# # # # # #             assistant_response = self.process_schema_confirmation(user_input, user_id)
# # # # # #             return Response({"response": assistant_response, "show_generate_notebook": True})

# # # # # #         # Generate assistant response
# # # # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # # # #         print(f"[DEBUG] Assistant response: {assistant_response}")  # Debugging statement

# # # # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # # # #             print(f"[DEBUG] GENERATE_NOTEBOOK_PROMPT detected for user: {user_id}")
# # # # # #             return Response({
# # # # # #                 "response": assistant_response,
# # # # # #                 "show_generate_notebook": True
# # # # # #             })

# # # # # #         return Response({
# # # # # #             "response": assistant_response
# # # # # #         })

# # # # # #     def process_schema_confirmation(self, user_input, user_id):
# # # # # #         """
# # # # # #         Processes user confirmation or adjustment of the schema.
# # # # # #         """
# # # # # #         uploaded_file_info = user_schemas[user_id][0]
# # # # # #         suggestions = uploaded_file_info['suggestions']

# # # # # #         # Assume user confirms or provides adjustments
# # # # # #         if 'yes' in user_input.lower():
# # # # # #             user_confirmations[user_id] = suggestions
# # # # # #             # Provide confirmed details and prompt to generate notebook
# # # # # #             assistant_response = self.format_confirmation_message(suggestions)
# # # # # #             return assistant_response
# # # # # #         else:
# # # # # #             # Parse user adjustments
# # # # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # # # #             if adjusted_columns:
# # # # # #                 user_confirmations[user_id] = adjusted_columns
# # # # # #                 # Provide confirmed details and prompt to generate notebook
# # # # # #                 assistant_response = self.format_confirmation_message(adjusted_columns)
# # # # # #                 return assistant_response
# # # # # #             else:
# # # # # #                 return "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."

# # # # # #     def format_confirmation_message(self, confirmation):
# # # # # #         """
# # # # # #         Formats the confirmation message with confirmed details and includes 'GENERATE_NOTEBOOK_PROMPT'.
# # # # # #         """
# # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # #         target_column = confirmation['target_column']
# # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # #         confirmation_text = (
# # # # # #             f"Great! You've confirmed the following details:\n\n"
# # # # # #             f"Entity ID Column: {entity_id_column}\n"
# # # # # #             f"Target Column: {target_column}\n"
# # # # # #             f"Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # #             "You can now generate the notebook to proceed with your analysis."
# # # # # #             "\n\nGENERATE_NOTEBOOK_PROMPT"
# # # # # #         )
# # # # # #         return confirmation_text

# # # # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # # # #         """
# # # # # #         Parses user input for schema adjustments.
# # # # # #         """
# # # # # #         import re

# # # # # #         # Normalize the input
# # # # # #         user_input = user_input.lower()

# # # # # #         # Patterns to match possible ways the user might specify the columns
# # # # # #         entity_id_patterns = [
# # # # # #             r"entity\s*[:\-]?\s*(\w+)",
# # # # # #             r"entity id\s*[:\-]?\s*(\w+)",
# # # # # #             r"entity_id\s*[:\-]?\s*(\w+)",
# # # # # #             r"entity column\s*[:\-]?\s*(\w+)",
# # # # # #             r"entityid\s*[:\-]?\s*(\w+)",
# # # # # #             r"id\s*[:\-]?\s*(\w+)"
# # # # # #         ]

# # # # # #         target_column_patterns = [
# # # # # #             r"target\s*[:\-]?\s*(\w+)",
# # # # # #             r"target column\s*[:\-]?\s*(\w+)",
# # # # # #             r"predict\s*[:\-]?\s*(\w+)",
# # # # # #             r"prediction\s*[:\-]?\s*(\w+)",
# # # # # #             r"target is\s+(\w+)"
# # # # # #         ]

# # # # # #         entity_id_column = None
# # # # # #         target_column = None

# # # # # #         for pattern in entity_id_patterns:
# # # # # #             match = re.search(pattern, user_input)
# # # # # #             if match:
# # # # # #                 entity_id_column = match.group(1)
# # # # # #                 break

# # # # # #         for pattern in target_column_patterns:
# # # # # #             match = re.search(pattern, user_input)
# # # # # #             if match:
# # # # # #                 target_column = match.group(1)
# # # # # #                 break

# # # # # #         # Fallback to suggestions if not found
# # # # # #         suggestions = uploaded_file_info['suggestions']
# # # # # #         if not entity_id_column:
# # # # # #             entity_id_column = suggestions['entity_id_column']
# # # # # #         if not target_column:
# # # # # #             target_column = suggestions['target_column']

# # # # # #         # Check if the columns exist in the schema
# # # # # #         schema_columns = [col['column_name'] for col in uploaded_file_info['schema']]
# # # # # #         if entity_id_column not in schema_columns or target_column not in schema_columns:
# # # # # #             return None

# # # # # #         # Prepare feature columns
# # # # # #         feature_columns = [col for col in schema_columns if col not in [entity_id_column, target_column]]

# # # # # #         return {
# # # # # #             'entity_id_column': entity_id_column,
# # # # # #             'target_column': target_column,
# # # # # #             'feature_columns': [{'column_name': col} for col in feature_columns]
# # # # # #         }

# # # # # #     def reset_conversation(self, request):
# # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # #         # Remove user's conversation chain
# # # # # #         if user_id in user_conversations:
# # # # # #             del user_conversations[user_id]
# # # # # #         # Remove user's uploaded schema and confirmations
# # # # # #         if user_id in user_schemas:
# # # # # #             del user_schemas[user_id]
# # # # # #         if user_id in user_confirmations:
# # # # # #             del user_confirmations[user_id]
# # # # # #         if user_id in user_notebook_flags:
# # # # # #             del user_notebook_flags[user_id]
# # # # # #         if user_id in user_notebooks:
# # # # # #             del user_notebooks[user_id]
# # # # # #         print(f"[DEBUG] Conversation reset for user: {user_id}")  # Debugging statement
# # # # # #         return Response({"message": "Conversation reset successful."})

# # # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # # #         """
# # # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # # #         """
# # # # # #         schema = uploaded_file['schema']
# # # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # # # #         schema_text = (
# # # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # # #             "Please confirm:\n"
# # # # # #             "- Is the Target Column correct?\n"
# # # # # #             "- Is the Entity ID Column correct?\n"
# # # # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # # # #         )
# # # # # #         return schema_text

# # # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # # #         """
# # # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # # #         """
# # # # # #         glue = get_glue_client()
# # # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # # #         storage_descriptor = {
# # # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # # #             'Location': s3_location,
# # # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # # #             'SerdeInfo': {
# # # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # # #                 'Parameters': {
# # # # # #                     'field.delim': ',',
# # # # # #                     'skip.header.line.count': '1'
# # # # # #                 }
# # # # # #             }
# # # # # #         }
# # # # # #         try:
# # # # # #             glue.update_table(
# # # # # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # # # # #                 TableInput={
# # # # # #                     'Name': table_name,
# # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # #                 }
# # # # # #             )
# # # # # #             print(f"[DEBUG] Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # # # #         except glue.exceptions.EntityNotFoundException:
# # # # # #             print(f"[DEBUG] Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # # # #             glue.create_table(
# # # # # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # # # # #                 TableInput={
# # # # # #                     'Name': table_name,
# # # # # #                     'StorageDescriptor': storage_descriptor,
# # # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # # #                 }
# # # # # #             )
# # # # # #             print(f"[DEBUG] Glue table '{table_name}' created successfully.")  # Debugging statement
# # # # # #         except Exception as e:
# # # # # #             print(f"[ERROR] Glue operation failed: {str(e)}")  # Debugging statement

# # # # # #     def generate_notebook(self, request):
# # # # # #         """
# # # # # #         Generates notebooks with pre-filled SQL queries and executed results.
# # # # # #         """
# # # # # #         user_id = request.data.get("user_id", "default_user")
# # # # # #         print(f"[DEBUG] Generating notebook for user: {user_id}")

# # # # # #         if user_id not in user_confirmations:
# # # # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # #         confirmation = user_confirmations[user_id]
# # # # # #         entity_id_column = confirmation['entity_id_column']
# # # # # #         target_column = confirmation['target_column']
# # # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # # #         # Get the table name from the uploaded file info
# # # # # #         if user_id in user_schemas:
# # # # # #             uploaded_file_info = user_schemas[user_id][0]
# # # # # #             table_name = os.path.splitext(uploaded_file_info['name'])[0].lower().replace(' ', '_')
# # # # # #         else:
# # # # # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # # # # #         # Create notebooks with SQL queries and executed results
# # # # # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name)
# # # # # #         notebook_features = self.create_features_notebook(feature_columns, table_name)

# # # # # #         # Store notebooks in user_notebooks dictionary
# # # # # #         import nbformat
# # # # # #         user_notebooks[user_id] = {
# # # # # #             'entity_target_notebook': nbformat.writes(notebook_entity_target),
# # # # # #             'features_notebook': nbformat.writes(notebook_features)
# # # # # #         }

# # # # # #         print("[DEBUG] Notebooks generated and stored successfully.")  # Debugging statement

# # # # # #         return Response({
# # # # # #             "message": "Notebooks generated successfully.",
# # # # # #             "notebooks": user_notebooks[user_id]
# # # # # #         }, status=status.HTTP_200_OK)

# # # # # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name):
# # # # # #         """
# # # # # #         Creates a notebook for Entity ID and Target analysis with SQL queries and executed results.
# # # # # #         """
# # # # # #         import nbformat
# # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # #         nb = new_notebook()
# # # # # #         cells = []

# # # # # #         # Introduction cell
# # # # # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # # # # #         # SQL query cell
# # # # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM {table_name} LIMIT 100;"

# # # # # #         # Add the SQL query to the cell
# # # # # #         cells.append(new_code_cell(f"-- SQL Query\n{sql_query_entity_target}"))

# # # # # #         # Execute the query and get results
# # # # # #         df_result = execute_sql_query(sql_query_entity_target)

# # # # # #         # Add the results to the notebook as a pandas DataFrame display
# # # # # #         if not df_result.empty:
# # # # # #             result_markdown = df_result.to_markdown(index=False)
# # # # # #             cells.append(new_markdown_cell(f"**Result:**\n\n{result_markdown}"))
# # # # # #         else:
# # # # # #             cells.append(new_markdown_cell("**Result:**\n\nNo data returned or an error occurred during query execution."))

# # # # # #         nb['cells'] = cells

# # # # # #         return nb

# # # # # #     def create_features_notebook(self, feature_columns, table_name):
# # # # # #         """
# # # # # #         Creates a notebook for Features analysis with SQL queries and executed results.
# # # # # #         """
# # # # # #         import nbformat
# # # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

# # # # # #         nb = new_notebook()
# # # # # #         cells = []

# # # # # #         # Introduction cell
# # # # # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # # # # #         # Feature columns analysis
# # # # # #         for feature in feature_columns:
# # # # # #             # Add a markdown cell for each feature
# # # # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))

# # # # # #             # SQL query cell
# # # # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) as count FROM {table_name} GROUP BY {feature} ORDER BY count DESC LIMIT 100;"

# # # # # #             cells.append(new_code_cell(f"-- SQL Query\n{sql_query_feature}"))

# # # # # #             # Execute the query and get results
# # # # # #             df_result = execute_sql_query(sql_query_feature)

# # # # # #             # Add the results to the notebook as a pandas DataFrame display
# # # # # #             if not df_result.empty:
# # # # # #                 result_markdown = df_result.to_markdown(index=False)
# # # # # #                 cells.append(new_markdown_cell(f"**Result:**\n\n{result_markdown}"))
# # # # # #             else:
# # # # # #                 cells.append(new_markdown_cell("**Result:**\n\nNo data returned or an error occurred during query execution."))

# # # # # #         nb['cells'] = cells

# # # # # #         return nb




# # # # # # chat/views.py

# # # # # import os
# # # # # import datetime
# # # # # from io import BytesIO
# # # # # from typing import Any, Dict, List
# # # # # import boto3
# # # # # import pandas as pd
# # # # # import openai
# # # # # import json
# # # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # # from django.conf import settings
# # # # # from rest_framework import status
# # # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # # from rest_framework.response import Response
# # # # # from rest_framework.views import APIView
# # # # # from langchain.chains import ConversationChain
# # # # # from langchain.chat_models import ChatOpenAI
# # # # # from langchain.prompts import PromptTemplate
# # # # # from langchain.memory import ConversationBufferMemory
# # # # # from langchain.schema import AIMessage
# # # # # from .models import FileSchema, UploadedFile
# # # # # from .serializers import UploadedFileSerializer

# # # # # # ===========================
# # # # # # AWS Configuration
# # # # # # ===========================
# # # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # e.g., 's3://your-athena-query-results-bucket/'
# # # # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # # # Set the Athena database (schema) name
# # # # # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Replace with your actual Athena database name

# # # # # # ===========================
# # # # # # OpenAI Configuration
# # # # # # ===========================
# # # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # # openai.api_key = OPENAI_API_KEY

# # # # # # ===========================
# # # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # # ===========================
# # # # # llm_chatgpt = ChatOpenAI(
# # # # #     model="gpt-3.5-turbo-16k",
# # # # #     temperature=0.7,
# # # # #     openai_api_key=OPENAI_API_KEY,
# # # # # )

# # # # # # Global dictionaries to store user-specific data
# # # # # user_conversations = {}
# # # # # user_schemas = {}
# # # # # user_confirmations = {}
# # # # # user_notebook_flags = {}
# # # # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # # # Modify the prompt in the existing code
# # # # # prompt_chatgpt = PromptTemplate(
# # # # #     input_variables=["history", "user_input"],
# # # # #     template=(
# # # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # # #         "Steps:\n"
# # # # #         "1. Discuss the Subject they want to predict.\n"
# # # # #         "2. Confirm the Target Value they want to predict.\n"
# # # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # # #         "4. Reference the dataset schema if available.\n"
# # # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # # #         "Conversation history: {history}\n"
# # # # #         "User input: {user_input}\n"
# # # # #         "Assistant:"
# # # # #     ),
# # # # # )

# # # # # # ===========================
# # # # # # Utility Functions
# # # # # # ===========================

# # # # # def get_s3_client():
# # # # #     """
# # # # #     Creates and returns an AWS S3 client.
# # # # #     """
# # # # #     return boto3.client(
# # # # #         's3',
# # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # #         region_name=AWS_S3_REGION_NAME
# # # # #     )

# # # # # def get_glue_client():
# # # # #     """
# # # # #     Creates and returns an AWS Glue client.
# # # # #     """
# # # # #     return boto3.client(
# # # # #         'glue',
# # # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # #         region_name=AWS_S3_REGION_NAME
# # # # #     )

# # # # # def infer_column_dtype(series: pd.Series) -> str:
# # # # #     """
# # # # #     Infers the correct data type for a column by handling mixed types.
# # # # #     """
# # # # #     series = series.dropna().astype(str).str.strip()

# # # # #     # Try datetime
# # # # #     try:
# # # # #         pd.to_datetime(series, errors='raise')
# # # # #         return "timestamp"
# # # # #     except ValueError:
# # # # #         pass

# # # # #     # Try boolean
# # # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # # #     unique_values = set(series.str.lower().unique())
# # # # #     if unique_values.issubset(boolean_values):
# # # # #         return "boolean"

# # # # #     # Try integer
# # # # #     try:
# # # # #         int_series = pd.to_numeric(series, errors='raise')
# # # # #         if (int_series % 1 == 0).all():
# # # # #             int_min = int_series.min()
# # # # #             int_max = int_series.max()
# # # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # # #                 return "int"
# # # # #             else:
# # # # #                 return "bigint"
# # # # #     except ValueError:
# # # # #         pass

# # # # #     # Try double
# # # # #     try:
# # # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # # #         return "double"
# # # # #     except ValueError:
# # # # #         pass

# # # # #     # Default to string
# # # # #     return "string"

# # # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # # #     """
# # # # #     Suggests a target column based on user input or predictive question.
# # # # #     """
# # # # #     # Use the last column as a default suggestion
# # # # #     return df.columns[-1]

# # # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # # #     """
# # # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # # #     """
# # # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # # #     for col in likely_id_columns:
# # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # #             return col

# # # # #     # Fallback: Find any column with >95% unique values
# # # # #     for col in df.columns:
# # # # #         if df[col].nunique() / len(df) > 0.95:
# # # # #             return col
# # # # #     return None

# # # # # def execute_sql_query(query: str) -> pd.DataFrame:
# # # # #     """
# # # # #     Executes a SQL query using AWS Athena and returns the results as a Pandas DataFrame.
# # # # #     """
# # # # #     from pyathena import connect
# # # # #     try:
# # # # #         conn = connect(
# # # # #             aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # # #             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # # #             s3_staging_dir=AWS_ATHENA_S3_STAGING_DIR,
# # # # #             region_name=AWS_REGION_NAME,
# # # # #             schema_name=ATHENA_SCHEMA_NAME  # Specify the Athena database (schema)
# # # # #         )
# # # # #         df = pd.read_sql(query, conn)
# # # # #         print(f"[DEBUG] Query executed successfully: {query}")
# # # # #         return df
# # # # #     except Exception as e:
# # # # #         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
# # # # #         return pd.DataFrame()  # Return an empty DataFrame on error

# # # # # # ===========================
# # # # # # Unified ChatGPT API
# # # # # # ===========================
# # # # # class UnifiedChatGPTAPI(APIView):
# # # # #     """
# # # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # # #     Endpoint: /api/chatgpt/
# # # # #     """
# # # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # # #     def post(self, request):
# # # # #         """
# # # # #         Handles POST requests for chat messages and file uploads.
# # # # #         Differentiates based on the presence of files in the request.
# # # # #         """
# # # # #         action = request.data.get('action', '')
# # # # #         if action == 'reset':
# # # # #             return self.reset_conversation(request)
# # # # #         if action == 'generate_notebook':
# # # # #             return self.generate_notebook(request)
# # # # #         if "file" in request.FILES:
# # # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # # #         # Else, handle chat message
# # # # #         return self.handle_chat(request)

# # # # #     def handle_file_upload(self, request, files: List[Any]):
# # # # #         """
# # # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # # #         After processing, appends schema details to the chat messages.
# # # # #         """
# # # # #         files = request.FILES.getlist("file")
# # # # #         if not files:
# # # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # # #         user_id = request.data.get("user_id", "default_user")
# # # # #         print(f"[DEBUG] Handling file upload for user: {user_id}")

# # # # #         try:
# # # # #             uploaded_files_info = []
# # # # #             s3 = get_s3_client()
# # # # #             glue = get_glue_client()

# # # # #             for file in files:
# # # # #                 print(f"[DEBUG] Processing file: {file.name}")
# # # # #                 # Validate file format
# # # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # # #                 # Read file into Pandas DataFrame
# # # # #                 if file.name.lower().endswith('.csv'):
# # # # #                     df = pd.read_csv(file)
# # # # #                 else:
# # # # #                     df = pd.read_excel(file)

# # # # #                 # Normalize column headers
# # # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # # #                 print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # # #                 # Infer schema with precision
# # # # #                 schema = [
# # # # #                     {
# # # # #                         "column_name": col,
# # # # #                         "data_type": infer_column_dtype(df[col])
# # # # #                     }
# # # # #                     for col in df.columns
# # # # #                 ]
# # # # #                 print(f"[DEBUG] Inferred schema: {schema}")  # Debugging statement

# # # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # # #                 for col in boolean_columns:
# # # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # # #                 print(f"[DEBUG] Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # # #                 # Handle Duplicate Files Dynamically
# # # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # # #                 if existing_file:
# # # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # # #                     file.name = new_file_name
# # # # #                     print(f"[DEBUG] Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # # #                 else:
# # # # #                     print(f"[DEBUG] File name is unique: {file.name}")  # Debugging statement

# # # # #                 # Save Metadata to Database
# # # # #                 file.seek(0)
# # # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # # #                 if file_serializer.is_valid():
# # # # #                     file_instance = file_serializer.save()

# # # # #                     # Convert DataFrame to CSV and Upload to S3
# # # # #                     csv_buffer = BytesIO()
# # # # #                     df.to_csv(csv_buffer, index=False)
# # # # #                     csv_buffer.seek(0)
# # # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # # #                     file_key = f"uploads/{s3_file_name}"

# # # # #                     # Upload to AWS S3
# # # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # # #                     print(f"[DEBUG] File uploaded to S3: {file_key}")  # Debugging statement

# # # # #                     # Generate file URL
# # # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # # #                     file_instance.file_url = file_url
# # # # #                     file_instance.save()

# # # # #                     # Save Schema to Database
# # # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # # #                     print(f"[DEBUG] Schema saved to database for file: {file.name}")  # Debugging statement

# # # # #                     # Trigger AWS Glue Table Update
# # # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # # #                     # Append file info to response
# # # # #                     uploaded_files_info.append({
# # # # #                         'id': file_instance.id,
# # # # #                         'name': file_instance.name,
# # # # #                         'file_url': file_instance.file_url,
# # # # #                         'schema': schema,
# # # # #                         'suggestions': {
# # # # #                             'target_column': suggest_target_column(df, []),
# # # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
# # # # #                         }
# # # # #                     })

# # # # #                 else:
# # # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # # #             # Store schema for user
# # # # #             user_schemas[user_id] = uploaded_files_info
# # # # #             print(f"[DEBUG] Stored schema for user: {user_id}")

# # # # #             # Initiate schema discussion with the user
# # # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # # #             print(f"[DEBUG] Schema discussion message: {schema_discussion}")  # Debugging statement

# # # # #             # Create or get user's conversation chain
# # # # #             if user_id not in user_conversations:
# # # # #                 conversation_chain = ConversationChain(
# # # # #                     llm=llm_chatgpt,
# # # # #                     prompt=prompt_chatgpt,
# # # # #                     input_key="user_input",
# # # # #                     memory=ConversationBufferMemory()
# # # # #                 )
# # # # #                 user_conversations[user_id] = conversation_chain
# # # # #             else:
# # # # #                 conversation_chain = user_conversations[user_id]

# # # # #             # Add the schema discussion to the assistant's messages
# # # # #             conversation_chain.memory.chat_memory.messages.append(
# # # # #                 AIMessage(content=schema_discussion)
# # # # #             )

# # # # #             return Response({
# # # # #                 "message": "Files uploaded and processed successfully.",
# # # # #                 "uploaded_files": uploaded_files_info,
# # # # #                 "chat_message": schema_discussion
# # # # #             }, status=status.HTTP_201_CREATED)

# # # # #         except pd.errors.EmptyDataError:
# # # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # # #         except NoCredentialsError:
# # # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # # #         except ClientError as e:
# # # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # # #         except Exception as e:
# # # # #             print(f"[ERROR] Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # # #     def handle_chat(self, request):
# # # # #         user_input = request.data.get("message", "").strip()
# # # # #         user_id = request.data.get("user_id", "default_user")
# # # # #         print(f"[DEBUG] Handling chat for user: {user_id}, message: {user_input}")

# # # # #         if not user_input:
# # # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # # #         # Get or create conversation chain for the user
# # # # #         if user_id not in user_conversations:
# # # # #             conversation_chain = ConversationChain(
# # # # #                 llm=llm_chatgpt,
# # # # #                 prompt=prompt_chatgpt,
# # # # #                 input_key="user_input",
# # # # #                 memory=ConversationBufferMemory()
# # # # #             )
# # # # #             user_conversations[user_id] = conversation_chain
# # # # #         else:
# # # # #             conversation_chain = user_conversations[user_id]

# # # # #         # Check if user is confirming schema
# # # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # # #             # Process user confirmation
# # # # #             assistant_response = self.process_schema_confirmation(user_input, user_id)
# # # # #             return Response({"response": assistant_response, "show_generate_notebook": True})

# # # # #         # Generate assistant response
# # # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # # #         print(f"[DEBUG] Assistant response: {assistant_response}")  # Debugging statement

# # # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # # #             print(f"[DEBUG] GENERATE_NOTEBOOK_PROMPT detected for user: {user_id}")
# # # # #             return Response({
# # # # #                 "response": assistant_response,
# # # # #                 "show_generate_notebook": True
# # # # #             })

# # # # #         return Response({
# # # # #             "response": assistant_response
# # # # #         })

# # # # #     def process_schema_confirmation(self, user_input, user_id):
# # # # #         """
# # # # #         Processes user confirmation or adjustment of the schema.
# # # # #         """
# # # # #         uploaded_file_info = user_schemas[user_id][0]
# # # # #         suggestions = uploaded_file_info['suggestions']

# # # # #         # Assume user confirms or provides adjustments
# # # # #         if 'yes' in user_input.lower():
# # # # #             user_confirmations[user_id] = suggestions
# # # # #             # Provide confirmed details and prompt to generate notebook
# # # # #             assistant_response = self.format_confirmation_message(suggestions)
# # # # #             return assistant_response
# # # # #         else:
# # # # #             # Parse user adjustments
# # # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # # #             if adjusted_columns:
# # # # #                 user_confirmations[user_id] = adjusted_columns
# # # # #                 # Provide confirmed details and prompt to generate notebook
# # # # #                 assistant_response = self.format_confirmation_message(adjusted_columns)
# # # # #                 return assistant_response
# # # # #             else:
# # # # #                 return "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."

# # # # #     def format_confirmation_message(self, confirmation):
# # # # #         """
# # # # #         Formats the confirmation message with confirmed details and includes 'GENERATE_NOTEBOOK_PROMPT'.
# # # # #         """
# # # # #         entity_id_column = confirmation['entity_id_column']
# # # # #         target_column = confirmation['target_column']
# # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # #         confirmation_text = (
# # # # #             f"Great! You've confirmed the following details:\n\n"
# # # # #             f"Entity ID Column: {entity_id_column}\n"
# # # # #             f"Target Column: {target_column}\n"
# # # # #             f"Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # #             "You can now generate the notebook to proceed with your analysis."
# # # # #             "\n\nGENERATE_NOTEBOOK_PROMPT"
# # # # #         )
# # # # #         return confirmation_text

# # # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # # #         """
# # # # #         Parses user input for schema adjustments.
# # # # #         """
# # # # #         import re

# # # # #         # Normalize the input
# # # # #         user_input = user_input.lower()

# # # # #         # Patterns to match possible ways the user might specify the columns
# # # # #         entity_id_patterns = [
# # # # #             r"entity\s*[:\-]?\s*(\w+)",
# # # # #             r"entity id\s*[:\-]?\s*(\w+)",
# # # # #             r"entity_id\s*[:\-]?\s*(\w+)",
# # # # #             r"entity column\s*[:\-]?\s*(\w+)",
# # # # #             r"entityid\s*[:\-]?\s*(\w+)",
# # # # #             r"id\s*[:\-]?\s*(\w+)"
# # # # #         ]

# # # # #         target_column_patterns = [
# # # # #             r"target\s*[:\-]?\s*(\w+)",
# # # # #             r"target column\s*[:\-]?\s*(\w+)",
# # # # #             r"predict\s*[:\-]?\s*(\w+)",
# # # # #             r"prediction\s*[:\-]?\s*(\w+)",
# # # # #             r"target is\s+(\w+)"
# # # # #         ]

# # # # #         entity_id_column = None
# # # # #         target_column = None

# # # # #         for pattern in entity_id_patterns:
# # # # #             match = re.search(pattern, user_input)
# # # # #             if match:
# # # # #                 entity_id_column = match.group(1)
# # # # #                 break

# # # # #         for pattern in target_column_patterns:
# # # # #             match = re.search(pattern, user_input)
# # # # #             if match:
# # # # #                 target_column = match.group(1)
# # # # #                 break

# # # # #         # Fallback to suggestions if not found
# # # # #         suggestions = uploaded_file_info['suggestions']
# # # # #         if not entity_id_column:
# # # # #             entity_id_column = suggestions['entity_id_column']
# # # # #         if not target_column:
# # # # #             target_column = suggestions['target_column']

# # # # #         # Check if the columns exist in the schema
# # # # #         schema_columns = [col['column_name'] for col in uploaded_file_info['schema']]
# # # # #         if entity_id_column not in schema_columns or target_column not in schema_columns:
# # # # #             return None

# # # # #         # Prepare feature columns
# # # # #         feature_columns = [col for col in schema_columns if col not in [entity_id_column, target_column]]

# # # # #         return {
# # # # #             'entity_id_column': entity_id_column,
# # # # #             'target_column': target_column,
# # # # #             'feature_columns': [{'column_name': col} for col in feature_columns]
# # # # #         }

# # # # #     def reset_conversation(self, request):
# # # # #         user_id = request.data.get("user_id", "default_user")
# # # # #         # Remove user's conversation chain
# # # # #         if user_id in user_conversations:
# # # # #             del user_conversations[user_id]
# # # # #         # Remove user's uploaded schema and confirmations
# # # # #         if user_id in user_schemas:
# # # # #             del user_schemas[user_id]
# # # # #         if user_id in user_confirmations:
# # # # #             del user_confirmations[user_id]
# # # # #         if user_id in user_notebook_flags:
# # # # #             del user_notebook_flags[user_id]
# # # # #         if user_id in user_notebooks:
# # # # #             del user_notebooks[user_id]
# # # # #         print(f"[DEBUG] Conversation reset for user: {user_id}")  # Debugging statement
# # # # #         return Response({"message": "Conversation reset successful."})

# # # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # # #         """
# # # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # # #         """
# # # # #         schema = uploaded_file['schema']
# # # # #         target_column = uploaded_file['suggestions']['target_column']
# # # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # # #         schema_text = (
# # # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # # #             "Please confirm:\n"
# # # # #             "- Is the Target Column correct?\n"
# # # # #             "- Is the Entity ID Column correct?\n"
# # # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # # #         )
# # # # #         return schema_text

# # # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # # #         """
# # # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # # #         """
# # # # #         glue = get_glue_client()
# # # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # # #         storage_descriptor = {
# # # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # # #             'Location': s3_location,
# # # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # # #             'SerdeInfo': {
# # # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # # #                 'Parameters': {
# # # # #                     'field.delim': ',',
# # # # #                     'skip.header.line.count': '1'
# # # # #                 }
# # # # #             }
# # # # #         }
# # # # #         try:
# # # # #             glue.update_table(
# # # # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # # # #                 TableInput={
# # # # #                     'Name': table_name,
# # # # #                     'StorageDescriptor': storage_descriptor,
# # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # #                 }
# # # # #             )
# # # # #             print(f"[DEBUG] Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # # #         except glue.exceptions.EntityNotFoundException:
# # # # #             print(f"[DEBUG] Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # # #             glue.create_table(
# # # # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # # # #                 TableInput={
# # # # #                     'Name': table_name,
# # # # #                     'StorageDescriptor': storage_descriptor,
# # # # #                     'TableType': 'EXTERNAL_TABLE'
# # # # #                 }
# # # # #             )
# # # # #             print(f"[DEBUG] Glue table '{table_name}' created successfully.")  # Debugging statement
# # # # #         except Exception as e:
# # # # #             print(f"[ERROR] Glue operation failed: {str(e)}")  # Debugging statement

# # # # #     def generate_notebook(self, request):
# # # # #         """
# # # # #         Generates notebooks with pre-filled SQL queries and executed results.
# # # # #         """
# # # # #         user_id = request.data.get("user_id", "default_user")
# # # # #         print(f"[DEBUG] Generating notebook for user: {user_id}")

# # # # #         if user_id not in user_confirmations:
# # # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # # #         confirmation = user_confirmations[user_id]
# # # # #         entity_id_column = confirmation['entity_id_column']
# # # # #         target_column = confirmation['target_column']
# # # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # # #         # Get the table name from the uploaded file info
# # # # #         if user_id in user_schemas:
# # # # #             uploaded_file_info = user_schemas[user_id][0]
# # # # #             table_name = os.path.splitext(uploaded_file_info['name'])[0].lower().replace(' ', '_')
# # # # #         else:
# # # # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # # # #         # Create notebooks with SQL queries and executed results
# # # # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name)
# # # # #         notebook_features = self.create_features_notebook(feature_columns, table_name)

# # # # #         # Store notebooks in user_notebooks dictionary
# # # # #         import nbformat
# # # # #         user_notebooks[user_id] = {
# # # # #             'entity_target_notebook': nbformat.writes(notebook_entity_target),
# # # # #             'features_notebook': nbformat.writes(notebook_features)
# # # # #         }

# # # # #         print("[DEBUG] Notebooks generated and stored successfully.")  # Debugging statement

# # # # #         return Response({
# # # # #             "message": "Notebooks generated successfully.",
# # # # #             "notebooks": user_notebooks[user_id]
# # # # #         }, status=status.HTTP_200_OK)

# # # # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name):
# # # # #         """
# # # # #         Creates a notebook for Entity ID and Target analysis with SQL queries and executed results.
# # # # #         """
# # # # #         import nbformat
# # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

# # # # #         nb = new_notebook()
# # # # #         cells = []

# # # # #         # Introduction cell
# # # # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # # # #         # SQL query cell
# # # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM {table_name} LIMIT 100;"

# # # # #         # Execute the query and get results
# # # # #         df_result = execute_sql_query(sql_query_entity_target)
# # # # #         result_json = df_result.to_dict(orient='records') if not df_result.empty else []

# # # # #         # Add the SQL query to the cell
# # # # #         code_cell = new_code_cell(sql_query_entity_target)

# # # # #         # Attach the result to the code cell's outputs
# # # # #         code_cell.outputs = [
# # # # #             {
# # # # #                 'output_type': 'execute_result',
# # # # #                 'execution_count': None,
# # # # #                 'data': {
# # # # #                     'application/json': result_json
# # # # #                 },
# # # # #                 'metadata': {}
# # # # #             }
# # # # #         ]

# # # # #         cells.append(code_cell)

# # # # #         nb['cells'] = cells

# # # # #         return nb

# # # # #     def create_features_notebook(self, feature_columns, table_name):
# # # # #         """
# # # # #         Creates a notebook for Features analysis with SQL queries and executed results.
# # # # #         """
# # # # #         import nbformat
# # # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

# # # # #         nb = new_notebook()
# # # # #         cells = []

# # # # #         # Introduction cell
# # # # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # # # #         # Feature columns analysis
# # # # #         for feature in feature_columns:
# # # # #             # Add a markdown cell for each feature
# # # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))

# # # # #             # SQL query cell
# # # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) as count FROM {table_name} GROUP BY {feature} ORDER BY count DESC LIMIT 100;"

# # # # #             # Execute the query and get results
# # # # #             df_result = execute_sql_query(sql_query_feature)
# # # # #             result_json = df_result.to_dict(orient='records') if not df_result.empty else []

# # # # #             # Add the SQL query to the cell
# # # # #             code_cell = new_code_cell(sql_query_feature)

# # # # #             # Attach the result to the code cell's outputs
# # # # #             code_cell.outputs = [
# # # # #                 {
# # # # #                     'output_type': 'execute_result',
# # # # #                     'execution_count': None,
# # # # #                     'data': {
# # # # #                         'application/json': result_json
# # # # #                     },
# # # # #                     'metadata': {}
# # # # #                 }
# # # # #             ]

# # # # #             cells.append(code_cell)

# # # # #         nb['cells'] = cells

# # # # #         return nb



# # # # # chat/views.py

# # # # import os
# # # # import datetime
# # # # from io import BytesIO
# # # # from typing import Any, Dict, List
# # # # import boto3
# # # # import pandas as pd
# # # # import openai
# # # # import json
# # # # from botocore.exceptions import ClientError, NoCredentialsError
# # # # from django.conf import settings
# # # # from rest_framework import status
# # # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # # from rest_framework.response import Response
# # # # from rest_framework.views import APIView
# # # # from langchain.chains import ConversationChain
# # # # from langchain.chat_models import ChatOpenAI
# # # # from langchain.prompts import PromptTemplate
# # # # from langchain.memory import ConversationBufferMemory
# # # # from langchain.schema import AIMessage
# # # # from .models import FileSchema, UploadedFile
# # # # from .serializers import UploadedFileSerializer

# # # # # ===========================
# # # # # AWS Configuration
# # # # # ===========================
# # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # e.g., 's3://your-athena-query-results-bucket/'
# # # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # # Set the Athena database (schema) name
# # # # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Replace with your actual Athena database name

# # # # # ===========================
# # # # # OpenAI Configuration
# # # # # ===========================
# # # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # # openai.api_key = OPENAI_API_KEY

# # # # # ===========================
# # # # # Initialize OpenAI LangChain model for ChatGPT
# # # # # ===========================
# # # # llm_chatgpt = ChatOpenAI(
# # # #     model="gpt-3.5-turbo-16k",
# # # #     temperature=0.7,
# # # #     openai_api_key=OPENAI_API_KEY,
# # # # )

# # # # # Global dictionaries to store user-specific data
# # # # user_conversations = {}
# # # # user_schemas = {}
# # # # user_confirmations = {}
# # # # user_notebook_flags = {}
# # # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # # Modify the prompt in the existing code
# # # # prompt_chatgpt = PromptTemplate(
# # # #     input_variables=["history", "user_input"],
# # # #     template=(
# # # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # # #         "Steps:\n"
# # # #         "1. Discuss the Subject they want to predict.\n"
# # # #         "2. Confirm the Target Value they want to predict.\n"
# # # #         "3. Check if there's a specific time frame for the prediction.\n"
# # # #         "4. Reference the dataset schema if available.\n"
# # # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # # #         "Conversation history: {history}\n"
# # # #         "User input: {user_input}\n"
# # # #         "Assistant:"
# # # #     ),
# # # # )

# # # # # ===========================
# # # # # Utility Functions
# # # # # ===========================

# # # # def get_s3_client():
# # # #     """
# # # #     Creates and returns an AWS S3 client.
# # # #     """
# # # #     return boto3.client(
# # # #         's3',
# # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #         region_name=AWS_S3_REGION_NAME
# # # #     )

# # # # def get_glue_client():
# # # #     """
# # # #     Creates and returns an AWS Glue client.
# # # #     """
# # # #     return boto3.client(
# # # #         'glue',
# # # #         aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #         region_name=AWS_S3_REGION_NAME
# # # #     )

# # # # def infer_column_dtype(series: pd.Series) -> str:
# # # #     """
# # # #     Infers the correct data type for a column by handling mixed types.
# # # #     """
# # # #     series = series.dropna().astype(str).str.strip()

# # # #     # Try datetime
# # # #     try:
# # # #         pd.to_datetime(series, errors='raise')
# # # #         return "timestamp"
# # # #     except ValueError:
# # # #         pass

# # # #     # Try boolean
# # # #     boolean_values = {'true', 'false', '1', '0', 'yes', 'no'}
# # # #     unique_values = set(series.str.lower().unique())
# # # #     if unique_values.issubset(boolean_values):
# # # #         return "boolean"

# # # #     # Try integer
# # # #     try:
# # # #         int_series = pd.to_numeric(series, errors='raise')
# # # #         if (int_series % 1 == 0).all():
# # # #             int_min = int_series.min()
# # # #             int_max = int_series.max()
# # # #             if int_min >= -2147483648 and int_max <= 2147483647:
# # # #                 return "int"
# # # #             else:
# # # #                 return "bigint"
# # # #     except ValueError:
# # # #         pass

# # # #     # Try double
# # # #     try:
# # # #         pd.to_numeric(series, errors='raise', downcast='float')
# # # #         return "double"
# # # #     except ValueError:
# # # #         pass

# # # #     # Default to string
# # # #     return "string"

# # # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # # #     """
# # # #     Suggests a target column based on user input or predictive question.
# # # #     """
# # # #     # Use the last column as a default suggestion
# # # #     return df.columns[-1]

# # # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # # #     """
# # # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # # #     """
# # # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # # #     for col in likely_id_columns:
# # # #         if df[col].nunique() / len(df) > 0.95:
# # # #             return col

# # # #     # Fallback: Find any column with >95% unique values
# # # #     for col in df.columns:
# # # #         if df[col].nunique() / len(df) > 0.95:
# # # #             return col
# # # #     return None

# # # # def execute_sql_query(query: str) -> pd.DataFrame:
# # # #     """
# # # #     Executes a SQL query using AWS Athena and returns the results as a Pandas DataFrame.
# # # #     """
# # # #     from pyathena import connect
# # # #     try:
# # # #         conn = connect(
# # # #             aws_access_key_id=AWS_ACCESS_KEY_ID,
# # # #             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
# # # #             s3_staging_dir=AWS_ATHENA_S3_STAGING_DIR,
# # # #             region_name=AWS_REGION_NAME,
# # # #             schema_name=ATHENA_SCHEMA_NAME  # Specify the Athena database (schema)
# # # #         )
# # # #         df = pd.read_sql(query, conn)
# # # #         print(f"[DEBUG] Query executed successfully: {query}")
# # # #         return df
# # # #     except Exception as e:
# # # #         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
# # # #         return pd.DataFrame()  # Return an empty DataFrame on error

# # # # # ===========================
# # # # # Unified ChatGPT API
# # # # # ===========================
# # # # class UnifiedChatGPTAPI(APIView):
# # # #     """
# # # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # # #     Endpoint: /api/chatgpt/
# # # #     """
# # # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # # #     def post(self, request):
# # # #         """
# # # #         Handles POST requests for chat messages and file uploads.
# # # #         Differentiates based on the presence of files in the request.
# # # #         """
# # # #         action = request.data.get('action', '')
# # # #         if action == 'reset':
# # # #             return self.reset_conversation(request)
# # # #         if action == 'generate_notebook':
# # # #             return self.generate_notebook(request)
# # # #         if "file" in request.FILES:
# # # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # # #         # Else, handle chat message
# # # #         return self.handle_chat(request)

# # # #     def handle_file_upload(self, request, files: List[Any]):
# # # #         """
# # # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # # #         After processing, appends schema details to the chat messages.
# # # #         """
# # # #         files = request.FILES.getlist("file")
# # # #         if not files:
# # # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # # #         user_id = request.data.get("user_id", "default_user")
# # # #         print(f"[DEBUG] Handling file upload for user: {user_id}")

# # # #         try:
# # # #             uploaded_files_info = []
# # # #             s3 = get_s3_client()
# # # #             glue = get_glue_client()

# # # #             for file in files:
# # # #                 print(f"[DEBUG] Processing file: {file.name}")
# # # #                 # Validate file format
# # # #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# # # #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# # # #                 # Read file into Pandas DataFrame
# # # #                 if file.name.lower().endswith('.csv'):
# # # #                     df = pd.read_csv(file)
# # # #                 else:
# # # #                     df = pd.read_excel(file)

# # # #                 # Normalize column headers
# # # #                 df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# # # #                 print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # # #                 # Infer schema with precision
# # # #                 schema = [
# # # #                     {
# # # #                         "column_name": col,
# # # #                         "data_type": infer_column_dtype(df[col])
# # # #                     }
# # # #                     for col in df.columns
# # # #                 ]
# # # #                 print(f"[DEBUG] Inferred schema: {schema}")  # Debugging statement

# # # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # # #                 for col in boolean_columns:
# # # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # # #                 print(f"[DEBUG] Boolean columns converted: {boolean_columns}")  # Debugging statement

# # # #                 # Handle Duplicate Files Dynamically
# # # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # # #                 if existing_file:
# # # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # # #                     file.name = new_file_name
# # # #                     print(f"[DEBUG] Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # # #                 else:
# # # #                     print(f"[DEBUG] File name is unique: {file.name}")  # Debugging statement

# # # #                 # Save Metadata to Database
# # # #                 file.seek(0)
# # # #                 file_serializer = UploadedFileSerializer(data={'name': file.name, 'file': file})
# # # #                 if file_serializer.is_valid():
# # # #                     file_instance = file_serializer.save()

# # # #                     # Convert DataFrame to CSV and Upload to S3
# # # #                     csv_buffer = BytesIO()
# # # #                     df.to_csv(csv_buffer, index=False)
# # # #                     csv_buffer.seek(0)
# # # #                     s3_file_name = os.path.splitext(file.name)[0] + '.csv'
# # # #                     file_key = f"uploads/{s3_file_name}"

# # # #                     # Upload to AWS S3
# # # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# # # #                     print(f"[DEBUG] File uploaded to S3: {file_key}")  # Debugging statement

# # # #                     # Generate file URL
# # # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # # #                     file_instance.file_url = file_url
# # # #                     file_instance.save()

# # # #                     # Save Schema to Database
# # # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # # #                     print(f"[DEBUG] Schema saved to database for file: {file.name}")  # Debugging statement

# # # #                     # Trigger AWS Glue Table Update
# # # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # # #                     # Append file info to response
# # # #                     uploaded_files_info.append({
# # # #                         'id': file_instance.id,
# # # #                         'name': file_instance.name,
# # # #                         'file_url': file_instance.file_url,
# # # #                         'schema': schema,
# # # #                         'suggestions': {
# # # #                             'target_column': suggest_target_column(df, []),
# # # #                             'entity_id_column': suggest_entity_id_column(df),
# # # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
# # # #                         }
# # # #                     })

# # # #                 else:
# # # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # # #             # Store schema for user
# # # #             user_schemas[user_id] = uploaded_files_info
# # # #             print(f"[DEBUG] Stored schema for user: {user_id}")

# # # #             # Initiate schema discussion with the user
# # # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # # #             print(f"[DEBUG] Schema discussion message: {schema_discussion}")  # Debugging statement

# # # #             # Create or get user's conversation chain
# # # #             if user_id not in user_conversations:
# # # #                 conversation_chain = ConversationChain(
# # # #                     llm=llm_chatgpt,
# # # #                     prompt=prompt_chatgpt,
# # # #                     input_key="user_input",
# # # #                     memory=ConversationBufferMemory()
# # # #                 )
# # # #                 user_conversations[user_id] = conversation_chain
# # # #             else:
# # # #                 conversation_chain = user_conversations[user_id]

# # # #             # Add the schema discussion to the assistant's messages
# # # #             conversation_chain.memory.chat_memory.messages.append(
# # # #                 AIMessage(content=schema_discussion)
# # # #             )

# # # #             return Response({
# # # #                 "message": "Files uploaded and processed successfully.",
# # # #                 "uploaded_files": uploaded_files_info,
# # # #                 "chat_message": schema_discussion
# # # #             }, status=status.HTTP_201_CREATED)

# # # #         except pd.errors.EmptyDataError:
# # # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # # #         except NoCredentialsError:
# # # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # # #         except ClientError as e:
# # # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # #         except Exception as e:
# # # #             print(f"[ERROR] Unexpected error during file upload: {str(e)}")  # Debugging statement
# # # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # # #     def handle_chat(self, request):
# # # #         user_input = request.data.get("message", "").strip()
# # # #         user_id = request.data.get("user_id", "default_user")
# # # #         print(f"[DEBUG] Handling chat for user: {user_id}, message: {user_input}")

# # # #         if not user_input:
# # # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # #         # Get or create conversation chain for the user
# # # #         if user_id not in user_conversations:
# # # #             conversation_chain = ConversationChain(
# # # #                 llm=llm_chatgpt,
# # # #                 prompt=prompt_chatgpt,
# # # #                 input_key="user_input",
# # # #                 memory=ConversationBufferMemory()
# # # #             )
# # # #             user_conversations[user_id] = conversation_chain
# # # #         else:
# # # #             conversation_chain = user_conversations[user_id]

# # # #         # Check if user is confirming schema
# # # #         if user_id in user_schemas and user_id not in user_confirmations:
# # # #             # Process user confirmation
# # # #             assistant_response = self.process_schema_confirmation(user_input, user_id)
# # # #             return Response({"response": assistant_response, "show_generate_notebook": True})

# # # #         # Generate assistant response
# # # #         assistant_response = conversation_chain.run(user_input=user_input)
# # # #         print(f"[DEBUG] Assistant response: {assistant_response}")  # Debugging statement

# # # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # # #             print(f"[DEBUG] GENERATE_NOTEBOOK_PROMPT detected for user: {user_id}")
# # # #             return Response({
# # # #                 "response": assistant_response,
# # # #                 "show_generate_notebook": True
# # # #             })

# # # #         return Response({
# # # #             "response": assistant_response
# # # #         })

# # # #     def process_schema_confirmation(self, user_input, user_id):
# # # #         """
# # # #         Processes user confirmation or adjustment of the schema.
# # # #         """
# # # #         uploaded_file_info = user_schemas[user_id][0]
# # # #         suggestions = uploaded_file_info['suggestions']

# # # #         # Assume user confirms or provides adjustments
# # # #         if 'yes' in user_input.lower():
# # # #             user_confirmations[user_id] = suggestions
# # # #             # Provide confirmed details and prompt to generate notebook
# # # #             assistant_response = self.format_confirmation_message(suggestions)
# # # #             return assistant_response
# # # #         else:
# # # #             # Parse user adjustments
# # # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # # #             if adjusted_columns:
# # # #                 user_confirmations[user_id] = adjusted_columns
# # # #                 # Provide confirmed details and prompt to generate notebook
# # # #                 assistant_response = self.format_confirmation_message(adjusted_columns)
# # # #                 return assistant_response
# # # #             else:
# # # #                 return "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."

# # # #     def format_confirmation_message(self, confirmation):
# # # #         """
# # # #         Formats the confirmation message with confirmed details and includes 'GENERATE_NOTEBOOK_PROMPT'.
# # # #         """
# # # #         entity_id_column = confirmation['entity_id_column']
# # # #         target_column = confirmation['target_column']
# # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # #         confirmation_text = (
# # # #             f"Great! You've confirmed the following details:\n\n"
# # # #             f"Entity ID Column: {entity_id_column}\n"
# # # #             f"Target Column: {target_column}\n"
# # # #             f"Feature Columns: {', '.join(feature_columns)}\n\n"
# # # #             "You can now generate the notebook to proceed with your analysis."
# # # #             "\n\nGENERATE_NOTEBOOK_PROMPT"
# # # #         )
# # # #         return confirmation_text

# # # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # # #         """
# # # #         Parses user input for schema adjustments.
# # # #         """
# # # #         import re

# # # #         # Normalize the input
# # # #         user_input = user_input.lower()

# # # #         # Patterns to match possible ways the user might specify the columns
# # # #         entity_id_patterns = [
# # # #             r"entity\s*[:\-]?\s*(\w+)",
# # # #             r"entity id\s*[:\-]?\s*(\w+)",
# # # #             r"entity_id\s*[:\-]?\s*(\w+)",
# # # #             r"entity column\s*[:\-]?\s*(\w+)",
# # # #             r"entityid\s*[:\-]?\s*(\w+)",
# # # #             r"id\s*[:\-]?\s*(\w+)"
# # # #         ]

# # # #         target_column_patterns = [
# # # #             r"target\s*[:\-]?\s*(\w+)",
# # # #             r"target column\s*[:\-]?\s*(\w+)",
# # # #             r"predict\s*[:\-]?\s*(\w+)",
# # # #             r"prediction\s*[:\-]?\s*(\w+)",
# # # #             r"target is\s+(\w+)"
# # # #         ]

# # # #         entity_id_column = None
# # # #         target_column = None

# # # #         for pattern in entity_id_patterns:
# # # #             match = re.search(pattern, user_input)
# # # #             if match:
# # # #                 entity_id_column = match.group(1)
# # # #                 break

# # # #         for pattern in target_column_patterns:
# # # #             match = re.search(pattern, user_input)
# # # #             if match:
# # # #                 target_column = match.group(1)
# # # #                 break

# # # #         # Fallback to suggestions if not found
# # # #         suggestions = uploaded_file_info['suggestions']
# # # #         if not entity_id_column:
# # # #             entity_id_column = suggestions['entity_id_column']
# # # #         if not target_column:
# # # #             target_column = suggestions['target_column']

# # # #         # Check if the columns exist in the schema
# # # #         schema_columns = [col['column_name'] for col in uploaded_file_info['schema']]
# # # #         if entity_id_column not in schema_columns or target_column not in schema_columns:
# # # #             return None

# # # #         # Prepare feature columns
# # # #         feature_columns = [col for col in schema_columns if col not in [entity_id_column, target_column]]

# # # #         return {
# # # #             'entity_id_column': entity_id_column,
# # # #             'target_column': target_column,
# # # #             'feature_columns': [{'column_name': col} for col in feature_columns]
# # # #         }

# # # #     def reset_conversation(self, request):
# # # #         user_id = request.data.get("user_id", "default_user")
# # # #         # Remove user's conversation chain
# # # #         if user_id in user_conversations:
# # # #             del user_conversations[user_id]
# # # #         # Remove user's uploaded schema and confirmations
# # # #         if user_id in user_schemas:
# # # #             del user_schemas[user_id]
# # # #         if user_id in user_confirmations:
# # # #             del user_confirmations[user_id]
# # # #         if user_id in user_notebook_flags:
# # # #             del user_notebook_flags[user_id]
# # # #         if user_id in user_notebooks:
# # # #             del user_notebooks[user_id]
# # # #         print(f"[DEBUG] Conversation reset for user: {user_id}")  # Debugging statement
# # # #         return Response({"message": "Conversation reset successful."})

# # # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # # #         """
# # # #         Formats the schema information to be appended as an assistant message in the chat.
# # # #         """
# # # #         schema = uploaded_file['schema']
# # # #         target_column = uploaded_file['suggestions']['target_column']
# # # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # # #         schema_text = (
# # # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # # #             "Please confirm:\n"
# # # #             "- Is the Target Column correct?\n"
# # # #             "- Is the Entity ID Column correct?\n"
# # # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # # #         )
# # # #         return schema_text

# # # #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str):
# # # #         """
# # # #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# # # #         """
# # # #         glue = get_glue_client()
# # # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# # # #         storage_descriptor = {
# # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # #             'Location': s3_location,
# # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # #             'SerdeInfo': {
# # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # #                 'Parameters': {
# # # #                     'field.delim': ',',
# # # #                     'skip.header.line.count': '1'
# # # #                 }
# # # #             }
# # # #         }
# # # #         try:
# # # #             glue.update_table(
# # # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # # #                 TableInput={
# # # #                     'Name': table_name,
# # # #                     'StorageDescriptor': storage_descriptor,
# # # #                     'TableType': 'EXTERNAL_TABLE'
# # # #                 }
# # # #             )
# # # #             print(f"[DEBUG] Glue table '{table_name}' updated successfully.")  # Debugging statement
# # # #         except glue.exceptions.EntityNotFoundException:
# # # #             print(f"[DEBUG] Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # # #             glue.create_table(
# # # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # # #                 TableInput={
# # # #                     'Name': table_name,
# # # #                     'StorageDescriptor': storage_descriptor,
# # # #                     'TableType': 'EXTERNAL_TABLE'
# # # #                 }
# # # #             )
# # # #             print(f"[DEBUG] Glue table '{table_name}' created successfully.")  # Debugging statement
# # # #         except Exception as e:
# # # #             print(f"[ERROR] Glue operation failed: {str(e)}")  # Debugging statement

# # # #     def generate_notebook(self, request):
# # # #         """
# # # #         Generates notebooks with pre-filled SQL queries and executed results.
# # # #         """
# # # #         user_id = request.data.get("user_id", "default_user")
# # # #         print(f"[DEBUG] Generating notebook for user: {user_id}")

# # # #         if user_id not in user_confirmations:
# # # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # # #         confirmation = user_confirmations[user_id]
# # # #         entity_id_column = confirmation['entity_id_column']
# # # #         target_column = confirmation['target_column']
# # # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # # #         # Get the table name from the uploaded file info
# # # #         if user_id in user_schemas:
# # # #             uploaded_file_info = user_schemas[user_id][0]
# # # #             table_name = os.path.splitext(uploaded_file_info['name'])[0].lower().replace(' ', '_')
# # # #         else:
# # # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # # #         # Create notebooks with SQL queries and executed results
# # # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name)
# # # #         notebook_features = self.create_features_notebook(feature_columns, table_name)

# # # #         # Store notebooks in user_notebooks dictionary
# # # #         import nbformat
# # # #         user_notebooks[user_id] = {
# # # #             'entity_target_notebook': nbformat.writes(notebook_entity_target),
# # # #             'features_notebook': nbformat.writes(notebook_features)
# # # #         }

# # # #         print("[DEBUG] Notebooks generated and stored successfully.")  # Debugging statement

# # # #         return Response({
# # # #             "message": "Notebooks generated successfully.",
# # # #             "notebooks": user_notebooks[user_id]
# # # #         }, status=status.HTTP_200_OK)

# # # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name):
# # # #         """
# # # #         Creates a notebook for Entity ID and Target analysis with SQL queries and executed results.
# # # #         """
# # # #         import nbformat
# # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

# # # #         nb = new_notebook()
# # # #         cells = []

# # # #         # Introduction cell
# # # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # # #         # SQL query cell
# # # #         sql_query_entity_target = f"SELECT {entity_id_column}, {target_column} FROM {table_name} LIMIT 100;"

# # # #         # Execute the query and get results
# # # #         df_result = execute_sql_query(sql_query_entity_target)
# # # #         result_json = df_result.to_dict(orient='records') if not df_result.empty else []

# # # #         # Add the SQL query to the cell
# # # #         code_cell = new_code_cell(sql_query_entity_target)

# # # #         # Attach the result to the code cell's outputs using new_output
# # # #         code_cell.outputs = [
# # # #             new_output(
# # # #                 output_type='execute_result',
# # # #                 data={
# # # #                     'application/json': result_json
# # # #                 },
# # # #                 metadata={}
# # # #             )
# # # #         ]

# # # #         cells.append(code_cell)

# # # #         nb['cells'] = cells

# # # #         return nb

# # # #     def create_features_notebook(self, feature_columns, table_name):
# # # #         """
# # # #         Creates a notebook for Features analysis with SQL queries and executed results.
# # # #         """
# # # #         import nbformat
# # # #         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

# # # #         nb = new_notebook()
# # # #         cells = []

# # # #         # Introduction cell
# # # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # # #         # Feature columns analysis
# # # #         for feature in feature_columns:
# # # #             # Add a markdown cell for each feature
# # # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))

# # # #             # SQL query cell
# # # #             sql_query_feature = f"SELECT {feature}, COUNT(*) as count FROM {table_name} GROUP BY {feature} ORDER BY count DESC LIMIT 100;"

# # # #             # Execute the query and get results
# # # #             df_result = execute_sql_query(sql_query_feature)
# # # #             result_json = df_result.to_dict(orient='records') if not df_result.empty else []

# # # #             # Add the SQL query to the cell
# # # #             code_cell = new_code_cell(sql_query_feature)

# # # #             # Attach the result to the code cell's outputs using new_output
# # # #             code_cell.outputs = [
# # # #                 new_output(
# # # #                     output_type='execute_result',
# # # #                     data={
# # # #                         'application/json': result_json
# # # #                     },
# # # #                     metadata={}
# # # #                 )
# # # #             ]

# # # #             cells.append(code_cell)

# # # #         nb['cells'] = cells

# # # #         return nb




# # # # chat/views.py

# # # import os
# # # import datetime
# # # from io import BytesIO
# # # from typing import Any, Dict, List
# # # import boto3
# # # import pandas as pd
# # # import openai
# # # import json
# # # from botocore.exceptions import ClientError, NoCredentialsError
# # # from django.conf import settings
# # # from rest_framework import status
# # # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # # from rest_framework.response import Response
# # # from rest_framework.views import APIView
# # # from langchain.chains import ConversationChain
# # # from langchain.chat_models import ChatOpenAI
# # # from langchain.prompts import PromptTemplate
# # # from langchain.memory import ConversationBufferMemory
# # # from langchain.schema import AIMessage
# # # from sqlalchemy.engine import create_engine
# # # from .models import FileSchema, UploadedFile
# # # from .serializers import UploadedFileSerializer
# # # import re
# # # import nbformat
# # # from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
# # # import numpy as np

# # # # ===========================
# # # # AWS Configuration
# # # # ===========================
# # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # Ensure it's set in settings.py
# # # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # # Set the Athena database (schema) name
# # # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Replace with your actual Athena database name

# # # # ===========================
# # # # OpenAI Configuration
# # # # ===========================
# # # OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# # # openai.api_key = OPENAI_API_KEY

# # # # ===========================
# # # # Initialize OpenAI LangChain model for ChatGPT
# # # # ===========================
# # # llm_chatgpt = ChatOpenAI(
# # #     model="gpt-3.5-turbo-16k",
# # #     temperature=0.7,
# # #     openai_api_key=OPENAI_API_KEY,
# # # )

# # # # Global dictionaries to store user-specific data
# # # user_conversations = {}
# # # user_schemas = {}
# # # user_confirmations = {}
# # # user_notebook_flags = {}
# # # user_notebooks = {}  # Stores generated notebooks for each user

# # # # Modify the prompt in the existing code
# # # prompt_chatgpt = PromptTemplate(
# # #     input_variables=["history", "user_input"],
# # #     template=(
# # #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# # #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# # #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# # #         "Steps:\n"
# # #         "1. Discuss the Subject they want to predict.\n"
# # #         "2. Confirm the Target Value they want to predict.\n"
# # #         "3. Check if there's a specific time frame for the prediction.\n"
# # #         "4. Reference the dataset schema if available.\n"
# # #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# # #         "Conversation history: {history}\n"
# # #         "User input: {user_input}\n"
# # #         "Assistant:"
# # #     ),
# # # )

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
# # #     series = series.dropna().astype(str).str.strip()

# # #     # Try datetime
# # #     try:
# # #         pd.to_datetime(series, errors='raise')
# # #         return "timestamp"
# # #     except ValueError:
# # #         pass

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

# # # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# # #     """
# # #     Suggests a target column based on user input or predictive question.
# # #     """
# # #     # Use the last column as a default suggestion
# # #     return df.columns[-1]

# # # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# # #     """
# # #     Suggests an entity ID column based on uniqueness and naming conventions.
# # #     """
# # #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# # #     for col in likely_id_columns:
# # #         if df[col].nunique() / len(df) > 0.95:
# # #             return col

# # #     # Fallback: Find any column with >95% unique values
# # #     for col in df.columns:
# # #         if df[col].nunique() / len(df) > 0.95:
# # #             return col
# # #     return None

# # # def execute_sql_query(query: str) -> pd.DataFrame:
# # #     """
# # #     Executes a SQL query using AWS Athena and returns the results as a Pandas DataFrame.
# # #     """
# # #     try:
# # #         if not AWS_ATHENA_S3_STAGING_DIR:
# # #             raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set. Please set it in your environment variables or settings.")
# # #         print(f"[DEBUG] AWS_ATHENA_S3_STAGING_DIR: {AWS_ATHENA_S3_STAGING_DIR}")

# # #         # Create an SQLAlchemy engine for Athena
# # #         connection_string = (
# # #             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}@athena.{AWS_REGION_NAME}.amazonaws.com:443/"
# # #             f"{ATHENA_SCHEMA_NAME}?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}"
# # #         )
# # #         engine = create_engine(connection_string)

# # #         print(f"[DEBUG] Executing query: {query}")
# # #         df = pd.read_sql_query(query, engine)
# # #         print(f"[DEBUG] Query executed successfully, retrieved {len(df)} rows.")
# # #         return df
# # #     except Exception as e:
# # #         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
# # #         return pd.DataFrame()  # Return an empty DataFrame on error

# # # # ===========================
# # # # Unified ChatGPT API
# # # # ===========================
# # # class UnifiedChatGPTAPI(APIView):
# # #     """
# # #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# # #     Endpoint: /api/chatgpt/
# # #     """
# # #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# # #     def post(self, request):
# # #         """
# # #         Handles POST requests for chat messages and file uploads.
# # #         Differentiates based on the presence of files in the request.
# # #         """
# # #         action = request.data.get('action', '')
# # #         if action == 'reset':
# # #             return self.reset_conversation(request)
# # #         if action == 'generate_notebook':
# # #             return self.generate_notebook(request)
# # #         if "file" in request.FILES:
# # #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# # #         # Else, handle chat message
# # #         return self.handle_chat(request)

# # #     def handle_file_upload(self, request, files: List[Any]):
# # #         """
# # #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# # #         After processing, appends schema details to the chat messages.
# # #         """
# # #         files = request.FILES.getlist("file")
# # #         if not files:
# # #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# # #         user_id = request.data.get("user_id", "default_user")
# # #         print(f"[DEBUG] Handling file upload for user: {user_id}")

# # #         try:
# # #             uploaded_files_info = []
# # #             s3 = get_s3_client()
# # #             glue = get_glue_client()

# # #             for file in files:
# # #                 print(f"[DEBUG] Processing file: {file.name}")
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
# # #                 print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")  # Debugging statement

# # #                 # Infer schema with precision
# # #                 schema = [
# # #                     {
# # #                         "column_name": col,
# # #                         "data_type": infer_column_dtype(df[col])
# # #                     }
# # #                     for col in df.columns
# # #                 ]
# # #                 print(f"[DEBUG] Inferred schema: {schema}")  # Debugging statement

# # #                 # Convert Boolean Columns to 'true'/'false' Strings
# # #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# # #                 for col in boolean_columns:
# # #                     df[col] = df[col].astype(str).str.strip().str.lower()
# # #                     df[col] = df[col].replace({'1': 'true', '0': 'false', 'yes': 'true', 'no': 'false'})
# # #                 print(f"[DEBUG] Boolean columns converted: {boolean_columns}")  # Debugging statement

# # #                 # Handle Duplicate Files Dynamically
# # #                 file_name_base, file_extension = os.path.splitext(file.name)
# # #                 file_name_base = file_name_base.lower().replace(' ', '_')

# # #                 existing_file = UploadedFile.objects.filter(name=file.name).first()
# # #                 if existing_file:
# # #                     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# # #                     new_file_name = f"{file_name_base}_{timestamp}{file_extension}"
# # #                     file.name = new_file_name
# # #                     print(f"[DEBUG] Duplicate file detected. Renaming file to: {new_file_name}")  # Debugging statement
# # #                 else:
# # #                     print(f"[DEBUG] File name is unique: {file.name}")  # Debugging statement

# # #                 # Save Metadata to Database
# # #                 file.seek(0)
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
# # #                     print(f"[DEBUG] File uploaded to S3: {file_key}")  # Debugging statement

# # #                     # Generate file URL
# # #                     file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# # #                     file_instance.file_url = file_url
# # #                     file_instance.save()

# # #                     # Save Schema to Database
# # #                     FileSchema.objects.create(file=file_instance, schema=schema)
# # #                     print(f"[DEBUG] Schema saved to database for file: {file.name}")  # Debugging statement

# # #                     # Trigger AWS Glue Table Update
# # #                     self.trigger_glue_update(file_name_base, schema, file_key)

# # #                     # Append file info to response
# # #                     uploaded_files_info.append({
# # #                         'id': file_instance.id,
# # #                         'name': file_instance.name,
# # #                         'file_url': file_instance.file_url,
# # #                         'schema': schema,
# # #                         'suggestions': {
# # #                             'target_column': suggest_target_column(df, []),
# # #                             'entity_id_column': suggest_entity_id_column(df),
# # #                             'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
# # #                         }
# # #                     })

# # #                 else:
# # #                     return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # #             # Store schema for user
# # #             user_schemas[user_id] = uploaded_files_info
# # #             print(f"[DEBUG] Stored schema for user: {user_id}")

# # #             # Initiate schema discussion with the user
# # #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# # #             print(f"[DEBUG] Schema discussion message: {schema_discussion}")  # Debugging statement

# # #             # Create or get user's conversation chain
# # #             if user_id not in user_conversations:
# # #                 conversation_chain = ConversationChain(
# # #                     llm=llm_chatgpt,
# # #                     prompt=prompt_chatgpt,
# # #                     input_key="user_input",
# # #                     memory=ConversationBufferMemory()
# # #                 )
# # #                 user_conversations[user_id] = conversation_chain
# # #             else:
# # #                 conversation_chain = user_conversations[user_id]

# # #             # Add the schema discussion to the assistant's messages
# # #             conversation_chain.memory.chat_memory.messages.append(
# # #                 AIMessage(content=schema_discussion)
# # #             )

# # #             return Response({
# # #                 "message": "Files uploaded and processed successfully.",
# # #                 "uploaded_files": uploaded_files_info,
# # #                 "chat_message": schema_discussion
# # #             }, status=status.HTTP_201_CREATED)

# # #         except pd.errors.EmptyDataError:
# # #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# # #         except NoCredentialsError:
# # #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# # #         except ClientError as e:
# # #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # #         except Exception as e:
# # #             print(f"[ERROR] Unexpected error during file upload: {str(e)}")  # Debugging statement
# # #             return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # #     def handle_chat(self, request):
# # #         user_input = request.data.get("message", "").strip()
# # #         user_id = request.data.get("user_id", "default_user")
# # #         print(f"[DEBUG] Handling chat for user: {user_id}, message: {user_input}")

# # #         if not user_input:
# # #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# # #         # Get or create conversation chain for the user
# # #         if user_id not in user_conversations:
# # #             conversation_chain = ConversationChain(
# # #                 llm=llm_chatgpt,
# # #                 prompt=prompt_chatgpt,
# # #                 input_key="user_input",
# # #                 memory=ConversationBufferMemory()
# # #             )
# # #             user_conversations[user_id] = conversation_chain
# # #         else:
# # #             conversation_chain = user_conversations[user_id]

# # #         # Check if user is confirming schema
# # #         if user_id in user_schemas and user_id not in user_confirmations:
# # #             # Process user confirmation
# # #             assistant_response = self.process_schema_confirmation(user_input, user_id)
# # #             return Response({"response": assistant_response, "show_generate_notebook": True})

# # #         # Generate assistant response
# # #         assistant_response = conversation_chain.run(user_input=user_input)
# # #         print(f"[DEBUG] Assistant response: {assistant_response}")  # Debugging statement

# # #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# # #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# # #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# # #             print(f"[DEBUG] GENERATE_NOTEBOOK_PROMPT detected for user: {user_id}")
# # #             return Response({
# # #                 "response": assistant_response,
# # #                 "show_generate_notebook": True
# # #             })

# # #         return Response({
# # #             "response": assistant_response
# # #         })

# # #     def process_schema_confirmation(self, user_input, user_id):
# # #         """
# # #         Processes user confirmation or adjustment of the schema.
# # #         """
# # #         uploaded_file_info = user_schemas[user_id][0]
# # #         suggestions = uploaded_file_info['suggestions']

# # #         # Assume user confirms or provides adjustments
# # #         if 'yes' in user_input.lower():
# # #             user_confirmations[user_id] = {
# # #                 'entity_id_column': suggestions['entity_id_column'],
# # #                 'target_column': suggestions['target_column'],
# # #                 'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']]
# # #             }
# # #             # Provide confirmed details and prompt to generate notebook
# # #             assistant_response = self.format_confirmation_message(user_confirmations[user_id])
# # #             return assistant_response
# # #         else:
# # #             # Parse user adjustments
# # #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# # #             if adjusted_columns:
# # #                 user_confirmations[user_id] = adjusted_columns
# # #                 # Provide confirmed details and prompt to generate notebook
# # #                 assistant_response = self.format_confirmation_message(adjusted_columns)
# # #                 return assistant_response
# # #             else:
# # #                 return "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."

# # #     def format_confirmation_message(self, confirmation):
# # #         """
# # #         Formats the confirmation message with confirmed details and includes 'GENERATE_NOTEBOOK_PROMPT'.
# # #         """
# # #         entity_id_column = confirmation['entity_id_column']
# # #         target_column = confirmation['target_column']
# # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # #         confirmation_text = (
# # #             f"Great! You've confirmed the following details:\n\n"
# # #             f"Entity ID Column: {entity_id_column}\n"
# # #             f"Target Column: {target_column}\n"
# # #             f"Feature Columns: {', '.join(feature_columns)}\n\n"
# # #             "You can now generate the notebook to proceed with your analysis."
# # #             "\n\nGENERATE_NOTEBOOK_PROMPT"
# # #         )
# # #         return confirmation_text

# # #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# # #         """
# # #         Parses user input for schema adjustments.
# # #         """
# # #         import re

# # #         # Normalize the input
# # #         user_input = user_input.lower()

# # #         # Patterns to match possible ways the user might specify the columns
# # #         entity_id_patterns = [
# # #             r"entity\s*[:\-]?\s*(\w+)",
# # #             r"entity id\s*[:\-]?\s*(\w+)",
# # #             r"entity_id\s*[:\-]?\s*(\w+)",
# # #             r"entity column\s*[:\-]?\s*(\w+)",
# # #             r"entityid\s*[:\-]?\s*(\w+)",
# # #             r"id\s*[:\-]?\s*(\w+)"
# # #         ]

# # #         target_column_patterns = [
# # #             r"target\s*[:\-]?\s*(\w+)",
# # #             r"target column\s*[:\-]?\s*(\w+)",
# # #             r"predict\s*[:\-]?\s*(\w+)",
# # #             r"prediction\s*[:\-]?\s*(\w+)",
# # #             r"target is\s+(\w+)"
# # #         ]

# # #         entity_id_column = None
# # #         target_column = None

# # #         for pattern in entity_id_patterns:
# # #             match = re.search(pattern, user_input)
# # #             if match:
# # #                 entity_id_column = match.group(1)
# # #                 break

# # #         for pattern in target_column_patterns:
# # #             match = re.search(pattern, user_input)
# # #             if match:
# # #                 target_column = match.group(1)
# # #                 break

# # #         # Fallback to suggestions if not found
# # #         suggestions = uploaded_file_info['suggestions']
# # #         if not entity_id_column:
# # #             entity_id_column = suggestions['entity_id_column']
# # #         if not target_column:
# # #             target_column = suggestions['target_column']

# # #         # Check if the columns exist in the schema
# # #         schema_columns = [col['column_name'] for col in uploaded_file_info['schema']]
# # #         if entity_id_column not in schema_columns or target_column not in schema_columns:
# # #             return None

# # #         # Prepare feature columns
# # #         feature_columns = [col for col in schema_columns if col not in [entity_id_column, target_column]]

# # #         return {
# # #             'entity_id_column': entity_id_column,
# # #             'target_column': target_column,
# # #             'feature_columns': [{'column_name': col} for col in feature_columns]
# # #         }

# # #     def reset_conversation(self, request):
# # #         user_id = request.data.get("user_id", "default_user")
# # #         # Remove user's conversation chain
# # #         if user_id in user_conversations:
# # #             del user_conversations[user_id]
# # #         # Remove user's uploaded schema and confirmations
# # #         if user_id in user_schemas:
# # #             del user_schemas[user_id]
# # #         if user_id in user_confirmations:
# # #             del user_confirmations[user_id]
# # #         if user_id in user_notebook_flags:
# # #             del user_notebook_flags[user_id]
# # #         if user_id in user_notebooks:
# # #             del user_notebooks[user_id]
# # #         print(f"[DEBUG] Conversation reset for user: {user_id}")  # Debugging statement
# # #         return Response({"message": "Conversation reset successful."})

# # #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# # #         """
# # #         Formats the schema information to be appended as an assistant message in the chat.
# # #         """
# # #         schema = uploaded_file['schema']
# # #         target_column = uploaded_file['suggestions']['target_column']
# # #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# # #         feature_columns = uploaded_file['suggestions']['feature_columns']
# # #         schema_text = (
# # #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# # #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# # #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# # #             f"Suggested Target Column: {target_column or 'None'}\n"
# # #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# # #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# # #             "Please confirm:\n"
# # #             "- Is the Target Column correct?\n"
# # #             "- Is the Entity ID Column correct?\n"
# # #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# # #         )
# # #         return schema_text

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
# # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # #                 TableInput={
# # #                     'Name': table_name,
# # #                     'StorageDescriptor': storage_descriptor,
# # #                     'TableType': 'EXTERNAL_TABLE'
# # #                 }
# # #             )
# # #             print(f"[DEBUG] Glue table '{table_name}' updated successfully.")  # Debugging statement
# # #         except glue.exceptions.EntityNotFoundException:
# # #             print(f"[DEBUG] Table '{table_name}' not found. Creating a new table...")  # Debugging statement
# # #             glue.create_table(
# # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # #                 TableInput={
# # #                     'Name': table_name,
# # #                     'StorageDescriptor': storage_descriptor,
# # #                     'TableType': 'EXTERNAL_TABLE'
# # #                 }
# # #             )
# # #             print(f"[DEBUG] Glue table '{table_name}' created successfully.")  # Debugging statement
# # #         except Exception as e:
# # #             print(f"[ERROR] Glue operation failed: {str(e)}")  # Debugging statement


# # #     def sanitize_identifier(name):
# # #         # Replace any character that is not a letter, number, or underscore with an underscore
# # #         return re.sub(r'\W+', '_', name.lower())

# # #     def validate_column_exists(column_name, columns_list):
# # #         if column_name not in columns_list:
# # #             print(f"[ERROR] Column '{column_name}' does not exist in the dataset.")
# # #             return False
# # #         return True

# # #     def sanitize_notebook(nb):
# # #         """Recursively sanitize the notebook object to replace NaN and Infinity values."""
# # #         import numpy as np
# # #         if isinstance(nb, dict):
# # #             return {k: sanitize_notebook(v) for k, v in nb.items()}
# # #         elif isinstance(nb, list):
# # #             return [sanitize_notebook(v) for v in nb]
# # #         elif isinstance(nb, float):
# # #             if np.isnan(nb) or np.isinf(nb):
# # #                 return None
# # #             else:
# # #                 return nb
# # #         else:
# # #             return nb

# # #     def generate_notebook(self, request):
# # #         """
# # #         Generates notebooks with pre-filled SQL queries and executed results.
# # #         """
# # #         user_id = request.data.get("user_id", "default_user")
# # #         print(f"[DEBUG] Generating notebook for user: {user_id}")

# # #         if user_id not in user_confirmations:
# # #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# # #         confirmation = user_confirmations[user_id]
# # #         entity_id_column = confirmation['entity_id_column']
# # #         target_column = confirmation['target_column']
# # #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# # #         # Get the table name from the uploaded file info
# # #         if user_id in user_schemas:
# # #             uploaded_file_info = user_schemas[user_id][0]
# # #             table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
# # #             table_name = sanitize_identifier(table_name_raw)
# # #         else:
# # #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# # #         # Get the list of columns from the schema
# # #         columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

# # #         # Validate entity_id_column and target_column
# # #         if not validate_column_exists(entity_id_column, columns_list):
# # #             return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

# # #         if not validate_column_exists(target_column, columns_list):
# # #             return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

# # #         # Ensure the table exists
# # #         if not wait_for_table_creation(table_name):
# # #             return Response({"error": f"Table '{table_name}' is not available."}, status=status.HTTP_400_BAD_REQUEST)

# # #         # Create notebooks with SQL queries and executed results
# # #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name, columns_list)
# # #         notebook_features = self.create_features_notebook(feature_columns, table_name, columns_list)

# # #         # Sanitize notebooks to replace NaN and Infinity values
# # #         notebook_entity_target_sanitized = sanitize_notebook(notebook_entity_target)
# # #         notebook_features_sanitized = sanitize_notebook(notebook_features)

# # #         # Store notebooks in user_notebooks dictionary
# # #         user_notebooks[user_id] = {
# # #             'entity_target_notebook': notebook_entity_target_sanitized,
# # #             'features_notebook': notebook_features_sanitized
# # #         }

# # #         print("[DEBUG] Notebooks generated and stored successfully.")  # Debugging statement

# # #         return Response({
# # #             "message": "Notebooks generated successfully.",
# # #             "notebooks": user_notebooks[user_id]
# # #         }, status=status.HTTP_200_OK)

# # #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name, columns_list):
# # #         """
# # #         Creates a notebook for Entity ID and Target analysis with SQL queries and executed results.
# # #         """
# # #         nb = new_notebook()
# # #         cells = []

# # #         # Introduction cell
# # #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# # #         # Sanitize columns
# # #         sanitized_entity_id_column = sanitize_identifier(entity_id_column)
# # #         sanitized_target_column = sanitize_identifier(target_column)

# # #         # SQL query cell
# # #         sql_query_entity_target = f"SELECT {sanitized_entity_id_column}, {sanitized_target_column} FROM {table_name} LIMIT 100;"

# # #         # Execute the query and get results
# # #         df_result = execute_sql_query(sql_query_entity_target)
# # #         if df_result.empty:
# # #             error_message = f"No data returned for query: {sql_query_entity_target}"
# # #             print(f"[ERROR] {error_message}")
# # #             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
# # #         else:
# # #             print(f"[DEBUG] DataFrame shape: {df_result.shape}")
# # #             print(f"[DEBUG] DataFrame head:\n{df_result.head()}")

# # #             # Replace NaN and Inf values with None
# # #             df_result = df_result.replace([np.nan, np.inf, -np.inf], None)

# # #             result_json = df_result.to_dict(orient='records')

# # #             # Add the SQL query to the cell
# # #             code_cell = new_code_cell(sql_query_entity_target)

# # #             # Attach the result to the code cell's outputs using new_output
# # #             code_cell.outputs = [
# # #                 new_output(
# # #                     output_type='execute_result',
# # #                     data={
# # #                         'application/json': result_json
# # #                     },
# # #                     metadata={},
# # #                     execution_count=None
# # #                 )
# # #             ]

# # #             cells.append(code_cell)

# # #         nb['cells'] = cells

# # #         return nb

# # #     def create_features_notebook(self, feature_columns, table_name, columns_list):
# # #         """
# # #         Creates a notebook for Features analysis with SQL queries and executed results.
# # #         """
# # #         nb = new_notebook()
# # #         cells = []

# # #         # Introduction cell
# # #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# # #         # Feature columns analysis
# # #         for feature in feature_columns:
# # #             # Validate feature column exists
# # #             if not validate_column_exists(feature, columns_list):
# # #                 cells.append(new_markdown_cell(f"**Error:** Feature column '{feature}' does not exist in the dataset."))
# # #                 continue

# # #             # Add a markdown cell for each feature
# # #             cells.append(new_markdown_cell(f"## Feature Column: {feature}"))

# # #             # Sanitize column name
# # #             sanitized_feature = sanitize_identifier(feature)

# # #             # SQL query cell
# # #             sql_query_feature = f"SELECT {sanitized_feature}, COUNT(*) as count FROM {table_name} GROUP BY {sanitized_feature} ORDER BY count DESC LIMIT 100;"

# # #             # Execute the query and get results
# # #             df_result = execute_sql_query(sql_query_feature)
# # #             if df_result.empty:
# # #                 error_message = f"No data returned for feature '{feature}'."
# # #                 print(f"[ERROR] {error_message}")
# # #                 cells.append(new_markdown_cell(f"**Error:** {error_message}"))
# # #                 continue
# # #             else:
# # #                 print(f"[DEBUG] Feature: {feature}, DataFrame shape: {df_result.shape}")
# # #                 print(f"[DEBUG] DataFrame head:\n{df_result.head()}")

# # #                 # Replace NaN and Inf values with None
# # #                 df_result = df_result.replace([np.nan, np.inf, -np.inf], None)

# # #                 result_json = df_result.to_dict(orient='records')

# # #                 # Add the SQL query to the cell
# # #                 code_cell = new_code_cell(sql_query_feature)

# # #                 # Attach the result to the code cell's outputs using new_output
# # #                 code_cell.outputs = [
# # #                     new_output(
# # #                         output_type='execute_result',
# # #                         data={
# # #                             'application/json': result_json
# # #                         },
# # #                         metadata={},
# # #                         execution_count=None
# # #                     )
# # #                 ]

# # #                 cells.append(code_cell)

# # #         nb['cells'] = cells

# # #         return nb

# # #     def wait_for_table_creation(table_name, timeout=60):
# # #         import time
# # #         glue_client = get_glue_client()
# # #         start_time = time.time()
# # #         while time.time() - start_time < timeout:
# # #             try:
# # #                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
# # #                 print(f"[DEBUG] Glue table '{table_name}' is now available.")
# # #                 return True
# # #             except glue_client.exceptions.EntityNotFoundException:
# # #                 time.sleep(5)
# # #             except Exception as e:
# # #                 print(f"[ERROR] Unexpected error while checking table availability: {str(e)}")
# # #                 return False
# # #             print(f"[ERROR] Glue table '{table_name}' did not become available within {timeout} seconds.")
# # #             return False




# # # chat/views.py

# # import os
# # import datetime
# # from io import BytesIO
# # from typing import Any, Dict, List
# # import uuid
# # import boto3
# # import pandas as pd
# # import openai
# # from django.db import transaction
# # from botocore.exceptions import ClientError, NoCredentialsError
# # from django.conf import settings
# # from rest_framework import status
# # from django.core.exceptions import ValidationError
# # from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# # from rest_framework.response import Response
# # from rest_framework.views import APIView
# # from langchain.chains import ConversationChain
# # from langchain.chat_models import ChatOpenAI
# # from langchain.prompts import PromptTemplate
# # from langchain.memory import ConversationBufferMemory
# # from langchain.schema import AIMessage
# # from sqlalchemy import Transaction
# # from sqlalchemy.engine import create_engine
# # from .models import FileSchema, UploadedFile
# # from .serializers import UploadedFileSerializer
# # import re
# # # import nbformat
# # from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
# # import numpy as np

# # # ===========================
# # # AWS Configuration
# # # ===========================
# # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')  # Ensure it's set in settings.py
# # AWS_REGION_NAME = AWS_S3_REGION_NAME  # Assuming it's the same as the S3 region

# # # Set the Athena database (schema) name
# # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Replace with your actual Athena database name

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

# # # Global dictionaries to store user-specific data
# # user_conversations = {}
# # user_schemas = {}
# # user_confirmations = {}
# # user_notebook_flags = {}
# # user_notebooks = {}  # Stores generated notebooks for each user

# # # Modify the prompt in the existing code
# # prompt_chatgpt = PromptTemplate(
# #     input_variables=["history", "user_input"],
# #     template=(
# #         "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
# #         "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
# #         "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
# #         "Steps:\n"
# #         "1. Discuss the Subject they want to predict.\n"
# #         "2. Confirm the Target Value they want to predict.\n"
# #         "3. Check if there's a specific time frame for the prediction.\n"
# #         "4. Reference the dataset schema if available.\n"
# #         "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
# #         "Conversation history: {history}\n"
# #         "User input: {user_input}\n"
# #         "Assistant:"
# #     ),
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
# #     series = series.dropna().astype(str).str.strip()

# #     # Try datetime with specified format
# #     date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]  # Add more formats as needed
# #     for date_format in date_formats:
# #         try:
# #             pd.to_datetime(series, format=date_format, errors='raise')
# #             return "timestamp"
# #         except ValueError:
# #             continue

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

# # def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
# #     """
# #     Suggests a target column based on user input or predictive question.
# #     """
# #     # Use the last column as a default suggestion
# #     return df.columns[-1]

# # def suggest_entity_id_column(df: pd.DataFrame) -> Any:
# #     """
# #     Suggests an entity ID column based on uniqueness and naming conventions.
# #     """
# #     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
# #     for col in likely_id_columns:
# #         if df[col].nunique() / len(df) > 0.95:
# #             return col

# #     # Fallback: Find any column with >95% unique values
# #     for col in df.columns:
# #         if df[col].nunique() / len(df) > 0.95:
# #             return col
# #     return None

# # def execute_sql_query(query: str) -> pd.DataFrame:
# #     """
# #     Executes a SQL query using AWS Athena and returns the results as a Pandas DataFrame.
# #     """
# #     try:
# #         if not AWS_ATHENA_S3_STAGING_DIR:
# #             raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set. Please set it in your environment variables or settings.")
# #         print(f"[DEBUG] AWS_ATHENA_S3_STAGING_DIR: {AWS_ATHENA_S3_STAGING_DIR}")

# #         # Create an SQLAlchemy engine for Athena
# #         connection_string = (
# #             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}@athena.{AWS_REGION_NAME}.amazonaws.com:443/"
# #             f"{ATHENA_SCHEMA_NAME}?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}"
# #         )
# #         engine = create_engine(connection_string)

# #         print(f"[DEBUG] Executing query: {query}")
# #         df = pd.read_sql_query(query, engine)
# #         print(f"[DEBUG] Query executed successfully, retrieved {len(df)} rows.")
# #         return df
# #     except Exception as e:
# #         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
# #         return pd.DataFrame()  # Return an empty DataFrame on error

# # # ===========================
# # # Unified ChatGPT API
# # # ===========================

# # class UnifiedChatGPTAPI(APIView):
# #     """
# #     Unified API for handling ChatGPT-based chat interactions and file uploads.
# #     Endpoint: /api/chatgpt/
# #     """
# #     parser_classes = [MultiPartParser, FormParser, JSONParser]

# #     def post(self, request):
# #         """
# #         Handles POST requests for chat messages and file uploads.
# #         Differentiates based on the presence of files in the request.
# #         """
# #         action = request.data.get('action', '')
# #         if action == 'reset':
# #             return self.reset_conversation(request)
# #         if action == 'generate_notebook':
# #             return self.generate_notebook(request)
# #         if "file" in request.FILES:
# #             return self.handle_file_upload(request, request.FILES.getlist("file"))

# #         # Else, handle chat message
# #         return self.handle_chat(request)

# #     def handle_file_upload(self, request, files: List[Any]):
# #         """
# #         Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
# #         After processing, appends schema details to the chat messages.
# #         """
# #         files = request.FILES.getlist("file")
# #         if not files:
# #             print("[ERROR] No files provided in the request.")
# #             return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)

# #         user_id = request.data.get("user_id", "default_user")
# #         print(f"[DEBUG] Handling file upload for user: {user_id}")

# #         try:
# #             uploaded_files_info = []
# #             s3 = get_s3_client()
# #             glue = get_glue_client()

# #             for file in files:
# #                 print(f"[DEBUG] Processing file: {file.name}")

# #                 # Validate file format
# #                 if not file.name.lower().endswith(('.csv', '.xlsx')):
# #                     print(f"[ERROR] Unsupported file format for file {file.name}. Only CSV and Excel are allowed.")
# #                     return Response({"error": f"Unsupported file format for file {file.name}. Only CSV and Excel are allowed."}, status=status.HTTP_400_BAD_REQUEST)

# #                 # Read file into Pandas DataFrame
# #                 try:
# #                     if file.name.lower().endswith('.csv'):
# #                         df = pd.read_csv(file, low_memory=False, encoding='utf-8', delimiter=',', na_values=['NA', 'N/A', ''])
# #                     else:
# #                         df = pd.read_excel(file, engine='openpyxl')
# #                     print(f"[DEBUG] File loaded successfully: {file.name}, Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# #                     if df.empty:
# #                         print(f"[ERROR] Uploaded file {file.name} is empty.")
# #                         return Response({"error": f"Uploaded file {file.name} is empty."}, status=status.HTTP_400_BAD_REQUEST)

# #                     if not df.columns.any():
# #                         print(f"[ERROR] Uploaded file {file.name} has no columns.")
# #                         return Response({"error": f"Uploaded file {file.name} has no columns."}, status=status.HTTP_400_BAD_REQUEST)
# #                 except pd.errors.ParserError as e:
# #                     print(f"[ERROR] CSV parsing error for file {file.name}: {e}")
# #                     return Response({"error": f"CSV parsing error for file {file.name}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
# #                 except Exception as e:
# #                     print(f"[ERROR] Error reading file {file.name}: {e}")
# #                     return Response({"error": f"Error reading file {file.name}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

# #                 # Normalize column headers
# #                 normalized_columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
# #                 if len(normalized_columns) != len(set(normalized_columns)):
# #                     print("[ERROR] Duplicate columns detected after normalization.")
# #                     return Response({"error": "Duplicate columns detected after normalization."}, status=status.HTTP_400_BAD_REQUEST)

# #                 if any(col == '' for col in normalized_columns):
# #                     print("[ERROR] Some columns have empty names after normalization.")
# #                     return Response({"error": "Some columns have empty names after normalization."}, status=status.HTTP_400_BAD_REQUEST)

# #                 df.columns = normalized_columns
# #                 print(f"[DEBUG] Normalized DataFrame columns: {df.columns.tolist()}")

# #                 # Infer schema with precision
# #                 schema = [
# #                     {
# #                         "column_name": col,
# #                         "data_type": infer_column_dtype(df[col])
# #                     }
# #                     for col in df.columns
# #                 ]
# #                 print(f"[DEBUG] Inferred schema: {schema}")

# #                 # Convert Boolean Columns to 'true'/'false' Strings
# #                 boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
# #                 replacement_dict = {
# #                     '1': 'true',
# #                     '0': 'false',
# #                     'yes': 'true',
# #                     'no': 'false',
# #                     't': 'true',
# #                     'f': 'false',
# #                     'y': 'true',
# #                     'n': 'false',
# #                     'true': 'true',
# #                     'false': 'false',
# #                 }
# #                 for col in boolean_columns:
# #                     df[col] = df[col].astype(str).str.strip().str.lower().replace(replacement_dict)

# #                     # Validate boolean conversion
# #                     unexpected_values = df[col].unique().tolist()
# #                     unexpected_values = [val for val in unexpected_values if val not in ['true', 'false']]
# #                     if unexpected_values:
# #                         print(f"[ERROR] Unexpected boolean values in column {col}: {unexpected_values}")
# #                         return Response({"error": f"Unexpected boolean values in column {col}: {unexpected_values}"}, status=status.HTTP_400_BAD_REQUEST)
# #                 print(f"[DEBUG] Boolean columns converted: {boolean_columns}")

# #                 # Handle Duplicate Files Dynamically with UUID
# #                 file_name_base, file_extension = os.path.splitext(file.name)
# #                 file_name_base = file_name_base.lower().replace(' ', '_')
# #                 unique_id = uuid.uuid4().hex[:8]
# #                 new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
# #                 s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
# #                 file_key = f"uploads/{unique_id}/{s3_file_name}"

# #                 print(f"[DEBUG] Uploading file as: {new_file_name} to S3 key: {file_key}")

# #                 # Save Metadata to Database within a transaction
# #                 try:
# #                     with transaction.atomic():
# #                         # Initialize serializer with original file
# #                         file_serializer = UploadedFileSerializer(data={'name': new_file_name, 'file': file})

# #                         # Debug: Check file content snippet
# #                         file.seek(0)
# #                         content_snippet = file.read(100).decode('utf-8', errors='ignore')
# #                         print(f"[DEBUG] File content snippet:\n{content_snippet}")
# #                         file.seek(0)  # Reset pointer after reading

# #                         if file_serializer.is_valid():
# #                             file_instance = file_serializer.save()

# #                             # Convert DataFrame to CSV and Upload to S3
# #                             csv_buffer = BytesIO()
# #                             df.to_csv(csv_buffer, index=False, encoding='utf-8')
# #                             csv_buffer.seek(0)

# #                             # Upload to AWS S3 with validation
# #                             try:
# #                                 s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
# #                                 # Verify upload
# #                                 s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
# #                                 print(f"[DEBUG] File uploaded to S3: {file_key}")
# #                             except ClientError as e:
# #                                 print(f"[ERROR] S3 upload failed for {file_key}: {e}")
# #                                 raise

# #                             # Generate file URL
# #                             file_url = f"https://{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_S3_REGION_NAME}.amazonaws.com/{file_key}"
# #                             file_instance.file_url = file_url
# #                             file_instance.save()

# #                             # Save Schema to Database
# #                             FileSchema.objects.create(file=file_instance, schema=schema)
# #                             print(f"[DEBUG] Schema saved to database for file: {new_file_name}")

# #                             # Calculate file size in megabytes
# #                             file_size_mb = file.size / (1024 * 1024)
# #                             print(f"[DEBUG] File size: {file_size_mb:.2f} MB")

# #                             # Trigger AWS Glue Table Update
# #                             self.trigger_glue_update(new_file_name, schema, file_key, file_size_mb)

# #                             # Append file info to response
# #                             uploaded_files_info.append({
# #                                 'id': file_instance.id,
# #                                 'name': file_instance.name,
# #                                 'file_url': file_instance.file_url,
# #                                 'schema': schema,
# #                                 'file_size_mb': file_size_mb,
# #                                 'suggestions': {
# #                                     'target_column': suggest_target_column(df, []),
# #                                     'entity_id_column': suggest_entity_id_column(df),
# #                                     'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
# #                                 }
# #                             })
# #                         else:
# #                             print(f"[ERROR] File serializer validation failed: {file_serializer.errors}")
# #                             return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# #                 except ClientError as e:
# #                     print(f"[ERROR] AWS ClientError during file upload: {e}")
# #                     return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# #                 except Exception as e:
# #                     print(f"[ERROR] Unexpected error during database operations: {e}")
# #                     return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# #             # Store schema for user
# #             user_schemas[user_id] = uploaded_files_info
# #             print(f"[DEBUG] Stored schema for user: {user_id}")

# #             # Initiate schema discussion with the user
# #             schema_discussion = self.format_schema_message(uploaded_files_info[0])
# #             print(f"[DEBUG] Schema discussion message: {schema_discussion}")

# #             # Create or get user's conversation chain
# #             if user_id not in user_conversations:
# #                 conversation_chain = ConversationChain(
# #                     llm=llm_chatgpt,
# #                     prompt=prompt_chatgpt,
# #                     input_key="user_input",
# #                     memory=ConversationBufferMemory()
# #                 )
# #                 user_conversations[user_id] = conversation_chain
# #             else:
# #                 conversation_chain = user_conversations[user_id]

# #             # Add the schema discussion to the assistant's messages
# #             conversation_chain.memory.chat_memory.messages.append(
# #                 AIMessage(content=schema_discussion)
# #             )

# #             return Response({
# #                 "message": "Files uploaded and processed successfully.",
# #                 "uploaded_files": uploaded_files_info,
# #                 "chat_message": schema_discussion
# #             }, status=status.HTTP_201_CREATED)

# #         except pd.errors.EmptyDataError:
# #             print("[ERROR] One of the files is empty or invalid.")
# #             return Response({'error': 'One of the files is empty or invalid.'}, status=status.HTTP_400_BAD_REQUEST)
# #         except NoCredentialsError:
# #             print("[ERROR] AWS credentials not available.")
# #             return Response({'error': 'AWS credentials not available.'}, status=status.HTTP_403_FORBIDDEN)
# #         except ClientError as e:
# #             print(f"[ERROR] AWS ClientError: {e}")
# #             return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# #         except ValidationError as e:
# #             print(f"[ERROR] Validation error: {e}")
# #             return Response({'error': f'Validation error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
# #         except Exception as e:
# #             print(f"[ERROR] Unexpected error during file upload: {str(e)}")
# #             return Response({'error': 'File processing failed due to an unexpected error.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



# #     def handle_chat(self, request):
# #         user_input = request.data.get("message", "").strip()
# #         user_id = request.data.get("user_id", "default_user")
# #         print(f"[DEBUG] Handling chat for user: {user_id}, message: {user_input}")

# #         if not user_input:
# #             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

# #         # Get or create conversation chain for the user
# #         if user_id not in user_conversations:
# #             conversation_chain = ConversationChain(
# #                 llm=llm_chatgpt,
# #                 prompt=prompt_chatgpt,
# #                 input_key="user_input",
# #                 memory=ConversationBufferMemory()
# #             )
# #             user_conversations[user_id] = conversation_chain
# #         else:
# #             conversation_chain = user_conversations[user_id]

# #         # Check if user is confirming schema
# #         if user_id in user_schemas and user_id not in user_confirmations:
# #             # Process user confirmation
# #             assistant_response = self.process_schema_confirmation(user_input, user_id)
# #             return Response({"response": assistant_response, "show_generate_notebook": True})

# #         # Generate assistant response
# #         assistant_response = conversation_chain.run(user_input=user_input)
# #         print(f"[DEBUG] Assistant response: {assistant_response}")  # Debugging statement

# #         # Check if assistant should prompt 'GENERATE_NOTEBOOK_PROMPT'
# #         if 'GENERATE_NOTEBOOK_PROMPT' in assistant_response:
# #             assistant_response = assistant_response.replace('GENERATE_NOTEBOOK_PROMPT', '').strip()
# #             print(f"[DEBUG] GENERATE_NOTEBOOK_PROMPT detected for user: {user_id}")
# #             return Response({
# #                 "response": assistant_response,
# #                 "show_generate_notebook": True
# #             })

# #         return Response({
# #             "response": assistant_response
# #         })

# #     def process_schema_confirmation(self, user_input, user_id):
# #         """
# #         Processes user confirmation or adjustment of the schema.
# #         """
# #         uploaded_file_info = user_schemas[user_id][0]
# #         suggestions = uploaded_file_info['suggestions']

# #         # Assume user confirms or provides adjustments
# #         if 'yes' in user_input.lower():
# #             user_confirmations[user_id] = {
# #                 'entity_id_column': suggestions['entity_id_column'],
# #                 'target_column': suggestions['target_column'],
# #                 'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']]
# #             }
# #             # Provide confirmed details and prompt to generate notebook
# #             assistant_response = self.format_confirmation_message(user_confirmations[user_id])
# #             return assistant_response
# #         else:
# #             # Parse user adjustments
# #             adjusted_columns = self.parse_user_adjustments(user_input, uploaded_file_info)
# #             if adjusted_columns:
# #                 user_confirmations[user_id] = adjusted_columns
# #                 # Provide confirmed details and prompt to generate notebook
# #                 assistant_response = self.format_confirmation_message(adjusted_columns)
# #                 return assistant_response
# #             else:
# #                 return "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."

# #     def format_confirmation_message(self, confirmation):
# #         """
# #         Formats the confirmation message with confirmed details and includes 'GENERATE_NOTEBOOK_PROMPT'.
# #         """
# #         entity_id_column = confirmation['entity_id_column']
# #         target_column = confirmation['target_column']
# #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# #         confirmation_text = (
# #             f"Great! You've confirmed the following details:\n\n"
# #             f"Entity ID Column: {entity_id_column}\n"
# #             f"Target Column: {target_column}\n"
# #             f"Feature Columns: {', '.join(feature_columns)}\n\n"
# #             "You can now generate the notebook to proceed with your analysis."
# #             # "\n\nGENERATE_NOTEBOOK_PROMPT"
# #         )
# #         return confirmation_text

# #     def parse_user_adjustments(self, user_input, uploaded_file_info):
# #         """
# #         Parses user input for schema adjustments.
# #         """
# #         import re

# #         # Normalize the input
# #         user_input = user_input.lower()

# #         # Patterns to match possible ways the user might specify the columns
# #         entity_id_patterns = [
# #             r"entity\s*[:\-]?\s*(\w+)",
# #             r"entity id\s*[:\-]?\s*(\w+)",
# #             r"entity_id\s*[:\-]?\s*(\w+)",
# #             r"entity column\s*[:\-]?\s*(\w+)",
# #             r"entityid\s*[:\-]?\s*(\w+)",
# #             r"id\s*[:\-]?\s*(\w+)"
# #         ]

# #         target_column_patterns = [
# #             r"target\s*[:\-]?\s*(\w+)",
# #             r"target column\s*[:\-]?\s*(\w+)",
# #             r"predict\s*[:\-]?\s*(\w+)",
# #             r"prediction\s*[:\-]?\s*(\w+)",
# #             r"target is\s+(\w+)"
# #         ]

# #         entity_id_column = None
# #         target_column = None

# #         for pattern in entity_id_patterns:
# #             match = re.search(pattern, user_input)
# #             if match:
# #                 entity_id_column = match.group(1)
# #                 break

# #         for pattern in target_column_patterns:
# #             match = re.search(pattern, user_input)
# #             if match:
# #                 target_column = match.group(1)
# #                 break

# #         # Fallback to suggestions if not found
# #         suggestions = uploaded_file_info['suggestions']
# #         if not entity_id_column:
# #             entity_id_column = suggestions['entity_id_column']
# #         if not target_column:
# #             target_column = suggestions['target_column']

# #         # Check if the columns exist in the schema
# #         schema_columns = [col['column_name'] for col in uploaded_file_info['schema']]
# #         if entity_id_column not in schema_columns or target_column not in schema_columns:
# #             return None

# #         # Prepare feature columns
# #         feature_columns = [col for col in schema_columns if col not in [entity_id_column, target_column]]

# #         return {
# #             'entity_id_column': entity_id_column,
# #             'target_column': target_column,
# #             'feature_columns': [{'column_name': col} for col in feature_columns]
# #         }

# #     def reset_conversation(self, request):
# #         user_id = request.data.get("user_id", "default_user")
# #         # Remove user's conversation chain
# #         if user_id in user_conversations:
# #             del user_conversations[user_id]
# #         # Remove user's uploaded schema and confirmations
# #         if user_id in user_schemas:
# #             del user_schemas[user_id]
# #         if user_id in user_confirmations:
# #             del user_confirmations[user_id]
# #         if user_id in user_notebook_flags:
# #             del user_notebook_flags[user_id]
# #         if user_id in user_notebooks:
# #             del user_notebooks[user_id]
# #         print(f"[DEBUG] Conversation reset for user: {user_id}")  # Debugging statement
# #         return Response({"message": "Conversation reset successful."})

# #     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
# #         """
# #         Formats the schema information to be appended as an assistant message in the chat.
# #         """
# #         schema = uploaded_file['schema']
# #         target_column = uploaded_file['suggestions']['target_column']
# #         entity_id_column = uploaded_file['suggestions']['entity_id_column']
# #         feature_columns = uploaded_file['suggestions']['feature_columns']
# #         schema_text = (
# #             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
# #             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
# #             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
# #             f"Suggested Target Column: {target_column or 'None'}\n"
# #             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
# #             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
# #             "Please confirm:\n"
# #             "- Is the Target Column correct?\n"
# #             "- Is the Entity ID Column correct?\n"
# #             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
# #         )
# #         return schema_text

# #     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
# #         """
# #         Dynamically updates or creates an AWS Glue table based on the uploaded file's schema.
# #         """
# #         glue = get_glue_client()
# #         # s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/"
# #         unique_id = file_key.split('/')[1]  # Assuming file_key is "uploads/{unique_id}/{s3_file_name}"
# #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/{unique_id}/"  # Ensure it matches the upload path
# #         table_name_without_extension = os.path.splitext(table_name)[0]  # Remove .csv
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
# #                 DatabaseName=ATHENA_SCHEMA_NAME,
# #                 TableInput={
# #                     'Name': table_name_without_extension,
# #                     'StorageDescriptor': storage_descriptor,
# #                     'TableType': 'EXTERNAL_TABLE'
# #                 }
# #             )
# #             print(f"[DEBUG] Glue table '{table_name_without_extension}' updated successfully.")  # Debugging statement
# #         except glue.exceptions.EntityNotFoundException:
# #             print(f"[DEBUG] Table '{table_name_without_extension}' not found. Creating a new table...")  # Debugging statement
# #             glue.create_table(
# #                 DatabaseName=ATHENA_SCHEMA_NAME,
# #                 TableInput={
# #                     'Name': table_name_without_extension,
# #                     'StorageDescriptor': storage_descriptor,
# #                     'TableType': 'EXTERNAL_TABLE'
# #                 }
# #             )
# #             print(f"[DEBUG] Glue table '{table_name_without_extension}' created successfully.")  # Debugging statement
# #         except Exception as e:
# #             print(f"[ERROR] Glue operation failed: {str(e)}")  # Debugging statement

# #         # Calculate dynamic timeout based on file size
# #         base_timeout = 80  # Base timeout in seconds
# #         additional_timeout_per_mb = 5  # Additional timeout per MB in seconds
# #         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
# #         print(f"[DEBUG] Dynamic timeout for table creation: {dynamic_timeout} seconds")

# #         # Wait for the table to be created
# #         self.wait_for_table_creation(table_name_without_extension, timeout=dynamic_timeout)

# #     def sanitize_identifier(self, name):
# #         """
# #         Replace any character that is not a letter, number, or underscore with an underscore.
# #         """
# #         return re.sub(r'\W+', '_', name.lower())

# #     def validate_column_exists(self, column_name, columns_list):
# #         """
# #         Checks if a column exists in the provided list of columns.
# #         """
# #         if column_name not in columns_list:
# #             print(f"[ERROR] Column '{column_name}' does not exist in the dataset.")
# #             return False
# #         return True

# #     def sanitize_notebook(self, nb):
# #         """
# #         Recursively sanitize the notebook object to replace NaN and Infinity values.
# #         """
# #         import numpy as np
# #         if isinstance(nb, dict):
# #             return {k: self.sanitize_notebook(v) for k, v in nb.items()}
# #         elif isinstance(nb, list):
# #             return [self.sanitize_notebook(v) for v in nb]
# #         elif isinstance(nb, float):
# #             if np.isnan(nb) or np.isinf(nb):
# #                 return None
# #             else:
# #                 return nb
# #         else:
# #             return nb

# #     def generate_notebook(self, request):
# #         """
# #         Generates notebooks with pre-filled SQL queries and executed results.
# #         """
# #         user_id = request.data.get("user_id", "default_user")
# #         print(f"[DEBUG] Generating notebook for user: {user_id}")

# #         if user_id not in user_confirmations:
# #             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

# #         confirmation = user_confirmations[user_id]
# #         entity_id_column = confirmation['entity_id_column']
# #         target_column = confirmation['target_column']
# #         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

# #         # Get the table name from the uploaded file info
# #         if user_id in user_schemas:
# #             uploaded_file_info = user_schemas[user_id][0]
# #             table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
# #             table_name = self.sanitize_identifier(table_name_raw)
# #         else:
# #             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

# #         # Get the list of columns from the schema
# #         columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

# #         # Validate entity_id_column and target_column
# #         if not self.validate_column_exists(entity_id_column, columns_list):
# #             return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

# #         if not self.validate_column_exists(target_column, columns_list):
# #             return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)
        
       

# #         # Create notebooks with SQL queries and executed results
# #         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, table_name, columns_list)
# #         notebook_features = self.create_features_notebook(feature_columns, table_name, columns_list)

# #         # Sanitize notebooks to replace NaN and Infinity values
# #         notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
# #         notebook_features_sanitized = self.sanitize_notebook(notebook_features)

# #         # Store notebooks in user_notebooks dictionary
# #         user_notebooks[user_id] = {
# #             'entity_target_notebook': notebook_entity_target_sanitized,
# #             'features_notebook': notebook_features_sanitized
# #         }

# #         print("[DEBUG] Notebooks generated and stored successfully.")  # Debugging statement

# #         return Response({
# #             "message": "Notebooks generated successfully.",
# #             "notebooks": user_notebooks[user_id]
# #         }, status=status.HTTP_200_OK)

# #     def create_entity_target_notebook(self, entity_id_column, target_column, table_name, columns_list):
# #         """
# #         Creates a notebook for Entity ID and Target analysis with SQL queries and executed results.
# #         """
# #         nb = new_notebook()
# #         cells = []

# #         # Introduction cell
# #         cells.append(new_markdown_cell("# Entity ID and Target Analysis Notebook"))

# #         # Sanitize columns
# #         sanitized_entity_id_column = self.sanitize_identifier(entity_id_column)
# #         sanitized_target_column = self.sanitize_identifier(target_column)

# #         # SQL query cell
# #         sql_query_entity_target = f"SELECT {sanitized_entity_id_column}, {sanitized_target_column} FROM {table_name} LIMIT 10;"

# #         # Execute the query and get results
# #         df_result = execute_sql_query(sql_query_entity_target)
# #         if df_result.empty:
# #             error_message = f"No data returned for query: {sql_query_entity_target}"
# #             print(f"[ERROR] {error_message}")
# #             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
# #         else:
# #             print(f"[DEBUG] DataFrame shape: {df_result.shape}")
# #             print(f"[DEBUG] DataFrame head:\n{df_result.head()}")

# #             # Replace NaN and Inf values with None
# #             df_result = df_result.replace([np.nan, np.inf, -np.inf], None)

# #             result_json = df_result.to_dict(orient='records')

# #             # Add the SQL query to the cell
# #             code_cell = new_code_cell(sql_query_entity_target)

# #             # Attach the result to the code cell's outputs using new_output
# #             code_cell.outputs = [
# #                 new_output(
# #                     output_type='execute_result',
# #                     data={
# #                         'application/json': result_json
# #                     },
# #                     metadata={},
# #                     execution_count=None
# #                 )
# #             ]

# #             cells.append(code_cell)

# #         nb['cells'] = cells

# #         return nb


# #     def create_features_notebook(self, feature_columns, table_name, columns_list):
# #         """
# #         Creates a notebook for Features analysis by querying all features in a single shell.
# #         """
# #         nb = new_notebook()
# #         cells = []

# #         # Introduction cell
# #         cells.append(new_markdown_cell("# Features Analysis Notebook"))

# #         # Sanitize feature column names
# #         sanitized_features = [self.sanitize_identifier(feature) for feature in feature_columns]

# #         # Validate that all columns exist
# #         missing_columns = [feature for feature in sanitized_features if feature not in columns_list]
# #         if missing_columns:
# #             error_message = f"The following feature columns do not exist in the dataset: {', '.join(missing_columns)}"
# #             print(f"[ERROR] {error_message}")
# #             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
# #             nb['cells'] = cells
# #             return nb

# #         # Create a single SQL query for all features with line-by-line formatting
# #         feature_query = (
# #             "SELECT\n    " + ",\n    ".join(sanitized_features) + f"\nFROM {table_name}\nLIMIT 10;"
# #         )
# #         print(f"[DEBUG] Executing SQL query for features:\n{feature_query}")

# #         # Execute the query and get results
# #         df_result = execute_sql_query(feature_query)
# #         if df_result.empty:
# #             error_message = f"No data returned for query: {feature_query}"
# #             print(f"[ERROR] {error_message}")
# #             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
# #         else:
# #             print(f"[DEBUG] DataFrame shape: {df_result.shape}")
# #             print(f"[DEBUG] DataFrame head:\n{df_result.head()}")

# #             # Replace NaN and Inf values with None
# #             df_result = df_result.replace([np.nan, np.inf, -np.inf], None)

# #             result_json = df_result.to_dict(orient='records')

# #             # Add the SQL query to the cell
# #             code_cell = new_code_cell(feature_query)

# #             # Attach the result to the code cell's outputs using new_output
# #             code_cell.outputs = [
# #                 new_output(
# #                     output_type='execute_result',
# #                     data={
# #                         'application/json': result_json
# #                     },
# #                     metadata={},
# #                     execution_count=None
# #                 )
# #             ]

# #             cells.append(code_cell)

# #         nb['cells'] = cells

# #         return nb



# #     def wait_for_table_creation(self, table_name, timeout):
# #         """
# #         Waits for the AWS Glue table to be created within the specified timeout.
# #         """
# #         import time
# #         glue_client = get_glue_client()
# #         start_time = time.time()
# #         while time.time() - start_time < timeout:
# #             try:
# #                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
# #                 print(f"[DEBUG] Glue table '{table_name}' is now available.")
# #                 return True
# #             except glue_client.exceptions.EntityNotFoundException:
# #                 time.sleep(20)
# #             except Exception as e:
# #                 print(f"[ERROR] Unexpected error while checking table availability: {str(e)}")
# #                 return False
# #         print(f"[ERROR] Glue table '{table_name}' did not become available within {timeout} seconds.")
# #         return False





# import os
# import datetime
# from io import BytesIO
# from typing import Any, Dict, List
# import uuid
# import boto3
# import pandas as pd
# import openai
# from django.db import transaction
# from botocore.exceptions import ClientError, NoCredentialsError
# from django.conf import settings
# import requests
# from rest_framework import status
# from django.core.exceptions import ValidationError
# from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from langchain.chains import ConversationChain
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import AIMessage
# from sqlalchemy import create_engine
# from .models import FileSchema, UploadedFile
# from .serializers import UploadedFileSerializer
# import re
# import nbformat
# from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
# import numpy as np

# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
# AWS_REGION_NAME = AWS_S3_REGION_NAME
# ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Adjust as needed

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY

# llm_chatgpt = ChatOpenAI(
#     model="gpt-3.5-turbo-16k",
#     temperature=0,
#     openai_api_key=OPENAI_API_KEY,
# )

# user_conversations = {}
# user_schemas = {}
# user_confirmations = {}
# user_notebook_flags = {}
# user_notebooks = {}

# prompt_chatgpt = PromptTemplate(
#     input_variables=["history", "user_input"],
#     template=(
#         "You are a highly intelligent and helpful AI assistant. Your "
#         "responses should be clear and concise. Assist the user in confirming "
#         "the dataset schema and any corrections they provide. Reflect the "
#         "confirmed or corrected schema back to the user before proceeding.\n\n"
#         "Steps:\n"
#         "1. Discuss the subject they want to predict.\n"
#         "2. Confirm the target value they want to predict.\n"
#         "3. Check if there's a specific time frame for the prediction.\n"
#         "4. Reference the dataset schema if available.\n"
#         "5. Once you have confirmed all necessary information with the user, "
#         "provide a summary of the inputs and let them know they can generate the notebook.\n\n"
#         "Conversation history:\n{history}\n"
#         "User input:\n{user_input}\n"
#         "Assistant:"
#     ),
# )

# def get_s3_client():
#     print("[DEBUG] Creating S3 client...")
#     return boto3.client(
#         's3',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )

# def get_glue_client():
#     print("[DEBUG] Creating Glue client...")
#     return boto3.client(
#         'glue',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )

# def infer_column_dtype(series: pd.Series) -> str:
#     series = series.dropna().astype(str).str.strip()
#     date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
#     for date_format in date_formats:
#         try:
#             pd.to_datetime(series, format=date_format, errors='raise')
#             return "timestamp"
#         except ValueError:
#             continue

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

#     return "string"

# def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
#     return df.columns[-1]

# def suggest_entity_id_column(df: pd.DataFrame) -> Any:
#     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
#     for col in likely_id_columns:
#         if df[col].nunique() / len(df) > 0.95:
#             return col
#     for col in df.columns:
#         if df[col].nunique() / len(df) > 0.95:
#             return col
#     return None

# def execute_sql_query(query: str) -> pd.DataFrame:
#     print("[DEBUG] Executing Athena query:", query)
#     try:
#         if not AWS_ATHENA_S3_STAGING_DIR:
#             raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")

#         connection_string = (
#             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
#             f"@athena.{AWS_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}"
#             f"?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
#         )
#         print("[DEBUG] Athena connection string created.")
#         engine = create_engine(connection_string)
#         df = pd.read_sql_query(query, engine)
#         print(f"[DEBUG] Query executed successfully. Rows returned: {len(df)}")
#         return df
#     except Exception as e:
#         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
#         return pd.DataFrame()

# def normalize_column_name(col_name: str) -> str:
#     return col_name.strip().lower().replace(' ', '_')

# def parse_user_adjustments(user_input, uploaded_file_info):
#     print("[DEBUG] Parsing user adjustments...")
#     columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
#     normalized_columns = [normalize_column_name(c) for c in columns_list]
#     print("[DEBUG] Current columns_list:", columns_list)
#     print("[DEBUG] Normalized columns_list:", normalized_columns)
#     print("[DEBUG] User input:", user_input)

#     adjustments = {}
#     lines = user_input.strip().split(',')
#     for line in lines:
#         if ':' in line:
#             key, value = line.split(':', 1)
#             key = key.strip().lower()
#             value = value.strip()
#             val_norm = normalize_column_name(value)
#             if 'entity' in key and 'column' in key:
#                 if val_norm in normalized_columns:
#                     match_col = columns_list[normalized_columns.index(val_norm)]
#                     adjustments['entity_id_column'] = match_col
#                 else:
#                     print("[DEBUG] Entity ID column not found:", val_norm)
#             elif 'target' in key and 'column' in key:
#                 if val_norm in normalized_columns:
#                     match_col = columns_list[normalized_columns.index(val_norm)]
#                     adjustments['target_column'] = match_col
#                 else:
#                     print("[DEBUG] Target column not found:", val_norm)

#     if adjustments.get('entity_id_column') and adjustments.get('target_column'):
#         entity_id = adjustments['entity_id_column']
#         target_col = adjustments['target_column']
#         feature_columns = [
#             {'column_name': col} for col in columns_list
#             if col not in [entity_id, target_col]
#         ]
#         adjustments['feature_columns'] = feature_columns
#         print("[DEBUG] Adjustments found:", adjustments)
#         return adjustments

#     print("[DEBUG] No valid adjustments found.")
#     return None

# class UnifiedChatGPTAPI(APIView):
#     parser_classes = [MultiPartParser, FormParser, JSONParser]

#     def post(self, request):
#         action = request.data.get('action', '')
#         if action == 'reset':
#             return self.reset_conversation(request)
#         if action == 'generate_notebook':
#             return self.generate_notebook(request)
#         if "file" in request.FILES:
#             return self.handle_file_upload(request, request.FILES.getlist("file"))
#         return self.handle_chat(request)

#     def handle_file_upload(self, request, files: List[Any]):
#         user_id = request.data.get("user_id", "default_user")
#         s3 = get_s3_client()
#         glue = get_glue_client()
#         uploaded_files_info = []

#         for file in files:
#             print(f"[DEBUG] Processing file: {file.name}")
#             try:
#                 if file.name.lower().endswith('.csv'):
#                     df = pd.read_csv(file, low_memory=False, encoding='utf-8', delimiter=',', na_values=['NA', 'N/A', ''])
#                 else:
#                     df = pd.read_excel(file, engine='openpyxl')

#                 if df.empty:
#                     print("[ERROR] File is empty:", file.name)
#                     return Response({"error": f"Uploaded file {file.name} is empty."}, status=status.HTTP_400_BAD_REQUEST)

#                 if not df.columns.any():
#                     print("[ERROR] File has no columns:", file.name)
#                     return Response({"error": f"Uploaded file {file.name} has no columns."}, status=status.HTTP_400_BAD_REQUEST)
#             except pd.errors.ParserError as e:
#                 print("[ERROR] CSV parsing error:", e)
#                 return Response({"error": f"CSV parsing error for file {file.name}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
#             except Exception as e:
#                 print("[ERROR] Error reading file:", e)
#                 return Response({"error": f"Error reading file {file.name}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#             normalized_columns = [normalize_column_name(c) for c in df.columns]
#             if len(normalized_columns) != len(set(normalized_columns)):
#                 print("[ERROR] Duplicate columns after normalization.")
#                 return Response({"error": "Duplicate columns detected after normalization."}, status=status.HTTP_400_BAD_REQUEST)
#             if any(col == '' for col in normalized_columns):
#                 print("[ERROR] Empty column names after normalization.")
#                 return Response({"error": "Some columns have empty names after normalization."}, status=status.HTTP_400_BAD_REQUEST)

#             df.columns = normalized_columns
#             schema = [
#                 {
#                     "column_name": col,
#                     "data_type": infer_column_dtype(df[col])
#                 }
#                 for col in df.columns
#             ]

#             boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
#             replacement_dict = {
#                 '1': 'true',
#                 '0': 'false',
#                 'yes': 'true',
#                 'no': 'false',
#                 't': 'true',
#                 'f': 'false',
#                 'y': 'true',
#                 'n': 'false',
#                 'true': 'true',
#                 'false': 'false',
#             }
#             for col in boolean_columns:
#                 df[col] = df[col].astype(str).str.strip().str.lower().replace(replacement_dict)
#                 unexpected_values = df[col].unique().tolist()
#                 unexpected_values = [val for val in unexpected_values if val not in ['true', 'false']]
#                 if unexpected_values:
#                     print("[ERROR] Unexpected boolean values:", unexpected_values)
#                     return Response({"error": f"Unexpected boolean values in column {col}: {unexpected_values}"}, status=status.HTTP_400_BAD_REQUEST)

#             file_name_base, file_extension = os.path.splitext(file.name)
#             file_name_base = file_name_base.lower().replace(' ', '_')
#             unique_id = uuid.uuid4().hex[:8]
#             new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
#             s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
#             file_key = f"uploads/{unique_id}/{s3_file_name}"

#             print("[DEBUG] Uploading file to S3 at key:", file_key)
#             try:
#                 with transaction.atomic():
#                     file.seek(0)
#                     file_serializer = UploadedFileSerializer(data={'name': new_file_name, 'file': file})
#                     if file_serializer.is_valid():
#                         file_instance = file_serializer.save()
#                         csv_buffer = BytesIO()
#                         df.to_csv(csv_buffer, index=False, encoding='utf-8')
#                         csv_buffer.seek(0)
#                         s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
#                         print("[DEBUG] S3 upload successful:", file_key)
#                         s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
#                         file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_key}"
#                         file_instance.file_url = file_url
#                         file_instance.save()
#                         FileSchema.objects.create(file=file_instance, schema=schema)

#                         file_size_mb = file.size / (1024 * 1024)
#                         self.trigger_glue_update(new_file_name, schema, file_key, file_size_mb)

#                         uploaded_files_info.append({
#                             'id': file_instance.id,
#                             'name': file_instance.name,
#                             'file_url': file_instance.file_url,
#                             'schema': schema,
#                             'file_size_mb': file_size_mb,
#                             'suggestions': {
#                                 'target_column': suggest_target_column(df, []),
#                                 'entity_id_column': suggest_entity_id_column(df),
#                                 'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
#                             }
#                         })
#                     else:
#                         print("[ERROR] File serializer errors:", file_serializer.errors)
#                         return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#             except ClientError as e:
#                 print("[ERROR] AWS ClientError:", e)
#                 return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#             except Exception as e:
#                 print("[ERROR] Unexpected error during file processing:", e)
#                 return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         user_id = request.data.get("user_id", "default_user")
#         user_schemas[user_id] = uploaded_files_info

#         if user_id not in user_conversations:
#             conversation_chain = ConversationChain(
#                 llm=llm_chatgpt,
#                 prompt=prompt_chatgpt,
#                 input_key="user_input",
#                 memory=ConversationBufferMemory()
#             )
#             user_conversations[user_id] = conversation_chain
#         else:
#             conversation_chain = user_conversations[user_id]

#         schema_discussion = self.format_schema_message(uploaded_files_info[0])
#         conversation_chain.memory.chat_memory.messages.append(
#             AIMessage(content=schema_discussion)
#         )

#         print("[DEBUG] Files uploaded and schema discussion initiated.")
#         return Response({
#             "message": "Files uploaded and processed successfully.",
#             "uploaded_files": uploaded_files_info,
#             "chat_message": schema_discussion
#         }, status=status.HTTP_201_CREATED)

#     # def handle_chat(self, request):
#     #     user_input = request.data.get("message", "").strip()
#     #     user_id = request.data.get("user_id", "default_user")
#     #     print(f"[DEBUG] Handling chat for user: {user_id}, user_input: {user_input}")

#     #     if not user_input:
#     #         return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

#     #     if user_id not in user_conversations:
#     #         conversation_chain = ConversationChain(
#     #             llm=llm_chatgpt,
#     #             prompt=prompt_chatgpt,
#     #             input_key="user_input",
#     #             memory=ConversationBufferMemory()
#     #         )
#     #         user_conversations[user_id] = conversation_chain
#     #     else:
#     #         conversation_chain = user_conversations[user_id]

#     #     if user_id in user_schemas and user_id not in user_confirmations:
#     #         print("[DEBUG] User provided schema confirmation or adjustments.")
#     #         # Here we run process_schema_confirmation and ensure we return show_generate_notebook: true
#     #         assistant_response = self.process_schema_confirmation(user_input, user_id)
#     #         return Response({"response": assistant_response, "show_generate_notebook": True})

#     #     assistant_response = conversation_chain.run(user_input=user_input)
#     #     conversation_chain.memory.chat_memory.messages.append(AIMessage(content=assistant_response))
#     #     print("[DEBUG] Assistant response:", assistant_response)
#     #     return Response({"response": assistant_response})

#     def handle_chat(self, request):
#         user_input = request.data.get("message", "").strip()
#         user_id = request.data.get("user_id", "default_user")
#         print(f"[DEBUG] Handling chat for user: {user_id}, user_input: {user_input}")

#         if not user_input:
#             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

#         if user_id not in user_conversations:
#             conversation_chain = ConversationChain(
#                 llm=llm_chatgpt,
#                 prompt=prompt_chatgpt,
#                 input_key="user_input",
#                 memory=ConversationBufferMemory()
#             )
#             user_conversations[user_id] = conversation_chain
#         else:
#             conversation_chain = user_conversations[user_id]

#         if user_id in user_schemas and user_id not in user_confirmations:
#             print("[DEBUG] User provided schema confirmation or adjustments.")
#             assistant_response, show_generate_notebook = self.process_schema_confirmation(user_input, user_id)
#             return Response({"response": assistant_response, "show_generate_notebook": show_generate_notebook})

#         assistant_response = conversation_chain.run(user_input=user_input)
#         conversation_chain.memory.chat_memory.messages.append(AIMessage(content=assistant_response))
#         print("[DEBUG] Assistant response:", assistant_response)
#         return Response({"response": assistant_response})

#     # def process_schema_confirmation(self, user_input, user_id):
#     #     print("[DEBUG] Processing schema confirmation for user:", user_id)
#     #     conversation_chain = user_conversations[user_id]
#     #     uploaded_file_info = user_schemas[user_id][0]
#     #     suggestions = uploaded_file_info['suggestions']

#     #     if 'yes' in user_input.lower():
#     #         print("[DEBUG] User confirmed suggested schema.")
#     #         user_confirmations[user_id] = {
#     #             'entity_id_column': suggestions['entity_id_column'],
#     #             'target_column': suggestions['target_column'],
#     #             'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']]
#     #         }
#     #         assistant_response = (
#     #             f"Great! You've confirmed the schema:\n\n"
#     #             f"- Entity ID Column: {suggestions['entity_id_column']}\n"
#     #             f"- Target Column: {suggestions['target_column']}\n"
#     #             f"- Feature Columns: {', '.join(suggestions['feature_columns'])}\n\n"
            
#     #         )
#     #     else:
#     #         adjusted_columns = parse_user_adjustments(user_input, uploaded_file_info)
#     #         if adjusted_columns:
#     #             user_confirmations[user_id] = adjusted_columns
#     #             assistant_response = (
#     #                 f"Thanks for the corrections! The updated schema is:\n\n"
#     #                 f"- Entity ID Column: {adjusted_columns['entity_id_column']}\n"
#     #                 f"- Target Column: {adjusted_columns['target_column']}\n"
#     #                 f"- Feature Columns: {', '.join([col['column_name'] for col in adjusted_columns['feature_columns']])}\n\n"
                   
#     #             )
#     #         else:
#     #             print("[DEBUG] Could not find adjusted columns in the dataset.")
#     #             assistant_response = (
#     #                 "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."
#     #             )

#     #     conversation_chain.memory.chat_memory.messages.append(AIMessage(content=assistant_response))
#     #     # After confirming or correcting schema, we always show_generate_notebook: true if success
#     #     return 
    
#     def process_schema_confirmation(self, user_input, user_id):
#         print("[DEBUG] Processing schema confirmation for user:", user_id)
#         conversation_chain = user_conversations[user_id]
#         uploaded_file_info = user_schemas[user_id][0]
#         suggestions = uploaded_file_info['suggestions']

#         if 'yes' in user_input.lower():
#             print("[DEBUG] User confirmed suggested schema.")
#             user_confirmations[user_id] = {
#                 'entity_id_column': suggestions['entity_id_column'],
#                 'target_column': suggestions['target_column'],
#                 'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']]
#             }
#             assistant_response = (
#                 f"Great! You've confirmed the schema:\n\n"
#                 f"- Entity ID Column: {suggestions['entity_id_column']}\n"
#                 f"- Target Column: {suggestions['target_column']}\n"
#                 f"- Feature Columns: {', '.join(suggestions['feature_columns'])}\n\n"
#                 "You can now proceed to generate the notebook."
#             )
#             show_generate_notebook = True
#         else:
#             adjusted_columns = parse_user_adjustments(user_input, uploaded_file_info)
#             if adjusted_columns:
#                 user_confirmations[user_id] = adjusted_columns
#                 assistant_response = (
#                     f"Thanks for the corrections! The updated schema is:\n\n"
#                     f"- Entity ID Column: {adjusted_columns['entity_id_column']}\n"
#                     f"- Target Column: {adjusted_columns['target_column']}\n"
#                     f"- Feature Columns: {', '.join([col['column_name'] for col in adjusted_columns['feature_columns']])}\n\n"
#                     "You can now proceed to generate the notebook."
#                 )
#                 show_generate_notebook = True
#             else:
#                 print("[DEBUG] Could not find adjusted columns in the dataset.")
#                 assistant_response = (
#                     "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."
#                 )
#                 show_generate_notebook = False

#         conversation_chain.memory.chat_memory.messages.append(AIMessage(content=assistant_response))
#         return assistant_response, show_generate_notebook

#     def reset_conversation(self, request):
#         user_id = request.data.get("user_id", "default_user")
#         print("[DEBUG] Resetting conversation for user:", user_id)
#         if user_id in user_conversations:
#             del user_conversations[user_id]
#         if user_id in user_schemas:
#             del user_schemas[user_id]
#         if user_id in user_confirmations:
#             del user_confirmations[user_id]
#         if user_id in user_notebook_flags:
#             del user_notebook_flags[user_id]
#         if user_id in user_notebooks:
#             del user_notebooks[user_id]
#         return Response({"message": "Conversation reset successful."})

#     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
#         schema = uploaded_file['schema']
#         target_column = uploaded_file['suggestions']['target_column']
#         entity_id_column = uploaded_file['suggestions']['entity_id_column']
#         feature_columns = uploaded_file['suggestions']['feature_columns']
#         schema_text = (
#             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
#             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
#             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
#             f"Suggested Target Column: {target_column or 'None'}\n"
#             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
#             f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
#             "Please confirm:\n"
#             "- Is the Target Column correct?\n"
#             "- Is the Entity ID Column correct?\n"
#             "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
#         )
#         return schema_text

#     # def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
#     #     print("[DEBUG] Triggering Glue update for table:", table_name)
#     #     glue = get_glue_client()
#     #     unique_id = file_key.split('/')[1]
#     #     s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/{unique_id}/"
#     #     table_name_without_extension = self.sanitize_identifier(os.path.splitext(table_name)[0])
#     #     storage_descriptor = {
#     #         'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
#     #         'Location': s3_location,
#     #         'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
#     #         'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
#     #         'SerdeInfo': {
#     #             'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
#     #             'Parameters': {
#     #                 'field.delim': ',',
#     #                 'skip.header.line.count': '1'
#     #             }
#     #         }
#     #     }
#     #     try:
#     #         glue.update_table(
#     #             DatabaseName=ATHENA_SCHEMA_NAME,
#     #             TableInput={
#     #                 'Name': table_name_without_extension,
#     #                 'StorageDescriptor': storage_descriptor,
#     #                 'TableType': 'EXTERNAL_TABLE'
#     #             }
#     #         )
#     #         print("[DEBUG] Glue table updated:", table_name_without_extension)
#     #     except glue.exceptions.EntityNotFoundException:
#     #         print("[DEBUG] Glue table not found, creating new one:", table_name_without_extension)
#     #         glue.create_table(
#     #             DatabaseName=ATHENA_SCHEMA_NAME,
#     #             TableInput={
#     #                 'Name': table_name_without_extension,
#     #                 'StorageDescriptor': storage_descriptor,
#     #                 'TableType': 'EXTERNAL_TABLE'
#     #             }
#     #         )
#     #         print("[DEBUG] Glue table created:", table_name_without_extension)
#     #     except Exception as e:
#     #         print("[ERROR] Glue operation failed:", e)

#     #     base_timeout = 80
#     #     additional_timeout_per_mb = 5
#     #     dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
#     #     self.wait_for_table_creation(table_name_without_extension, timeout=dynamic_timeout)

#     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
#         """
#         Updates or creates a Glue table and stores the table name in `user_schemas`.
#         This function ensures that the dataset is registered in Glue, enabling Athena to query it.

#         Args:
#             table_name (str): Name of the table to create or update.
#             schema (List[Dict[str, str]]): Schema of the table (columns and types).
#             file_key (str): S3 file key where the dataset resides.
#             file_size_mb (float): Approximate file size in MB to calculate wait timeout.

#         Raises:
#             Exception: If Glue operations fail or timeout occurs.
#         """
#         print("[DEBUG] Triggering Glue update for table:", table_name)
#         glue = get_glue_client()
        
#         # Extract unique ID from file key to determine the S3 location
#         unique_id = file_key.split('/')[1]
#         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/{unique_id}/"
        
#         # Sanitize the table name to ensure it meets Glue's naming requirements
#         glue_table_name = self.sanitize_identifier(os.path.splitext(table_name)[0])

#         # Define the storage descriptor with schema, location, and SerDe settings
#         storage_descriptor = {
#             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
#             'Location': s3_location,
#             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
#             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
#             'SerdeInfo': {
#                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
#                 'Parameters': {
#                     'field.delim': ',',  # Specify delimiter as comma
#                     'skip.header.line.count': '1'  # Skip header row in CSV
#                 }
#             }
#         }

#         try:
#             # Attempt to update the Glue table if it already exists
#             glue.update_table(
#                 DatabaseName=ATHENA_SCHEMA_NAME,
#                 TableInput={
#                     'Name': glue_table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             print("[DEBUG] Glue table updated successfully:", glue_table_name)
#         except glue.exceptions.EntityNotFoundException:
#             # If the table does not exist, create a new one
#             print("[DEBUG] Glue table not found, creating a new one:", glue_table_name)
#             glue.create_table(
#                 DatabaseName=ATHENA_SCHEMA_NAME,
#                 TableInput={
#                     'Name': glue_table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             print("[DEBUG] Glue table created successfully:", glue_table_name)

#         # Store Glue table name in user_schemas for reference
#         user_id = self.get_user_id_from_file_key(file_key)
#         if user_id in user_schemas:
#             user_schemas[user_id][0]["glue_table_name"] = glue_table_name
#             print(f"[DEBUG] Stored Glue table name '{glue_table_name}' for user '{user_id}'.")

#         # Wait for Glue table creation and availability in Athena
#         base_timeout = 80
#         additional_timeout_per_mb = 5
#         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
#         self.wait_for_table_creation(glue_table_name, timeout=dynamic_timeout)

#     def get_user_id_from_file_key(self, file_key: str) -> str:
#         """
#         Extracts the user ID from the S3 file key.

#         Args:
#             file_key (str): The S3 file key (e.g., "uploads/<user_id>/file_name.csv").

#         Returns:
#             str: The extracted user ID. If no structure is defined, returns 'default_user'.

#         Notes:
#             - This function assumes that the user ID is the second segment of the file key.
#             - Adjust the parsing logic based on the actual S3 key structure.
#         """
#         try:
#             return file_key.split("/")[1]  # Extract the second segment as user ID
#         except IndexError:
#             print("[WARNING] Unable to extract user ID, defaulting to 'default_user'.")
#             return "default_user"


#     def sanitize_identifier(self, name):
#         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())

#     def validate_column_exists(self, column_name, columns_list):
#         print("[DEBUG] Validating column existence:", column_name)
#         print("[DEBUG] Available columns:", columns_list)
#         norm_col = normalize_column_name(column_name)
#         norm_list = [normalize_column_name(c) for c in columns_list]
#         if norm_col in norm_list:
#             return True
#         else:
#             print("[DEBUG] Column not found after normalization:", norm_col)
#             return False

#     def sanitize_notebook(self, nb):
#         def sanitize(obj):
#             if isinstance(obj, dict):
#                 for k in obj:
#                     obj[k] = sanitize(obj[k])
#                 return obj
#             elif isinstance(obj, list):
#                 return [sanitize(v) for v in obj]
#             elif isinstance(obj, float):
#                 if np.isnan(obj) or np.isinf(obj):
#                     return None
#                 else:
#                     return obj
#             else:
#                 return obj
#         sanitize(nb)
#         return nb

#     # def generate_notebook(self, request):
#     #     user_id = request.data.get("user_id", "default_user")
#     #     print("[DEBUG] Generating notebook for user:", user_id)

#     #     if user_id not in user_confirmations:
#     #         print("[ERROR] Schema not confirmed yet.")
#     #         return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

#     #     confirmation = user_confirmations[user_id]
#     #     entity_id_column = confirmation['entity_id_column']
#     #     target_column = confirmation['target_column']
#     #     feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

#     #     if user_id in user_schemas:
#     #         uploaded_file_info = user_schemas[user_id][0]
#     #         table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
#     #         sanitized_table_name = self.sanitize_identifier(table_name_raw)
#     #     else:
#     #         print("[ERROR] Uploaded file info not found.")
#     #         return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

#     #     columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

#     #     if not self.validate_column_exists(entity_id_column, columns_list):
#     #         return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

#     #     if not self.validate_column_exists(target_column, columns_list):
#     #         return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

#     #     notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, sanitized_table_name, columns_list)
#     #     notebook_features = self.create_features_notebook(feature_columns, sanitized_table_name, columns_list)

#     #     notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
#     #     notebook_features_sanitized = self.sanitize_notebook(notebook_features)

#     #     notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
#     #     notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

#     #     user_notebooks[user_id] = {
#     #         'entity_target_notebook': notebook_entity_target_json,
#     #         'features_notebook': notebook_features_json
#     #     }

#     #     print("[DEBUG] Notebooks generated successfully for user:", user_id)
#     #     return Response({
#     #         "message": "Notebooks generated successfully.",
#     #         "notebooks": user_notebooks[user_id]
#     #     }, status=status.HTTP_200_OK)

#     def generate_notebook(self, request):
#         user_id = request.data.get("user_id", "default_user")
#         print("[DEBUG] Generating notebook for user:", user_id)

#         if user_id not in user_confirmations:
#             print("[ERROR] Schema not confirmed yet.")
#             return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

#         confirmation = user_confirmations[user_id]
#         entity_id_column = confirmation['entity_id_column']
#         target_column = confirmation['target_column']
#         feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

#         if user_id in user_schemas:
#             uploaded_file_info = user_schemas[user_id][0]
#             table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
#             sanitized_table_name = self.sanitize_identifier(table_name_raw)
#             file_url = uploaded_file_info.get('file_url')  # Ensure we have the full S3 URL
#         else:
#             print("[ERROR] Uploaded file info not found.")
#             return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

#         columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

#         if not self.validate_column_exists(entity_id_column, columns_list):
#             return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

#         if not self.validate_column_exists(target_column, columns_list):
#             return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

#         # Generate notebooks (as before)
#         notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, sanitized_table_name, columns_list)
#         notebook_features = self.create_features_notebook(feature_columns, sanitized_table_name, columns_list)

#         notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
#         notebook_features_sanitized = self.sanitize_notebook(notebook_features)

#         notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
#         notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

#         user_notebooks[user_id] = {
#             'entity_target_notebook': notebook_entity_target_json,
#             'features_notebook': notebook_features_json
#         }

#         print("[DEBUG] Notebooks generated successfully for user:", user_id)
#         print("[DEBUG] Sending data to DataForAutomationAPI...")

#         # Prepare payload for DataForAutomationAPI
#         payload = {
#             "file_url": file_url,
#             "entity_column": entity_id_column,
#             "target_column": target_column,
#             "features": feature_columns
#         }

#         # Store the payload in user_schemas for later use
#         user_schemas[user_id][0]["automation_payload"] = payload

#         # AUTOMATION_API_URL = "http://localhost:8000/api/automation/"  # Adjust as needed

#         # try:
#         #     print("[DEBUG] POSTing to DataForAutomationAPI:", payload)
#         #     automation_response = requests.post(AUTOMATION_API_URL, json=payload)
#         #     print("[DEBUG] DataForAutomationAPI Response Code:", automation_response.status_code)
#         #     print("[DEBUG] DataForAutomationAPI Response:", automation_response.json())
#         #     if automation_response.status_code != 200:
#         #         return Response({"error": "Failed to pass data to DataForAutomationAPI.", "details": automation_response.json()}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#         # except Exception as e:
#         #     print("[ERROR] Failed to call DataForAutomationAPI:", str(e))
#         #     return Response({"error": "Failed to call DataForAutomationAPI.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         return Response({
#             "message": "Notebooks generated and data passed to DataForAutomationAPI successfully.",
#             "notebooks": user_notebooks[user_id]
#         }, status=status.HTTP_200_OK)




#     def create_entity_target_notebook(self, entity_id_column, target_column, table_name, columns_list):
#         print("[DEBUG] Creating entity-target notebook...")
#         nb = new_notebook()
#         cells = []
#         cells.append(new_markdown_cell("Core Set"))

#         sanitized_entity_id_column = self.sanitize_identifier(entity_id_column)
#         sanitized_target_column = self.sanitize_identifier(target_column)
#         sql_query_entity_target = (
#             f"SELECT {sanitized_entity_id_column}, {sanitized_target_column} "
#             f"FROM {table_name} LIMIT 10;"
#         )

#         df_result = execute_sql_query(sql_query_entity_target)
#         if df_result.empty:
#             error_message = f"No data returned for query: {sql_query_entity_target}"
#             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
#         else:
#             df_result = df_result.replace([np.nan, np.inf, -np.inf], None)
#             result_json = df_result.to_dict(orient='records')
#             columns = []
#             for col in df_result.columns:
#                 non_null_series = pd.Series([x for x in df_result[col] if x is not None])
#                 if non_null_series.empty:
#                     col_type = "string"
#                 else:
#                     col_type = infer_column_dtype(non_null_series)
#                 columns.append({'name': col, 'type': col_type})

#             text_repr = df_result.head().to_string(index=False)

#             code_cell = new_code_cell(sql_query_entity_target)
#             code_cell['execution_count'] = 1
#             code_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={
#                         'application/json': {
#                             'rows': result_json,
#                             'columns': columns
#                         },
#                         'text/plain': text_repr
#                     },
#                     metadata={},
#                     execution_count=1
#                 )
#             ]
#             cells.append(code_cell)

#         nb['cells'] = cells
#         return nb

#     def create_features_notebook(self, feature_columns, table_name, columns_list):
#         print("[DEBUG] Creating features notebook...")
#         nb = new_notebook()
#         cells = []
#         cells.append(new_markdown_cell("Features or Attributes Test"))
#         sanitized_features = [self.sanitize_identifier(feature) for feature in feature_columns]

#         missing_columns = [feature for feature in sanitized_features if feature not in columns_list]
#         if missing_columns:
#             error_message = f"The following feature columns do not exist in the dataset: {', '.join(missing_columns)}"
#             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
#             nb['cells'] = cells
#             return nb

#         feature_query = (
#             f"SELECT\n    " + ",\n    ".join(sanitized_features) +
#             f"\nFROM {table_name}\nLIMIT 10;"
#         )

#         df_result = execute_sql_query(feature_query)
#         if df_result.empty:
#             error_message = f"No data returned for query: {feature_query}"
#             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
#         else:
#             df_result = df_result.replace([np.nan, np.inf, -np.inf], None)
#             result_json = df_result.to_dict(orient='records')
#             columns = []
#             for col in df_result.columns:
#                 non_null_series = pd.Series([x for x in df_result[col] if x is not None])
#                 if non_null_series.empty:
#                     col_type = "string"
#                 else:
#                     col_type = infer_column_dtype(non_null_series)
#                 columns.append({'name': col, 'type': col_type})

#             text_repr = df_result.head().to_string(index=False)

#             code_cell = new_code_cell(feature_query)
#             code_cell['execution_count'] = 1
#             code_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={
#                         'application/json': {
#                             'rows': result_json,
#                             'columns': columns
#                         },
#                         'text/plain': text_repr
#                     },
#                     metadata={},
#                     execution_count=1
#                 )
#             ]
#             cells.append(code_cell)

#         nb['cells'] = cells
#         return nb

#     def wait_for_table_creation(self, table_name, timeout):
#         import time
#         glue_client = get_glue_client()
#         start_time = time.time()
#         glue_table_ready = False
#         athena_table_ready = False

#         print("[DEBUG] Waiting for Glue table creation:", table_name)
#         while time.time() - start_time < timeout:
#             try:
#                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
#                 print("[DEBUG] Glue table is now available:", table_name)
#                 glue_table_ready = True
#                 break
#             except glue_client.exceptions.EntityNotFoundException:
#                 time.sleep(5)
#             except Exception as e:
#                 print("[ERROR] Unexpected error while checking Glue table availability:", e)
#                 return False

#         if not glue_table_ready:
#             print(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
#             return False

#         print("[DEBUG] Checking Athena table availability:", table_name)
#         while time.time() - start_time < timeout:
#             try:
#                 query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
#                 df = execute_sql_query(query)
#                 if df.empty:
#                     print("[DEBUG] Athena recognizes the table (no error), table ready:", table_name)
#                     athena_table_ready = True
#                     break
#                 else:
#                     print("[DEBUG] Athena table ready with data:", table_name)
#                     athena_table_ready = True
#                     break
#             except Exception as e:
#                 error_message = str(e)
#                 if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
#                     print("[DEBUG] Still waiting for Athena to recognize table:", table_name)
#                     time.sleep(10)
#                 else:
#                     print("[ERROR] Unexpected error while checking Athena table availability:", e)
#                     return False

#         if not athena_table_ready:
#             print(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
#             return False

#         return True
























# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status, permissions
# from .models import Chat, Message
# from .serializers import ChatSerializer, MessageSerializer

# # class ChatListView(APIView):
# #     permission_classes = [permissions.IsAuthenticated]

# #     def get(self, request):
# #         chats = Chat.objects.filter(user=request.user)
# #         serializer = ChatSerializer(chats, many=True)
# #         return Response(serializer.data)

# #     def post(self, request):
# #         chat = Chat.objects.create(user=request.user, title=request.data.get('title', 'New Chat'))
# #         serializer = ChatSerializer(chat)
# #         return Response(serializer.data, status=status.HTTP_201_CREATED)
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import IsAuthenticated
# from .models import Chat, Message
# from .serializers import ChatSerializer, MessageSerializer

# class ChatListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def get(self, request):
#         chats = Chat.objects.filter(user=request.user)
#         serializer = ChatSerializer(chats, many=True)
#         return Response(serializer.data)

#     def post(self, request):
#         data = request.data
#         chat = Chat.objects.create(user=request.user, title=data.get('title'))
#         return Response({'chat_id': chat.chat_id})

    
    

# class MessageListView(APIView):
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, chat_id):
#         try:
#             chat = Chat.objects.get(id=chat_id, user=request.user)
#         except Chat.DoesNotExist:
#             return Response({'error': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND)

#         message = Message.objects.create(
#             chat=chat,
#             sender=request.data.get('sender'),
#             text=request.data.get('text')
#         )
#         serializer = MessageSerializer(message)
#         return Response(serializer.data, status=status.HTTP_201_CREATED)

# # import uuid
# # from django.contrib.auth.models import User
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework.permissions import IsAuthenticated
# # from rest_framework import status

# # from .models import Chat, Message, ChatBackup
# # from .serializers import ChatSerializer, MessageSerializer

# # class ChatListView(APIView):
# #     # permission_classes = [IsAuthenticated]

# #     def get(self, request):
# #         print("[DEBUG] ChatListView GET called")
# #         # Commenting out dynamic user:
# #         # chats = Chat.objects.filter(user=request.user)
# #         # Hard-coding user_id = 15
# #         user = User.objects.get(id=15)
# #         chats = Chat.objects.filter(user=user)
# #         serializer = ChatSerializer(chats, many=True)
# #         print("[DEBUG] Returning chats:", serializer.data)
# #         return Response(serializer.data)

# #     def post(self, request):
# #         print("[DEBUG] ChatListView POST called with data:", request.data)
# #         data = request.data
# #         # Commenting out dynamic user:
# #         # chat = Chat.objects.create(user=request.user, title=data.get('title', 'New Chat'))
# #         # Hard-coding user_id = 15
# #         user = User.objects.get(id=1)
# #         chat = Chat.objects.create(user=user, title=data.get('title', 'New Chat'))
# #         print("[DEBUG] Chat created with chat_id:", chat.chat_id, "and user_id: 15")

# #         # Create corresponding ChatBackup
# #         ChatBackup.objects.create(
# #             user=user,
# #             chat_id=str(chat.chat_id),
# #             title=chat.title,
# #             messages=[]
# #         )
# #         print("[DEBUG] ChatBackup created for chat_id:", chat.chat_id)

# #         return Response({'chat_id': str(chat.chat_id)}, status=status.HTTP_201_CREATED)


# # class ChatDetailView(APIView):
# #     # permission_classes = [IsAuthenticated]

# #     def delete(self, request, chat_id):
# #         print("[DEBUG] ChatDetailView DELETE called for chat_id:", chat_id)
# #         from uuid import UUID
# #         try:
# #             chat_uuid = UUID(chat_id)
# #             # Commenting out dynamic user:
# #             # chat = Chat.objects.get(chat_id=chat_uuid, user=request.user)
# #             # Hard-coding user_id = 15
# #             user = User.objects.get(id=1)
# #             chat = Chat.objects.get(chat_id=chat_uuid, user=user)
# #         except (Chat.DoesNotExist, ValueError):
# #             print("[DEBUG] Chat not found or invalid UUID for chat_id:", chat_id)
# #             return Response({'error': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND)

# #         print("[DEBUG] Deleting chat:", chat.chat_id, "and its backup.")
# #         ChatBackup.objects.filter(chat_id=str(chat.chat_id), user=user).delete()
# #         chat.delete()

# #         return Response({"message": "Chat and its backup deleted successfully."}, status=status.HTTP_204_NO_CONTENT)


# # class MessageListView(APIView):
# #     # permission_classes = [IsAuthenticated]

# #     def post(self, request, chat_id):
# #         print("[DEBUG] MessageListView POST called for chat_id:", chat_id, "with data:", request.data)
# #         from uuid import UUID
# #         try:
# #             chat_uuid = UUID(chat_id)
# #             # Commenting out dynamic user:
# #             # chat = Chat.objects.get(chat_id=chat_uuid, user=request.user)
# #             # Hard-coding user_id = 15
# #             user = User.objects.get(id=1)
# #             chat = Chat.objects.get(chat_id=chat_uuid, user=user)
# #         except (Chat.DoesNotExist, ValueError):
# #             print("[DEBUG] Chat not found for chat_id:", chat_id)
# #             return Response({'error': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND)

# #         message = Message.objects.create(
# #             chat=chat,
# #             sender=request.data.get('sender'),
# #             text=request.data.get('text')
# #         )
# #         print("[DEBUG] Message created:", message.text, "for chat_id:", chat_id)

# #         # Update ChatBackup messages
# #         try:
# #             backup = ChatBackup.objects.get(chat_id=str(chat.chat_id), user=user)
# #             backup_messages = backup.messages
# #             backup_messages.append({
# #                 "sender": message.sender,
# #                 "text": message.text,
# #                 "timestamp": message.timestamp.isoformat()
# #             })
# #             backup.messages = backup_messages
# #             backup.save()
# #             print("[DEBUG] ChatBackup updated with new message for chat_id:", chat_id)
# #         except ChatBackup.DoesNotExist:
# #             print("[WARNING] No ChatBackup found for chat_id:", chat_id, "This should not happen if chat creation is correct.")

# #         serializer = MessageSerializer(message)
# #         return Response(serializer.data, status=status.HTTP_201_CREATED)


# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .models import ChatBackup  # Use the updated model

# class ChatHistoryByUserView(APIView):
#     """
#     API to retrieve chat history for a specific user.
#     """

#     def get(self, request):
#         print("DEBUG: ChatHistoryByUserView GET method called")

#         # Retrieve user_id from query parameters
#         user_id = request.GET.get('user_id')
#         print(f"DEBUG: Received user_id: {user_id}")

#         # Validate the input
#         if not user_id:
#             print("ERROR: user_id is missing in the request")
#             return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             # Query the database for chats belonging to the user
#             print(f"DEBUG: Querying chats for user_id: {user_id}")
#             chats = ChatBackup.objects.filter(user_id=user_id)
#             # print(f"DEBUG: Number of chats found: {chats.count()}")

#             if not chats.exists():
#                 print(f"WARNING: No chats found for user_id: {user_id}")
#                 return Response(
#                     {"error": f"No chats found for the given user_id: {user_id}"},
#                     status=status.HTTP_404_NOT_FOUND,
#                 )

#             # Prepare the response data
#             response_data = []
#             for chat in chats:
#                 # print(f"DEBUG: Processing chat with chat_id: {chat.chat_id}")

#                 # Use the JSONField directly
#                 messages = chat.messages
#                 # print(f"DEBUG: Messages for chat_id {chat.chat_id}: {messages}")

#                 # Separate user and assistant messages
#                 user_messages = [msg for msg in messages if msg.get("sender") == "user"]
#                 assistant_messages = [msg for msg in messages if msg.get("sender") == "assistant"]

#                 # Append chat information to the response
#                 response_data.append({
#                     "chat_id": chat.chat_id,
#                     "title": chat.title,
#                     # "created_at": chat.created_at,  # From the model
#                     # "updated_at": chat.updated_at,  # From the model
#                     "user_messages": user_messages,
#                     "assistant_messages": assistant_messages,
#                 })

#             print("DEBUG: Successfully prepared response data")
#             return Response(response_data, status=status.HTTP_200_OK)

#         except ChatBackup.DoesNotExist:
#             print(f"ERROR: No records found for user_id: {user_id}")
#             return Response(
#                 {"error": f"No chat records found for user_id={user_id}"},
#                 status=status.HTTP_404_NOT_FOUND,
#             )

#         except Exception as e:
#             print(f"ERROR: Unexpected error occurred: {e}")
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




from django.db import transaction
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
import os
import datetime
from io import BytesIO
from typing import Any, Dict, List
import uuid
import boto3
import pandas as pd
import openai
import requests
import re
import nbformat
import numpy as np
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from botocore.exceptions import ClientError, NoCredentialsError
from django.conf import settings
from sqlalchemy import create_engine
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
# from langchain.schema import AIMessage
from langchain.schema import AIMessage, HumanMessage

from .models import FileSchema, UploadedFile, ChatBackup
# from .serializers import UploadedFileSerializer, ChatSerializer, MessageSerializer
from .serializers import UploadedFileSerializer

# AWS and OpenAI environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
AWS_REGION_NAME = AWS_S3_REGION_NAME
ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Adjust as needed

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# Initialize the ChatOpenAI model
llm_chatgpt = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# In-memory storage for user-specific data
user_conversations = {}
user_schemas = {}
user_confirmations = {}
user_notebook_flags = {}
user_notebooks = {}

# Prompt template for ChatGPT
prompt_chatgpt = PromptTemplate(
    input_variables=["history", "user_input"],
    template=(
        "You are a highly intelligent and helpful AI assistant. Your "
        "responses should be clear and concise. Assist the user in confirming "
        "the dataset schema and any corrections they provide. Reflect the "
        "confirmed or corrected schema back to the user before proceeding.\n\n"
        "Steps:\n"
        "1. Discuss the subject they want to predict.\n"
        "2. Confirm the target value they want to predict.\n"
        "3. Check if there's a specific time frame for the prediction.\n"
        "4. Reference the dataset schema if available.\n"
        "5. Once you have confirmed all necessary information with the user, "
        "provide a summary of the inputs and let them know they can generate the notebook.\n\n"
        "Conversation history:\n{history}\n"
        "User input:\n{user_input}\n"
        "Assistant:"
    ),
)

def get_s3_client():
    print("[DEBUG] Creating S3 client...")
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def get_glue_client():
    print("[DEBUG] Creating Glue client...")
    return boto3.client(
        'glue',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def infer_column_dtype(series: pd.Series) -> str:
    series = series.dropna().astype(str).str.strip()
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
    for date_format in date_formats:
        try:
            pd.to_datetime(series, format=date_format, errors='raise')
            return "timestamp"
        except ValueError:
            continue

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

    return "string"

def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
    return df.columns[-1]

def suggest_entity_id_column(df: pd.DataFrame) -> Any:
    likely_id_columns = [col for col in df.columns if "id" in col.lower()]
    for col in likely_id_columns:
        if df[col].nunique() / len(df) > 0.95:
            return col
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.95:
            return col
    return None

def execute_sql_query(query: str) -> pd.DataFrame:
    print("[DEBUG] Executing Athena query:", query)
    try:
        if not AWS_ATHENA_S3_STAGING_DIR:
            raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")

        connection_string = (
            f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
            f"@athena.{AWS_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}"
            f"?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
        )
        print("[DEBUG] Athena connection string created.")
        engine = create_engine(connection_string)
        df = pd.read_sql_query(query, engine)
        print(f"[DEBUG] Query executed successfully. Rows returned: {len(df)}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
        return pd.DataFrame()

def normalize_column_name(col_name: str) -> str:
    return col_name.strip().lower().replace(' ', '_')

def parse_user_adjustments(user_input, uploaded_file_info):
    print("[DEBUG] Parsing user adjustments...")
    columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
    normalized_columns = [normalize_column_name(c) for c in columns_list]
    print("[DEBUG] Current columns_list:", columns_list)
    print("[DEBUG] Normalized columns_list:", normalized_columns)
    print("[DEBUG] User input:", user_input)

    adjustments = {}
    lines = user_input.strip().split(',')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            val_norm = normalize_column_name(value)
            if 'entity' in key and 'column' in key:
                if val_norm in normalized_columns:
                    match_col = columns_list[normalized_columns.index(val_norm)]
                    adjustments['entity_id_column'] = match_col
                else:
                    print("[DEBUG] Entity ID column not found:", val_norm)
            elif 'target' in key and 'column' in key:
                if val_norm in normalized_columns:
                    match_col = columns_list[normalized_columns.index(val_norm)]
                    adjustments['target_column'] = match_col
                else:
                    print("[DEBUG] Target column not found:", val_norm)

    if adjustments.get('entity_id_column') and adjustments.get('target_column'):
        entity_id = adjustments['entity_id_column']
        target_col = adjustments['target_column']
        feature_columns = [
            {'column_name': col} for col in columns_list
            if col not in [entity_id, target_col]
        ]
        adjustments['feature_columns'] = feature_columns
        print("[DEBUG] Adjustments found:", adjustments)
        return adjustments

    print("[DEBUG] No valid adjustments found.")
    return None

class UnifiedChatGPTAPI(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        action = request.data.get('action', '')
        if action == 'reset':
            return self.reset_conversation(request)
        if action == 'generate_notebook':
            return self.generate_notebook(request)
        if "file" in request.FILES:
            return self.handle_file_upload(request, request.FILES.getlist("file"))
        return self.handle_chat(request)

    def handle_file_upload(self, request, files: List[Any]):
        user_id = request.data.get("user_id", "default_user")
        s3 = get_s3_client()
        glue = get_glue_client()
        uploaded_files_info = []

        for file in files:
            print(f"[DEBUG] Processing file: {file.name}")
            try:
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(file, low_memory=False, encoding='utf-8', delimiter=',', na_values=['NA', 'N/A', ''])
                else:
                    df = pd.read_excel(file, engine='openpyxl')

                if df.empty:
                    print("[ERROR] File is empty:", file.name)
                    return Response({"error": f"Uploaded file {file.name} is empty."}, status=status.HTTP_400_BAD_REQUEST)

                if not df.columns.any():
                    print("[ERROR] File has no columns:", file.name)
                    return Response({"error": f"Uploaded file {file.name} has no columns."}, status=status.HTTP_400_BAD_REQUEST)
            except pd.errors.ParserError as e:
                print("[ERROR] CSV parsing error:", e)
                return Response({"error": f"CSV parsing error for file {file.name}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                print("[ERROR] Error reading file:", e)
                return Response({"error": f"Error reading file {file.name}: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

            normalized_columns = [normalize_column_name(c) for c in df.columns]
            if len(normalized_columns) != len(set(normalized_columns)):
                print("[ERROR] Duplicate columns after normalization.")
                return Response({"error": "Duplicate columns detected after normalization."}, status=status.HTTP_400_BAD_REQUEST)
            if any(col == '' for col in normalized_columns):
                print("[ERROR] Empty column names after normalization.")
                return Response({"error": "Some columns have empty names after normalization."}, status=status.HTTP_400_BAD_REQUEST)

            df.columns = normalized_columns
            schema = [
                {
                    "column_name": col,
                    "data_type": infer_column_dtype(df[col])
                }
                for col in df.columns
            ]

            boolean_columns = [col['column_name'] for col in schema if col['data_type'] == 'boolean']
            replacement_dict = {
                '1': 'true',
                '0': 'false',
                'yes': 'true',
                'no': 'false',
                't': 'true',
                'f': 'false',
                'y': 'true',
                'n': 'false',
                'true': 'true',
                'false': 'false',
            }
            for col in boolean_columns:
                df[col] = df[col].astype(str).str.strip().str.lower().replace(replacement_dict)
                unexpected_values = df[col].unique().tolist()
                unexpected_values = [val for val in unexpected_values if val not in ['true', 'false']]
                if unexpected_values:
                    print("[ERROR] Unexpected boolean values:", unexpected_values)
                    return Response({"error": f"Unexpected boolean values in column {col}: {unexpected_values}"}, status=status.HTTP_400_BAD_REQUEST)

            file_name_base, file_extension = os.path.splitext(file.name)
            file_name_base = file_name_base.lower().replace(' ', '_')
            unique_id = uuid.uuid4().hex[:8]
            new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
            s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
            file_key = f"uploads/{unique_id}/{s3_file_name}"

            print("[DEBUG] Uploading file to S3 at key:", file_key)
            try:
                with transaction.atomic():
                    file.seek(0)
                    file_serializer = UploadedFileSerializer(data={'name': new_file_name, 'file': file})
                    if file_serializer.is_valid():
                        file_instance = file_serializer.save()
                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        csv_buffer.seek(0)
                        s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
                        print("[DEBUG] S3 upload successful:", file_key)
                        s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
                        file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_key}"
                        file_instance.file_url = file_url
                        file_instance.save()
                        FileSchema.objects.create(file=file_instance, schema=schema)

                        file_size_mb = file.size / (1024 * 1024)
                        self.trigger_glue_update(new_file_name, schema, file_key, file_size_mb)

                        uploaded_files_info.append({
                            'id': file_instance.id,
                            'name': file_instance.name,
                            'file_url': file_instance.file_url,
                            'schema': schema,
                            'file_size_mb': file_size_mb,
                            'suggestions': {
                                'target_column': suggest_target_column(df, []),
                                'entity_id_column': suggest_entity_id_column(df),
                                'feature_columns': [col for col in df.columns if col not in [suggest_entity_id_column(df), suggest_target_column(df, [])]]
                            }
                        })
                    else:
                        print("[ERROR] File serializer errors:", file_serializer.errors)
                        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            except ClientError as e:
                print("[ERROR] AWS ClientError:", e)
                return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                print("[ERROR] Unexpected error during file processing:", e)
                return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        user_id = request.data.get("user_id", "default_user")
        user_schemas[user_id] = uploaded_files_info

        if user_id not in user_conversations:
            conversation_chain = ConversationChain(
                llm=llm_chatgpt,
                prompt=prompt_chatgpt,
                input_key="user_input",
                memory=ConversationBufferMemory()
            )
            user_conversations[user_id] = conversation_chain
        else:
            conversation_chain = user_conversations[user_id]

        schema_discussion = self.format_schema_message(uploaded_files_info[0])
        conversation_chain.memory.chat_memory.messages.append(
            AIMessage(content=schema_discussion)
        )

        print("[DEBUG] Files uploaded and schema discussion initiated.")
        return Response({
            "message": "Files uploaded and processed successfully.",
            "uploaded_files": uploaded_files_info,
            "chat_message": schema_discussion
        }, status=status.HTTP_201_CREATED)


    # def handle_chat(self, request):
    #     user_input = request.data.get("message", "").strip()
    #     user_id = request.data.get("user_id", "default_user")
    #     chat_id = request.data.get("chat_id")  # Accept chat_id from frontend

    #     # Validate input
    #     if not user_input:
    #         return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

    #     try:
    #         user = User.objects.get(id=user_id)
    #     except User.DoesNotExist:
    #         return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    #     # Create a composite key to separate user and chat-specific memory
    #     memory_key = f"{user_id}_{chat_id}"

    #     # Generate a new chat_id if not provided (new chat)
    #     if not chat_id:
    #         chat_id = str(uuid.uuid4())  # New chat resets memory
    #         print(f"[DEBUG] New chat created with chat_id: {chat_id}")

    #         # Initialize fresh conversation memory for a new chat
    #         user_conversations[memory_key] = ConversationChain(
    #             llm=llm_chatgpt,
    #             prompt=prompt_chatgpt,
    #             input_key="user_input",
    #             memory=ConversationBufferMemory()
    #         )
    #     else:
    #         # Retrieve or create a chat session
    #         chat, created = ChatBackup.objects.get_or_create(
    #             user=user, chat_id=chat_id, defaults={"title": "User Chat", "messages": []}
    #         )

    #         if not created and memory_key not in user_conversations:
    #             # Restore memory only if this is an existing chat
    #             print(f"[DEBUG] Restoring memory for chat_id: {chat_id}")
    #             restored_memory = ConversationBufferMemory()
    #             for msg in chat.messages:
    #                 sender = msg["sender"]
    #                 text = msg["text"]
    #                 restored_memory.chat_memory.add_message(
    #                     AIMessage(content=text) if sender == "assistant" else HumanMessage(content=text)
    #                 )

    #             user_conversations[memory_key] = ConversationChain(
    #                 llm=llm_chatgpt,
    #                 prompt=prompt_chatgpt,
    #                 input_key="user_input",
    #                 memory=restored_memory
    #             )

    #     # Fetch the conversation chain
    #     conversation_chain = user_conversations[memory_key]

    #     # Generate assistant response
    #     assistant_response = conversation_chain.run(user_input=user_input)

    #     # Save the conversation to the database
    #     chat, _ = ChatBackup.objects.get_or_create(
    #         user=user, chat_id=chat_id, defaults={"title": "User Chat", "messages": []}
    #     )
    #     chat.messages.append({"sender": "user", "text": user_input, "timestamp": datetime.datetime.now().isoformat()})
    #     chat.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": datetime.datetime.now().isoformat()})
    #     chat.save()

    #     return Response({"response": assistant_response, "chat_id": chat_id})


    def handle_chat(self, request):
        user_input = request.data.get("message", "").strip()
        user_id = request.data.get("user_id", "default_user")
        chat_id = request.data.get("chat_id")  # Accept chat_id from frontend

        # Validate input
        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # Create a composite key to separate user and chat-specific memory
        memory_key = f"{user_id}_{chat_id}"

        # Generate a new chat_id if not provided (new chat)
        if not chat_id:
            chat_id = str(uuid.uuid4())  # New chat resets memory
            print(f"[DEBUG] New chat created with chat_id: {chat_id}")

            # Derive dynamic chat title from the first user input
            chat_title = user_input[:50]  # Use the first 50 characters of the input
            print(f"[DEBUG] New chat title: {chat_title}")

            # Initialize fresh conversation memory for a new chat
            user_conversations[memory_key] = ConversationChain(
                llm=llm_chatgpt,
                prompt=prompt_chatgpt,
                input_key="user_input",
                memory=ConversationBufferMemory()
            )

            # Save the new chat with dynamic title
            ChatBackup.objects.create(
                user=user, chat_id=chat_id, title=chat_title, messages=[]
            )
        else:
            # Retrieve or create a chat session
            chat, created = ChatBackup.objects.get_or_create(
                user=user, chat_id=chat_id, defaults={"title": user_input[:50], "messages": []}
            )

            if not created and memory_key not in user_conversations:
                # Restore memory only if this is an existing chat
                print(f"[DEBUG] Restoring memory for chat_id: {chat_id}")
                restored_memory = ConversationBufferMemory()
                for msg in chat.messages:
                    sender = msg["sender"]
                    text = msg["text"]
                    restored_memory.chat_memory.add_message(
                        AIMessage(content=text) if sender == "assistant" else HumanMessage(content=text)
                    )

                user_conversations[memory_key] = ConversationChain(
                    llm=llm_chatgpt,
                    prompt=prompt_chatgpt,
                    input_key="user_input",
                    memory=restored_memory
                )

        # Fetch the conversation chain
        conversation_chain = user_conversations[memory_key]

        # Generate assistant response
        assistant_response = conversation_chain.run(user_input=user_input)

        # Save the conversation to the database
        chat = ChatBackup.objects.get(chat_id=chat_id)
        chat.messages.append({"sender": "user", "text": user_input, "timestamp": datetime.datetime.now().isoformat()})
        chat.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": datetime.datetime.now().isoformat()})
        chat.save()

        return Response({"response": assistant_response, "chat_id": chat_id})









    # def save_chat_to_db(self, user, chat_id, user_message, assistant_response):
    #     """
    #     Save the user message and assistant response to the ChatBackup model.
    #     """
    #     try:
    #         # Fetch or create the chat record
    #         chat, created = ChatBackup.objects.get_or_create(
    #             user=user,
    #             chat_id=chat_id,
    #             defaults={
    #                 "title": "User Chat",
    #                 "messages": []
    #             }
    #         )
    #         # Append the user and assistant messages
    #         messages = chat.messages or []
    #         messages.append({"sender": "user", "text": user_message, "timestamp": datetime.datetime.now().isoformat()})
    #         messages.append({"sender": "assistant", "text": assistant_response, "timestamp": datetime.datetime.now().isoformat()})
    #         # Update the chat record
    #         chat.messages = messages
    #         chat.save()
    #         print(f"[DEBUG] Chat saved successfully for user {user.id} and chat_id {chat_id}.")
    #     except Exception as e:
    #         print(f"[ERROR] Failed to save chat: {str(e)}")


    def process_schema_confirmation(self, user_input, user_id):
        print("[DEBUG] Processing schema confirmation for user:", user_id)
        conversation_chain = user_conversations[user_id]
        uploaded_file_info = user_schemas[user_id][0]
        suggestions = uploaded_file_info['suggestions']

        if 'yes' in user_input.lower():
            print("[DEBUG] User confirmed suggested schema.")
            user_confirmations[user_id] = {
                'entity_id_column': suggestions['entity_id_column'],
                'target_column': suggestions['target_column'],
                'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']]
            }
            assistant_response = (
                f"Great! You've confirmed the schema:\n\n"
                f"- Entity ID Column: {suggestions['entity_id_column']}\n"
                f"- Target Column: {suggestions['target_column']}\n"
                f"- Feature Columns: {', '.join(suggestions['feature_columns'])}\n\n"
                "You can now proceed to generate the notebook."
            )
            show_generate_notebook = True
        else:
            adjusted_columns = parse_user_adjustments(user_input, uploaded_file_info)
            if adjusted_columns:
                user_confirmations[user_id] = adjusted_columns
                assistant_response = (
                    f"Thanks for the corrections! The updated schema is:\n\n"
                    f"- Entity ID Column: {adjusted_columns['entity_id_column']}\n"
                    f"- Target Column: {adjusted_columns['target_column']}\n"
                    f"- Feature Columns: {', '.join([col['column_name'] for col in adjusted_columns['feature_columns']])}\n\n"
                    "You can now proceed to generate the notebook."
                )
                show_generate_notebook = True
            else:
                print("[DEBUG] Could not find adjusted columns in the dataset.")
                assistant_response = (
                    "I couldn't find those columns in the dataset. Please specify valid column names for the Entity ID and Target columns."
                )
                show_generate_notebook = False

        conversation_chain.memory.chat_memory.messages.append(AIMessage(content=assistant_response))
        return assistant_response, show_generate_notebook

    def reset_conversation(self, request):
        user_id = request.data.get("user_id", "default_user")
        print("[DEBUG] Resetting conversation for user:", user_id)
        if user_id in user_conversations:
            del user_conversations[user_id]
        if user_id in user_schemas:
            del user_schemas[user_id]
        if user_id in user_confirmations:
            del user_confirmations[user_id]
        if user_id in user_notebook_flags:
            del user_notebook_flags[user_id]
        if user_id in user_notebooks:
            del user_notebooks[user_id]
        return Response({"message": "Conversation reset successful."})

    def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
        schema = uploaded_file['schema']
        target_column = uploaded_file['suggestions']['target_column']
        entity_id_column = uploaded_file['suggestions']['entity_id_column']
        feature_columns = uploaded_file['suggestions']['feature_columns']
        schema_text = (
            f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
            "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
            "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
            f"Suggested Target Column: {target_column or 'None'}\n"
            f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
            f"Suggested Feature Columns: {', '.join(feature_columns)}\n\n"
            "Please confirm:\n"
            "- Is the Target Column correct?\n"
            "- Is the Entity ID Column correct?\n"
            "(Reply 'yes' to confirm or provide the correct column names in the format 'Entity ID Column: <column_name>, Target Column: <column_name>')"
        )
        return schema_text

    # def generate_notebook(self, request):
    #     user_id = request.data.get("user_id", "default_user")
    #     print("[DEBUG] Generating notebook for user:", user_id)

    #     if user_id not in user_confirmations:
    #         print("[ERROR] Schema not confirmed yet.")
    #         return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

    #     confirmation = user_confirmations[user_id]
    #     entity_id_column = confirmation['entity_id_column']
    #     target_column = confirmation['target_column']
    #     feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

    #     if user_id in user_schemas:
    #         uploaded_file_info = user_schemas[user_id][0]
    #         table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
    #         sanitized_table_name = self.sanitize_identifier(table_name_raw)
    #         file_url = uploaded_file_info.get('file_url')
    #     else:
    #         print("[ERROR] Uploaded file info not found.")
    #         return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

    #     columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

    #     if not self.validate_column_exists(entity_id_column, columns_list):
    #         return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

    #     if not self.validate_column_exists(target_column, columns_list):
    #         return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

    #     # Generate notebooks (as before)
    #     notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, sanitized_table_name, columns_list)
    #     notebook_features = self.create_features_notebook(feature_columns, sanitized_table_name, columns_list)

    #     notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
    #     notebook_features_sanitized = self.sanitize_notebook(notebook_features)

    #     notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
    #     notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

    #     user_notebooks[user_id] = {
    #         'entity_target_notebook': notebook_entity_target_json,
    #         'features_notebook': notebook_features_json
    #     }

    #     print("[DEBUG] Notebooks generated successfully for user:", user_id)
    #     print("[DEBUG] Sending data to DataForAutomationAPI...")

    #     # Hardcoded user_id and chat_id for now
    #     user_id_for_payload = "12"
    #     chat_id_for_payload = "5"

    #     # If dynamic approach was desired (commented out):
    #     # user_id_for_payload = user_id  # This could come from request.user.id if authenticated
    #     # chat_id_for_payload = user_schemas[user_id][0].get("chat_id")  # if we stored it earlier

    #     # Prepare payload for DataForAutomationAPI with user_id and chat_id now included
    #     payload = {
    #         "file_url": file_url,
    #         "entity_column": entity_id_column,
    #         "target_column": target_column,
    #         "features": feature_columns,
    #         "user_id": user_id_for_payload,    # Hardcoded user_id for now
    #         "chat_id": chat_id_for_payload     # Hardcoded chat_id for now
    #     }

    #     # Store the payload in user_schemas for later use
    #     user_schemas[user_id][0]["automation_payload"] = payload

    #     print("[DEBUG] Payload prepared for DataForAutomationAPI:", payload)

    #     # Note: The actual POST call to DataForAutomationAPI is commented out as per the FYI
    #     return Response({
    #         "message": "Notebooks generated and data passed to DataForAutomationAPI successfully.",
    #         "notebooks": user_notebooks[user_id]
    #     }, status=status.HTTP_200_OK)



    # def generate_notebook(self, request):
    #     user_id = request.data.get("user_id", "default_user")
    #     print("[DEBUG] Generating notebook for user:", user_id)

    #     if user_id not in user_confirmations:
    #         print("[ERROR] Schema not confirmed yet.")
    #         return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

    #     confirmation = user_confirmations[user_id]
    #     entity_id_column = confirmation['entity_id_column']
    #     target_column = confirmation['target_column']
    #     feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

    #     if user_id in user_schemas:
    #         uploaded_file_info = user_schemas[user_id][0]
    #         table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
    #         sanitized_table_name = self.sanitize_identifier(table_name_raw)
    #         file_url = uploaded_file_info.get('file_url')
    #     else:
    #         print("[ERROR] Uploaded file info not found.")
    #         return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

    #     columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

    #     if not self.validate_column_exists(entity_id_column, columns_list):
    #         return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

    #     if not self.validate_column_exists(target_column, columns_list):
    #         return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

    #     # Generate notebooks as before (code omitted for brevity)
    #     notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, sanitized_table_name, columns_list)
    #     notebook_features = self.create_features_notebook(feature_columns, sanitized_table_name, columns_list)

    #     notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
    #     notebook_features_sanitized = self.sanitize_notebook(notebook_features)

    #     notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
    #     notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

    #     user_notebooks[user_id] = {
    #         'entity_target_notebook': notebook_entity_target_json,
    #         'features_notebook': notebook_features_json
    #     }

    #     print("[DEBUG] Notebooks generated successfully for user:", user_id)
    #     print("[DEBUG] Preparing data for the frontend...")

    #     # Hardcoded user_id and chat_id for now
    #     user_id_for_payload = "18"
    #     chat_id_for_payload = "2"

    #     # Now we return the automation-related details directly in the response
    #     # so that the frontend can use them when it decides to run the training/prediction.
    #     response_data = {
    #         "message": "Notebooks generated successfully. Use the returned data to start training when ready.",
    #         "notebooks": user_notebooks[user_id],

    #         # Return the information needed for the data pipeline endpoints
    #         "file_url": file_url,
    #         "entity_column": entity_id_column,
    #         "target_column": target_column,
    #         "features": feature_columns,
    #         # Hardcoded identifiers for now
    #         "user_id": user_id_for_payload,
    #         "chat_id": chat_id_for_payload
    #     }
    #     print("[DEBUG] Data prepared for the frontend:", response_data)

    #     return Response(response_data, status=status.HTTP_200_OK)

    def generate_notebook(self, request):
        # Retrieve user_id and chat_id from the request payload
        user_id = request.data.get("user_id")
        chat_id = request.data.get("chat_id")
        print("[DEBUG] Generating notebook for user_id:", user_id, "and chat_id:", chat_id)

        # Validate input
        if not user_id or not chat_id:
            print("[ERROR] user_id or chat_id missing in the request.")
            return Response({"error": "user_id and chat_id are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure the user exists
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            print(f"[ERROR] User with id {user_id} not found.")
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        # Ensure the chat exists
        try:
            chat = ChatBackup.objects.get(chat_id=chat_id, user=user)
        except ChatBackup.DoesNotExist:
            print(f"[ERROR] Chat with id {chat_id} not found for user {user_id}.")
            return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)

        # Ensure schema confirmation exists for the user
        if user_id not in user_confirmations:
            print("[ERROR] Schema not confirmed yet.")
            return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

        confirmation = user_confirmations[user_id]
        entity_id_column = confirmation['entity_id_column']
        target_column = confirmation['target_column']
        feature_columns = [col['column_name'] for col in confirmation['feature_columns']]

        # Get uploaded file info
        if user_id in user_schemas:
            uploaded_file_info = user_schemas[user_id][0]
            table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
            sanitized_table_name = self.sanitize_identifier(table_name_raw)
            file_url = uploaded_file_info.get('file_url')
        else:
            print("[ERROR] Uploaded file info not found.")
            return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

        columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

        # Validate columns
        if not self.validate_column_exists(entity_id_column, columns_list):
            return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)
        if not self.validate_column_exists(target_column, columns_list):
            return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

        # Generate notebooks
        notebook_entity_target = self.create_entity_target_notebook(entity_id_column, target_column, sanitized_table_name, columns_list)
        notebook_features = self.create_features_notebook(feature_columns, sanitized_table_name, columns_list)

        notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
        notebook_features_sanitized = self.sanitize_notebook(notebook_features)

        notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
        notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

        user_notebooks[user_id] = {
            'entity_target_notebook': notebook_entity_target_json,
            'features_notebook': notebook_features_json
        }

        print("[DEBUG] Notebooks generated successfully for user:", user_id)
        print("[DEBUG] Returning data to the frontend...")

        response_data = {
            "message": "Notebooks generated successfully.",
            "notebooks": user_notebooks[user_id],
            "file_url": file_url,
            "entity_column": entity_id_column,
            "target_column": target_column,
            "features": feature_columns,
            "user_id": user_id,
            "chat_id": chat_id,
        }

        print("[DEBUG] Response data prepared:", response_data)
        return Response(response_data, status=status.HTTP_200_OK)



    def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
        print("[DEBUG] Triggering Glue update for table:", table_name)
        glue = get_glue_client()

        unique_id = file_key.split('/')[1]
        s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/{unique_id}/"
        glue_table_name = self.sanitize_identifier(os.path.splitext(table_name)[0])

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
                DatabaseName=ATHENA_SCHEMA_NAME,
                TableInput={
                    'Name': glue_table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            print("[DEBUG] Glue table updated successfully:", glue_table_name)
        except glue.exceptions.EntityNotFoundException:
            print("[DEBUG] Glue table not found, creating a new one:", glue_table_name)
            glue.create_table(
                DatabaseName=ATHENA_SCHEMA_NAME,
                TableInput={
                    'Name': glue_table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            print("[DEBUG] Glue table created successfully:", glue_table_name)

        user_id = self.get_user_id_from_file_key(file_key)
        if user_id in user_schemas:
            user_schemas[user_id][0]["glue_table_name"] = glue_table_name
            print(f"[DEBUG] Stored Glue table name '{glue_table_name}' for user '{user_id}'.")

        base_timeout = 80
        additional_timeout_per_mb = 5
        dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
        self.wait_for_table_creation(glue_table_name, timeout=dynamic_timeout)

    def get_user_id_from_file_key(self, file_key: str) -> str:
        try:
            return file_key.split("/")[1]
        except IndexError:
            print("[WARNING] Unable to extract user ID, defaulting to 'default_user'.")
            return "default_user"

    def sanitize_identifier(self, name):
        return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())

    def validate_column_exists(self, column_name, columns_list):
        print("[DEBUG] Validating column existence:", column_name)
        print("[DEBUG] Available columns:", columns_list)
        norm_col = normalize_column_name(column_name)
        norm_list = [normalize_column_name(c) for c in columns_list]
        if norm_col in norm_list:
            return True
        else:
            print("[DEBUG] Column not found after normalization:", norm_col)
            return False

    def sanitize_notebook(self, nb):
        def sanitize(obj):
            if isinstance(obj, dict):
                for k in obj:
                    obj[k] = sanitize(obj[k])
                return obj
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                else:
                    return obj
            else:
                return obj
        sanitize(nb)
        return nb

    def create_entity_target_notebook(self, entity_id_column, target_column, table_name, columns_list):
        print("[DEBUG] Creating entity-target notebook...")
        nb = new_notebook()
        cells = []
        cells.append(new_markdown_cell("Core Set"))

        sanitized_entity_id_column = self.sanitize_identifier(entity_id_column)
        sanitized_target_column = self.sanitize_identifier(target_column)
        sql_query_entity_target = (
            f"SELECT {sanitized_entity_id_column}, {sanitized_target_column} "
            f"FROM {table_name} LIMIT 10;"
        )

        df_result = execute_sql_query(sql_query_entity_target)
        if df_result.empty:
            error_message = f"No data returned for query: {sql_query_entity_target}"
            cells.append(new_markdown_cell(f"**Error:** {error_message}"))
        else:
            df_result = df_result.replace([np.nan, np.inf, -np.inf], None)
            result_json = df_result.to_dict(orient='records')
            columns = []
            for col in df_result.columns:
                non_null_series = pd.Series([x for x in df_result[col] if x is not None])
                if non_null_series.empty:
                    col_type = "string"
                else:
                    col_type = infer_column_dtype(non_null_series)
                columns.append({'name': col, 'type': col_type})

            text_repr = df_result.head().to_string(index=False)

            code_cell = new_code_cell(sql_query_entity_target)
            code_cell['execution_count'] = 1
            code_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={
                        'application/json': {
                            'rows': result_json,
                            'columns': columns
                        },
                        'text/plain': text_repr
                    },
                    metadata={},
                    execution_count=1
                )
            ]
            cells.append(code_cell)

        nb['cells'] = cells
        return nb

    def create_features_notebook(self, feature_columns, table_name, columns_list):
        print("[DEBUG] Creating features notebook...")
        nb = new_notebook()
        cells = []
        cells.append(new_markdown_cell("Features or Attributes Test"))
        sanitized_features = [self.sanitize_identifier(feature) for feature in feature_columns]

        missing_columns = [feature for feature in sanitized_features if feature not in columns_list]
        if missing_columns:
            error_message = f"The following feature columns do not exist in the dataset: {', '.join(missing_columns)}"
            cells.append(new_markdown_cell(f"**Error:** {error_message}"))
            nb['cells'] = cells
            return nb

        feature_query = (
            f"SELECT\n    " + ",\n    ".join(sanitized_features) +
            f"\nFROM {table_name}\nLIMIT 10;"
        )

        df_result = execute_sql_query(feature_query)
        if df_result.empty:
            error_message = f"No data returned for query: {feature_query}"
            cells.append(new_markdown_cell(f"**Error:** {error_message}"))
        else:
            df_result = df_result.replace([np.nan, np.inf, -np.inf], None)
            result_json = df_result.to_dict(orient='records')
            columns = []
            for col in df_result.columns:
                non_null_series = pd.Series([x for x in df_result[col] if x is not None])
                if non_null_series.empty:
                    col_type = "string"
                else:
                    col_type = infer_column_dtype(non_null_series)
                columns.append({'name': col, 'type': col_type})

            text_repr = df_result.head().to_string(index=False)

            code_cell = new_code_cell(feature_query)
            code_cell['execution_count'] = 1
            code_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={
                        'application/json': {
                            'rows': result_json,
                            'columns': columns
                        },
                        'text/plain': text_repr
                    },
                    metadata={},
                    execution_count=1
                )
            ]
            cells.append(code_cell)

        nb['cells'] = cells
        return nb

    def wait_for_table_creation(self, table_name, timeout):
        import time
        glue_client = get_glue_client()
        start_time = time.time()
        glue_table_ready = False
        athena_table_ready = False

        print("[DEBUG] Waiting for Glue table creation:", table_name)
        while time.time() - start_time < timeout:
            try:
                glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
                print("[DEBUG] Glue table is now available:", table_name)
                glue_table_ready = True
                break
            except glue_client.exceptions.EntityNotFoundException:
                time.sleep(5)
            except Exception as e:
                print("[ERROR] Unexpected error while checking Glue table availability:", e)
                return False

        if not glue_table_ready:
            print(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
            return False

        print("[DEBUG] Checking Athena table availability:", table_name)
        while time.time() - start_time < timeout:
            try:
                query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
                df = execute_sql_query(query)
                if df.empty:
                    print("[DEBUG] Athena recognizes the table (no error), table ready:", table_name)
                    athena_table_ready = True
                    break
                else:
                    print("[DEBUG] Athena table ready with data:", table_name)
                    athena_table_ready = True
                    break
            except Exception as e:
                error_message = str(e)
                if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
                    print("[DEBUG] Still waiting for Athena to recognize table:", table_name)
                    time.sleep(10)
                else:
                    print("[ERROR] Unexpected error while checking Athena table availability:", e)
                    return False

        if not athena_table_ready:
            print(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
            return False

        return True


# ChatListView and MessageListView as provided earlier, unchanged except for comments:
# class ChatListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def get(self, request):
#         # Get chats for the authenticated user
#         chats = Chat.objects.filter(user=request.user)
#         serializer = ChatSerializer(chats, many=True)
#         return Response(serializer.data)

#     def post(self, request):
#         # Create a new chat for the authenticated user
#         data = request.data
#         chat = Chat.objects.create(user=request.user, title=data.get('title'))
#         return Response({'chat_id': chat.chat_id})


# class MessageListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request, chat_id):
#         # Create a new message in the specified chat for the authenticated user
#         try:
#             chat = Chat.objects.get(id=chat_id, user=request.user)
#         except Chat.DoesNotExist:
#             return Response({'error': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND)

#         message = Message.objects.create(
#             chat=chat,
#             sender=request.data.get('sender'),
#             text=request.data.get('text')
#         )
#         serializer = MessageSerializer(message)
#         return Response(serializer.data, status=status.HTTP_201_CREATED)


class ChatHistoryByUserView(APIView):
    """
    API to retrieve chat history for a specific user.
    """

    def get(self, request):
        print("DEBUG: ChatHistoryByUserView GET method called")

        # Retrieve user_id from query parameters
        user_id = request.GET.get('user_id')
        print(f"DEBUG: Received user_id: {user_id}")

        # Validate the input
        if not user_id:
            print("ERROR: user_id is missing in the request")
            return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Query the database for chats belonging to the user_id
            print(f"DEBUG: Querying chats for user_id: {user_id}")
            chats = ChatBackup.objects.filter(user_id=user_id)
            print("chatttttttttttt")
            print(chats)

            if not chats.exists():
                print(f"WARNING: No chats found for user_id: {user_id}")
                return Response(
                    {"error": f"No chats found for the given user_id: {user_id}"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Prepare the response data
            response_data = []
            for chat in chats:
                messages = chat.messages

                # Separate user and assistant messages
                user_messages = [msg for msg in messages if msg.get("sender") == "user"]
                assistant_messages = [msg for msg in messages if msg.get("sender") == "assistant"]

                response_data.append({
                    "chat_id": chat.chat_id,
                    "title": chat.title,
                    "user_messages": user_messages,
                    "assistant_messages": assistant_messages,
                })

            print("DEBUG: Successfully prepared response data")
            return Response(response_data, status=status.HTTP_200_OK)

        except ChatBackup.DoesNotExist:
            print(f"ERROR: No records found for user_id: {user_id}")
            return Response(
                {"error": f"No chat records found for user_id={user_id}"},
                status=status.HTTP_404_NOT_FOUND,
            )

        except Exception as e:
            print(f"ERROR: Unexpected error occurred: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
