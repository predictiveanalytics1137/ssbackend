

import os
import datetime
from io import BytesIO
from typing import Any, Dict, List
from langchain.schema import AIMessage, HumanMessage
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
from langchain.memory import ConversationBufferMemory
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
    # model="gpt-3.5-turbo-16k",
     model="gpt-4",
    
    # model = "gpt-4",

    temperature=0.7,
    openai_api_key=OPENAI_API_KEY,
)

# LangChain prompt with memory integration for ChatGPT
# prompt_chatgpt = PromptTemplate(
#     input_variables=["history", "user_input"],
#     template=(
#         "You are a helpful PACX AI assistant. You guide users through defining predictive questions and refining goals.\n"
#         # "If the user uploads a dataset, integrate the schema into the conversation to assist with column identification.\n\n"
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

user_conversations = {}

# Modify the prompt in the existing code
# Adding system instructions to guide the model
prompt_chatgpt = PromptTemplate(
    input_variables=["history", "user_input"],  # Remove system_instructions from here
    template=(
        "You are a helpful PACX AI assistant. Your job is to guide users through defining predictive questions and refining goals. "
        "You must strictly follow the step-by-step process outlined in the prompt. Do not deviate from the steps or answer prematurely. "
        "Wait for the user to confirm all necessary inputs before proceeding further.\n\n"
        "Steps:\n"
        "1. Discuss the Subject they want to predict.\n"
        "2. Confirm the Target Value they want to predict.\n"
        "3. Check if there's a specific time frame for the prediction.\n"
        "4. Reference the dataset schema if available.\n"
        "5. **Once you have confirmed all necessary information with the user, provide a summary of the inputs. At the very end of your summary, include only the phrase 'GENERATE_NOTEBOOK_PROMPT', and nothing else. Do not include 'GENERATE_NOTEBOOK_PROMPT' in any of your responses until all necessary information has been gathered and confirmed with the user.**\n\n"
        "Conversation history: {history}\n"
        "User input: {user_input}\n"
        "Assistant:"
    ),
)
from langchain.schema import SystemMessage

memory = ConversationBufferMemory()
memory.chat_memory.add_message(SystemMessage(content="You are a helpful PACX AI assistant. Follow the steps strictly and assist users with predictive questions."))

conversation_chain_chatgpt = ConversationChain(
    llm=llm_chatgpt,
    prompt=prompt_chatgpt,
    input_key="user_input",
    memory=ConversationBufferMemory(),  # Add memory instance
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

# def suggest_target_column(df: pd.DataFrame) -> Any:
#     """
#     Suggests a target column based on numeric data types.
#     """
#     numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
#     return numeric_cols[0] if len(numeric_cols) > 0 else None

def get_user_specified_target(chat_history: List[Any]) -> str:
    """
    Extracts user-specified target column from chat history if available.
    """
    for message in reversed(chat_history):  # Traverse messages in reverse order to find the latest target
        if isinstance(message, HumanMessage):  # Only check user messages
            if "target column" in message.content.lower():
                # Extract the column name (customize regex as needed)
                import re
                match = re.search(r"target column: (\w+)", message.content, re.IGNORECASE)
                if match:
                    return match.group(1)
    return None



from langchain.schema import HumanMessage

def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
    """
    Suggests a target column based on user input or predictive question.
    """
    # Check if the user specified a target column
    user_target = get_user_specified_target(chat_history)
    if user_target and user_target in df.columns:
        return user_target

    # Use LLM to predict the target column based on a description (if available)
    target_suggestion_prompt = (
        f"The user uploaded a dataset with the following columns: {', '.join(df.columns)}.\n"
        "Based on the context of their predictive question, suggest the best target column."
    )
    response = llm_chatgpt.invoke([HumanMessage(content=target_suggestion_prompt)])
    suggested_column = response.content.strip()
    return suggested_column if suggested_column in df.columns else None


# def suggest_entity_id_column(df: pd.DataFrame) -> Any:
#     """
#     Suggests an entity ID column based on uniqueness.
#     """
#     for col in df.columns:
#         if df[col].is_unique:
#             return col
#     return None

def suggest_entity_id_column(df: pd.DataFrame) -> Any:
    """
    Suggests an entity ID column based on uniqueness and naming conventions.
    """
    likely_id_columns = [col for col in df.columns if "id" in col.lower()]
    for col in likely_id_columns:
        if df[col].nunique() / len(df) > 0.95:  # At least 95% unique values
            return col

    # Fallback: Find any column with >95% unique values
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.95:
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
    uploaded_schema_by_user = {}

    

    def post(self, request):
        """
        Handles POST requests for both chat messages and file uploads.
        Differentiates based on the presence of files in the request.
        """
        action = request.data.get('action', '')
        if action == 'reset':
            return self.reset_conversation(request)
        if "file" in request.FILES:  # If files are present, handle file uploads
            return self.handle_file_upload(request, request.FILES.getlist("file"))

        # Else, handle chat message
        return self.handle_chat(request)
    
    # def post(self, request):
        

    def handle_file_upload(self, request, files: List[Any]):
        """
        Handles multiple file uploads, processes them, uploads to AWS S3, updates AWS Glue, and saves schema in DB.
        After processing, appends schema details to the chat messages.
        """

        files = request.FILES.getlist("file")
        if not files:
            return Response({"error": "No files provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        user_id = request.data.get("user_id", "default_user") 
        
        

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
                            'target_column': suggest_target_column(df, conversation_chain_chatgpt.memory.chat_memory.messages),
                            'entity_id_column': suggest_entity_id_column(df),
                        }
                    })
                    
                else:
                    return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            # Format schema messages to append to assistant conversation
            schema_messages = [self.format_schema_message(uploaded_file) for uploaded_file in uploaded_files_info]
            combined_schema_message = "\n\n".join(schema_messages)
            # print(f"Combined schema message for chat: {combined_schema_message}")  # Debugging statement


            # Store schema for user
            UnifiedChatGPTAPI.uploaded_schema_by_user[user_id] = combined_schema_message

            if hasattr(conversation_chain_chatgpt.memory, "chat_memory"):
                conversation_chain_chatgpt.memory.chat_memory.messages.append(
                    HumanMessage(content=f"Schema for '{file.name}': {combined_schema_message}")
                )


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
    
    # def handle_chat(self, request):
    #     """
    #     Handles user chat messages using ChatGPT.
    #     """
    #     user_input = request.data.get("message", "").strip()
    #     user_id = request.data.get("user_id", "default_user")

    #     if not user_input:
    #         return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

    #     # Fetch schema for the user session
    #     uploaded_schema = UnifiedChatGPTAPI.uploaded_schema_by_user.get(user_id, "")

    #     # Pass schema into the conversation prompt
    #     assistant_response = conversation_chain_chatgpt.run(
    #         user_input=f"{user_input}\n\nUploaded Schema:\n{uploaded_schema}"
    #     )

    #     return Response({
    #         "response": assistant_response
    #     })

    def handle_chat(self, request):
        user_input = request.data.get("message", "").strip()
        user_id = request.data.get("user_id", "default_user")

        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Fetch schema for the user session
        uploaded_schema = UnifiedChatGPTAPI.uploaded_schema_by_user.get(user_id, "")

        # Get or create conversation chain for the user
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

        assistant_response = conversation_chain.run(
            user_input=f"{user_input}\n\nUploaded Schema:\n{uploaded_schema}"
        )

        return Response({
            "response": assistant_response
        })

    def reset_conversation(self, request):
        user_id = request.data.get("user_id", "default_user")
        # Remove user's conversation chain
        if user_id in user_conversations:
            del user_conversations[user_id]
        # Remove user's uploaded schema
        if user_id in UnifiedChatGPTAPI.uploaded_schema_by_user:
            del UnifiedChatGPTAPI.uploaded_schema_by_user[user_id]
        return Response({"message": "Conversation reset successful."})



    
    


    
    def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
        """
        Formats the schema information to be appended as an assistant message in the chat.
        """
        schema = uploaded_file['schema']
        target_column = uploaded_file['suggestions']['target_column']
        entity_id_column = uploaded_file['suggestions']['entity_id_column']
        schema_text = (
            f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
            "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
            "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
            f"Target Column Suggestion: {target_column or 'None provided'}\n"
            f"Entity ID Column Suggestion: {entity_id_column or 'None provided'}\n\n"
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



