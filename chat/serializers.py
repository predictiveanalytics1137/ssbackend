# # # # serializers.py
# # # from rest_framework import serializers
# # # from .models import UploadedFile

# # # from django.conf import settings

# # # # class UploadedFileSerializer(serializers.ModelSerializer):
# # # #     class Meta:
# # # #         model = UploadedFile
# # # #         fields = '__all__'

# # # #     def create(self, validated_data):
# # # #         instance = super().create(validated_data)
# # # #         # Add the full URL to the file
# # # #         instance.file_url = f"{settings.MEDIA_URL}{instance.file.name}"
# # # #         instance.save()
# # # #         return instance


# # # from rest_framework import serializers
# # # from .models import UploadedFile


# # # class UploadedFileSerializer(serializers.ModelSerializer):
# # #     class Meta:
# # #         model = UploadedFile
# # #         fields = '__all__'

# # #     def validate_file(self, file):
# # #         # Ensure the file is not empty
# # #         if file.size == 0:
# # #             raise serializers.ValidationError("Uploaded file is empty.")

# # #         # Ensure the file is a CSV
# # #         if not file.name.endswith('.csv'):
# # #             raise serializers.ValidationError("Only CSV files are allowed.")
        
# # #         return file



# # from rest_framework import serializers
# # from .models import UploadedFile
# # import pandas as pd


# # class UploadedFileSerializer(serializers.ModelSerializer):
# #     class Meta:
# #         model = UploadedFile
# #         fields = '__all__'

# #     def validate_file(self, file):
# #         # Normalize file name (e.g., convert to lowercase, replace spaces)
# #         file.name = file.name.lower().replace(' ', '_')

# #         # Ensure the file is not empty
# #         if file.size == 0:
# #             raise serializers.ValidationError("Uploaded file is empty.")

# #         # Allow both CSV and Excel files
# #         allowed_extensions = ['.csv', '.xlsx']
# #         if not any(file.name.endswith(ext) for ext in allowed_extensions):
# #             raise serializers.ValidationError("Only CSV and Excel files are allowed.")

# #         # Check for duplicate file names
# #         if UploadedFile.objects.filter(name=file.name).exists():
# #             raise serializers.ValidationError("A file with this name already exists.")

# #         # Validate the file content for supported formats
# #         try:
# #             # Ensure the file content is valid and readable
# #             if file.name.endswith('.csv'):
# #                 pd.read_csv(file)  # Load CSV to check validity
# #             elif file.name.endswith('.xlsx'):
# #                 pd.read_excel(file)  # Load Excel file to check validity
# #         except Exception as e:
# #             raise serializers.ValidationError(f"File validation failed: {str(e)}")

# #         return file

# import pandas as pd
# import os
# from rest_framework import serializers
# from .models import UploadedFile


# class UploadedFileSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = UploadedFile
#         fields = '__all__'

#     def validate_file(self, file):
#         print(f"Validating file: {file.name}, Size: {file.size} bytes")

#         # Step 1: Check file size
#         if file.size == 0:
#             print("File validation failed: File is empty.")
#             raise serializers.ValidationError("Uploaded file is empty.")

#         # Step 2: Validate file extension
#         allowed_extensions = ['.csv', '.xlsx']
#         file_extension = os.path.splitext(file.name)[-1].lower()
#         if file_extension not in allowed_extensions:
#             print(f"Unsupported file extension: {file_extension}")
#             raise serializers.ValidationError("Only CSV and Excel files are allowed.")

#         # Step 3: Check for duplicates
#         normalized_name = file.name.lower().replace(' ', '_')
#         if UploadedFile.objects.filter(name=normalized_name).exists():
#             print(f"Duplicate file name: {normalized_name}")
#             raise serializers.ValidationError(f"A file named '{normalized_name}' already exists.")

#         # Step 4: Validate file content
#         try:
#             if file_extension == '.csv':
#                 df = pd.read_csv(file, nrows=10)  # Read first 10 rows to validate
#                 if df.empty or len(df.columns) == 0:
#                     raise serializers.ValidationError("The CSV file is empty or has no valid columns.")
#             elif file_extension == '.xlsx':
#                 try:
#                     pd.read_excel(file, nrows=10)
#                 except ImportError:
#                     raise serializers.ValidationError("The 'openpyxl' library is required to process Excel files.")
#         except Exception as e:
#             print(f"File content validation failed: {str(e)}")
#             raise serializers.ValidationError(f"File content validation failed: {str(e)}")

#         print("File validation successful.")
#         return file



import pandas as pd
import os
from rest_framework import serializers
from .models import UploadedFile


class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = '__all__'

    def validate_file(self, file):
        print(f"Validating file: {file.name}, Size: {file.size} bytes")

        # Step 1: Check file size
        if file.size == 0:
            print("File validation failed: File is empty.")
            raise serializers.ValidationError("Uploaded file is empty.")

        # Step 2: Validate file extension
        allowed_extensions = ['.csv', '.xlsx']
        file_extension = os.path.splitext(file.name)[-1].lower()
        if file_extension not in allowed_extensions:
            print(f"Unsupported file extension: {file_extension}")
            raise serializers.ValidationError("Only CSV and Excel files are allowed.")

        # Step 3: Check for duplicates
        normalized_name = file.name.lower().replace(' ', '_')
        if UploadedFile.objects.filter(name=normalized_name).exists():
            print(f"Duplicate file name: {normalized_name}")
            raise serializers.ValidationError(f"A file named '{normalized_name}' already exists.")

        # Step 4: Validate file content
        try:
            # Read a snippet of the file for debugging
            file.seek(0)
            content_snippet = file.read(100).decode('utf-8', errors='ignore')
            print(f"[DEBUG] Serializer received file content snippet:\n{content_snippet}")
            file.seek(0)  # Reset pointer after reading

            if file_extension == '.csv':
                df = pd.read_csv(file, nrows=10)  # Read first 10 rows to validate
                if df.empty or len(df.columns) == 0:
                    raise serializers.ValidationError("The CSV file is empty or has no valid columns.")
            elif file_extension == '.xlsx':
                try:
                    pd.read_excel(file, nrows=10, engine='openpyxl')
                except ImportError:
                    raise serializers.ValidationError("The 'openpyxl' library is required to process Excel files.")
                except Exception as e:
                    raise serializers.ValidationError(f"Excel file validation failed: {str(e)}")
        except pd.errors.ParserError as e:
            print(f"File content validation failed: {str(e)}")
            raise serializers.ValidationError(f"File content validation failed: {str(e)}")
        except Exception as e:
            print(f"File content validation failed: {str(e)}")
            raise serializers.ValidationError(f"File content validation failed: {str(e)}")

        print("File validation successful.")
        return file








# from rest_framework import serializers
# from .models import Chat, Message

# class MessageSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Message
#         fields = '__all__'

# class ChatSerializer(serializers.ModelSerializer):
#     messages = MessageSerializer(many=True, read_only=True)

#     class Meta:
#         model = Chat
#         fields = '__all__'
