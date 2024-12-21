# # from rest_framework import serializers
# # from django.contrib.auth.models import User

# # class UserSerializer(serializers.ModelSerializer):
# #     class Meta:
# #         model = User
# #         fields = ['username', '`email`', 'password']
# #         extra_kwargs = {'password': {'write_only': True}}

# #     def create(self, validated_data):
# #         user = User.objects.create_user(
# #             username=validated_data['username'],
# #             email=validated_data['email'],
# #             password=validated_data['password']
# #         )
# #         return user


# # from rest_framework import serializers
# from django.contrib.auth.models import User
# from django.core.exceptions import ValidationError

# # class UserSerializer(serializers.ModelSerializer):
# #     email = serializers.EmailField(required=True)  # Explicitly add the email field

# #     class Meta:
# #         model = User
# #         fields = ['username', 'email', 'password']
# #         extra_kwargs = {'password': {'write_only': True}}

# #     def create(self, validated_data):
# #         if 'email' not in validated_data:
# #             raise ValidationError("Email is required.")
# #         user = User.objects.create_user(
# #             username=validated_data['username'],
# #             email=validated_data['email'],
# #         )
# #         return user




# from rest_framework import serializers
# from django.contrib.auth.models import User

# class UserSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = ['username', 'email', 'password']
#         extra_kwargs = {'password': {'write_only': True}}

#     def create(self, validated_data):
#         user = User(
#             username=validated_data['username'],
#             email=validated_data['email']
#         )
#         user.set_password(validated_data['password'])  # Hash the password
#         user.save()
#         return user



from rest_framework import serializers
from django.contrib.auth.models import User

# Serializer
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        print("Validated data received in serializer:", validated_data)  # Debugging the data
        user = User(
            username=validated_data['username'],
            email=validated_data['email']
        )
        user.set_password(validated_data['password'])  # Hash the password
        print(f"User {user.username} created successfully, password hashed.")
        user.save()
        print(f"User {user.username} saved to the database.")
        return user