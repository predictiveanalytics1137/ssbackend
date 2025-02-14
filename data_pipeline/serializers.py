from rest_framework import serializers
from .models import DemoRequest

# class DemoRequestSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = DemoRequest
#         fields = ['id', 'first_name', 'last_name', 'email', 'company', 'contact']


from rest_framework import serializers
from .models import DemoRequest

class DemoRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = DemoRequest
        fields = ['id', 'first_name', 'last_name', 'email', 'company', 'contact']

    def validate_email(self, value):
        """Ensure the email is unique before saving"""
        if DemoRequest.objects.filter(email=value).exists():
            raise serializers.ValidationError("This email is already registered for a demo.")
        return value
