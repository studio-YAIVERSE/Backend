from rest_framework import serializers

from .models import Object3D


class Object3DSerializer(serializers.ModelSerializer):

    class Meta:
        model = Object3D
        fields = ['name', 'description', 'file', 'thumbnail']


class Object3DCreation(serializers.Serializer):

    name = serializers.CharField(max_length=32, required=True, help_text="3D Object 이름")
    description = serializers.CharField(max_length=256, allow_blank=True, help_text="3D Object 설명")
    text = serializers.CharField(help_text="3D Object 생성 prompt")

    def create(self, validated_data):
        raise NotImplementedError

    def update(self, instance, validated_data):
        raise NotImplementedError


__all__ = ['Object3DSerializer']
