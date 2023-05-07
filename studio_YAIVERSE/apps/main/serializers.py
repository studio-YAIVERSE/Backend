from rest_framework import serializers

from .models import Object3D


class Object3DSerializer(serializers.ModelSerializer):

    class Meta:
        model = Object3D
        fields = ['name', 'description', 'file', 'thumbnail']


class Object3DCreation(serializers.Serializer):  # NOQA

    name = serializers.CharField(max_length=32, required=True, help_text="3D Object 이름")
    description = serializers.CharField(max_length=256, allow_blank=True, help_text="3D Object 설명")
    text = serializers.CharField(help_text="3D Object 생성 prompt")
    thumbnail_uri = serializers.URLField(allow_blank=True, help_text="3D Object thumbnail uri (빈 칸으로 두면 자동 생성됨)")


class Object3DRetrieve(serializers.Serializer):  # NOQA

    uri = serializers.CharField()
