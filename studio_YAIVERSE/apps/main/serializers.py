from rest_framework import serializers

from .models import Object3D


class Object3DSerializer(serializers.ModelSerializer):

    class Meta:
        model = Object3D
        fields = ['name', 'description', 'toggle', 'file', 'thumbnail']

    file = serializers.SerializerMethodField()
    thumbnail = serializers.SerializerMethodField()

    def get_file(self, obj):  # NOQA
        return obj.file.url

    def get_thumbnail(self, obj):  # NOQA
        return obj.thumbnail.url


class Object3DCreationByText(serializers.Serializer):

    name = serializers.CharField(max_length=32, required=True, help_text="3D Object 이름")
    description = serializers.CharField(max_length=256, allow_blank=True, help_text="3D Object 설명")
    text = serializers.CharField(help_text="3D Object 생성 prompt")


class Object3DCreationByImage(serializers.Serializer):

    name = serializers.CharField(max_length=32, required=True, help_text="3D Object 이름")
    description = serializers.CharField(max_length=256, allow_blank=True, help_text="3D Object 설명")
    image = serializers.ImageField(required=True, help_text="3D Object 생성 target image")
