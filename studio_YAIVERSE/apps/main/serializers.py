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
    description = serializers.CharField(max_length=256, required=True, allow_blank=True, help_text="3D Object 설명")
    text = serializers.CharField(required=True, allow_blank=False, help_text="3D Object 생성용 prompt text")

    def get_infer_target(self):
        return self.data["text"]


class Object3DCreationByImage(serializers.Serializer):
    name = serializers.CharField(max_length=32, required=True, help_text="3D Object 이름")
    description = serializers.CharField(max_length=256, required=True, allow_blank=True, help_text="3D Object 설명")
    image = serializers.CharField(required=True, allow_blank=False, help_text="3D Object 생성용 base64-encoded 이미지")

    def get_infer_target(self):
        import io
        import base64
        return io.BytesIO(base64.b64decode(self.data["image"]))
