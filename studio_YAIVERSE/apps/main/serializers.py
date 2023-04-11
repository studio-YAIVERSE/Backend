from rest_framework import serializers

from .models import Object3D


class Object3DSerializer(serializers.ModelSerializer):

    class Meta:
        model = Object3D
        fields = ['name', 'description', 'file', 'thumbnail']


__all__ = ['Object3DSerializer']
