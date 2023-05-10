from django.shortcuts import get_list_or_404, get_object_or_404, Http404
from django.core.files import File
from django.http import FileResponse
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.viewsets import GenericViewSet
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema, no_body
from drf_yasg.openapi import Schema, TYPE_FILE
from django.contrib.auth.models import User

from . import serializers as s
from .models import Object3D
from .pytorch import inference


class Object3DModelViewSet(GenericViewSet):

    queryset = Object3D.objects.all()

    def get_serializer_class(self):
        if self.action == "create_initial":
            return s.Object3DCreation
        elif self.action == "toggle_effect":
            return s.Object3DToggleEffectSerializer
        elif self.action == "list":
            return s.Object3DSerializer
        raise Http404

    @swagger_auto_schema(method="GET", request_body=no_body, responses={200: Schema(type=TYPE_FILE)})
    @action(methods=["GET"], detail=True)
    def retrieve(self, request, username, name):  # NOQA
        instance = self.get_object()
        if instance.file:
            file_handle = instance.file.open()
            response = FileResponse(file_handle, content_type='whatever')
            response['Access-Control-Allow-Origin'] = '*'  # CORS
            response['Content-Length'] = instance.file.size
            response['Content-Disposition'] = 'attachment; filename="%s"' % instance.file.name
            return response
        raise Http404

    @action(methods=["GET"], detail=False)
    def list(self, request, username):  # NOQA
        queryset = self.filter_queryset(self.get_queryset())
        result = self.get_serializer(queryset, many=True).data
        for obj in result:
            obj["thumbnail"] = request.build_absolute_uri(obj["thumbnail"])
            obj["file"] = request.build_absolute_uri(obj["file"])
        return Response(result)

    @swagger_auto_schema(method='post', request_body=no_body, responses={204: 'success'})
    @action(methods=["POST"], detail=True)
    def destroy(self, request, username, name):  # NOQA
        self.get_object().delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(methods=["POST"], detail=False)
    def create_initial(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        queryset = Object3D.objects.filter(user__username=self.kwargs["username"], name=serializer.data["name"])
        if queryset.exists():
            try:
                instance = queryset.get()
            except Object3D.MultipleObjectsReturned:
                raise ValidationError("Multiple objects with the same name")
        else:
            instance = Object3D(name=serializer.data["name"], description=serializer.data["description"])
            instance.user = get_object_or_404(User, username=self.kwargs["username"])
        infer_result = inference(serializer.data["name"], serializer.data["text"])
        instance.with_effect_file = File(infer_result.file, name="{}_1.glb".format(instance.name))
        instance.with_effect_thumbnail = File(infer_result.thumbnail, name="{}_1.png".format(instance.name))
        instance.without_effect_file = File(infer_result.file, name="{}_0.glb".format(instance.name))
        instance.without_effect_thumbnail = File(infer_result.thumbnail, name="{}_0.png".format(instance.name))
        instance.save()
        result = dict(serializer.data)
        result["thumbnail_uri"] = request.build_absolute_uri(instance.thumbnail_uri)
        return Response(result, status=status.HTTP_201_CREATED)

    @action(methods=["GET"], detail=True)
    def toggle_effect(self, request, *args, **kwargs):
        instance = get_object_or_404(
                self.get_queryset(),
                user__username=self.kwargs["username"],
                name=self.kwargs["name"]
            )
        instance.toggle = not instance.toggle
        instance.save()
        data = {
            "toggle": instance.toggle,
            "thumbnail_uri": request.build_absolute_uri(instance.thumbnail_uri),
        }
        return Response(data, status=status.HTTP_200_OK)

    def get_object(self):
        obj = get_object_or_404(
            self.get_queryset(),
            user__username=self.kwargs["username"],
            name=self.kwargs["name"]
        )
        self.check_object_permissions(self.request, obj)
        return obj

    def filter_queryset(self, queryset):
        if self.action == "list":
            return get_list_or_404(queryset, user__username=self.kwargs["username"])
        return super().filter_queryset(queryset)
