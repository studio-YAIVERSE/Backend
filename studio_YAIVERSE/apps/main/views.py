from django.shortcuts import get_object_or_404, Http404
from django.core.files import File
from django.http import FileResponse
from rest_framework.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_204_NO_CONTENT
from rest_framework.decorators import action
from rest_framework.viewsets import GenericViewSet
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema, no_body
from drf_yasg.openapi import Schema, TYPE_FILE, FORMAT_BINARY
from django.contrib.auth.models import User

from .serializers import Object3DSerializer, Object3DCreationByText, Object3DCreationByImage
from .models import Object3D
from .pytorch import inference_text, inference_image


class Object3DModelViewSet(GenericViewSet):

    queryset = Object3D.objects.all()

    def get_serializer_class(self):
        if self.action == "create_by_text":
            return Object3DCreationByText
        elif self.action == "create_by_image":
            return Object3DCreationByImage
        return Object3DSerializer

    @swagger_auto_schema(method="GET", responses={200: Schema(format=FORMAT_BINARY, type=TYPE_FILE)})
    @action(methods=["GET"], detail=True)
    def retrieve(self, request, username, name):  # NOQA
        instance = self.get_object()
        if instance.file:
            response = FileResponse(instance.file.open(), content_type='whatever')
            response['Access-Control-Allow-Origin'] = '*'  # CORS
            response['Content-Length'] = instance.file.size
            response['Content-Disposition'] = 'attachment; filename="%r"' % instance.file.name
            return response
        raise Http404

    @swagger_auto_schema(method="GET", request_body=no_body, responses={200: Object3DSerializer(many=True)})
    @action(methods=["GET"], detail=False)
    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset().filter(user__username=self.kwargs["username"])
        return self.build_response(queryset, many=True, status=HTTP_200_OK)

    @swagger_auto_schema(method="POST", request_body=no_body, responses={204: "success"})
    @action(methods=["POST"], detail=True)
    def destroy(self, request, *args, **kwargs):
        self.get_object().delete()
        return self.build_response(status=HTTP_204_NO_CONTENT)

    @swagger_auto_schema(method="GET", request_body=no_body, responses={200: Object3DSerializer()})
    @action(methods=["GET"], detail=True)
    def toggle_effect(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.toggle = not instance.toggle
        instance.save()
        return self.build_response(instance, status=HTTP_200_OK)

    @swagger_auto_schema(method="POST", request_body=Object3DCreationByText, responses={201: Object3DSerializer()})
    @action(methods=["POST"], detail=False)
    def create_by_text(self, request, *args, **kwargs):
        return self.__create_logic(inference_function=inference_text, inference_refer_key="text")

    @swagger_auto_schema(method="POST", request_body=Object3DCreationByImage, responses={201: Object3DSerializer()})
    @action(methods=["POST"], detail=False)
    def create_by_image(self, request, *args, **kwargs):
        return self.__create_logic(inference_function=inference_image, inference_refer_key="image")

    def __create_logic(self, inference_function, inference_refer_key):
        serializer = self.get_serializer(data=self.request.data)
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
        infer_result = inference_function(serializer.data["name"], serializer.data[inference_refer_key])
        instance.with_effect_file = File(infer_result.file, name="{}_1.glb".format(instance.name))
        instance.with_effect_thumbnail = File(infer_result.thumbnail, name="{}_1.png".format(instance.name))
        instance.without_effect_file = File(infer_result.file, name="{}_0.glb".format(instance.name))
        instance.without_effect_thumbnail = File(infer_result.thumbnail, name="{}_0.png".format(instance.name))
        instance.save()
        return self.build_response(instance, status=HTTP_201_CREATED)

    def get_object(self):
        obj = get_object_or_404(
            self.get_queryset(),
            user__username=self.kwargs["username"],
            name=self.kwargs["name"]
        )
        self.check_object_permissions(self.request, obj)
        return obj

    def build_response(self, instance=None, *, many: bool = False, status: int = None):
        if instance is not None:
            result = Object3DSerializer(instance, many=many).data
            if many:
                for obj in result:
                    obj["thumbnail"] = self.request.build_absolute_uri(obj["thumbnail"])
                    obj["file"] = self.request.build_absolute_uri(obj["file"])
            else:
                result["thumbnail"] = self.request.build_absolute_uri(result["thumbnail"])
                result["file"] = self.request.build_absolute_uri(result["file"])
        else:
            result = None
        return Response(result, status=status)
