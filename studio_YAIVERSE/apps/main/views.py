from django.shortcuts import get_list_or_404, get_object_or_404, Http404, resolve_url
from django.core.files import File
from django.http import FileResponse
from rest_framework import status
from rest_framework.viewsets import GenericViewSet, mixins
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from django.contrib.auth.models import User

from . import serializers as s
from .models import Object3D
from .pytorch import inference


class Object3DModelCreationViews(GenericViewSet):

    queryset = Object3D.objects.all()

    def get_serializer_class(self):
        if self.action == "create_initial":
            return s.Object3DCreation
        elif self.action == "toggle_effect":
            return s.Object3DToggleEffectSerializer
        else:
            raise Http404

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
        result["thumbnail_uri"] = resolve_url(instance.thumbnail_uri)
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
            "thumbnail_uri": instance.thumbnail_uri,
        }
        return Response(data, status=status.HTTP_200_OK)


class Object3DModelViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    GenericViewSet
):

    queryset = Object3D.objects.all()

    @action(detail=True)
    def retrieve(self, request, username, name) -> FileResponse:
        instance = self.get_object()
        if instance.file:
            file_handle = instance.file.open()
            response = FileResponse(file_handle, content_type='whatever')
            response['Access-Control-Allow-Origin'] = '*'  # CORS
            response['Content-Length'] = instance.file.size
            response['Content-Disposition'] = 'attachment; filename="%s"' % instance.file.name
            return response
        else:
            raise Http404("No file")

    @action(detail=False)
    def list(self, request, username):
        queryset = self.filter_queryset(self.get_queryset())
        result = self.get_serializer(queryset, many=True).data
        for obj in result:
            obj["thumbnail_uri"] = request.build_absolute_uri(obj["thumbnail_uri"])
            obj["file_uri"] = request.build_absolute_uri(obj["file_uri"])
        return Response(result)

    @action(methods=["POST"], detail=True)
    def destroy(self, request, username, name):
        return super().destroy(request, username, name)

    def get_serializer_class(self):
        if self.action == "retrieve":
            return s.Object3DRetrieve
        elif self.action == "list":
            return s.Object3DSerializer
        else:
            print(self.action, "#"*100)
            raise Http404

    def get_object(self):
        return get_object_or_404(
            self.get_queryset(),
            user__username=self.kwargs["username"],
            name=self.kwargs["name"]
        )

    def filter_queryset(self, queryset):
        if self.action == "list":
            return get_list_or_404(queryset, user__username=self.kwargs["username"])
        return super().filter_queryset(queryset)
