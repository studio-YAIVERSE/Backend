from django.shortcuts import get_list_or_404, get_object_or_404, Http404, resolve_url
from django.core.files import File
from django.http import FileResponse
from rest_framework import status
from rest_framework.viewsets import GenericViewSet, mixins
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.response import Response
from django.contrib.auth.models import User

from .models import Object3D
from .serializers import Object3DSerializer, Object3DCreation, Object3DRetrieve

from .pytorch import inference


class Object3DModelViewSet(
    mixins.ListModelMixin,
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    GenericViewSet
):

    queryset = Object3D.objects.all()
    serializer_class = Object3DSerializer

    @action(detail=True)
    def retrieve(self, request, username, name) -> FileResponse:
        instance = self.get_object()
        if instance.file:
            file_handle = instance.file.open()
            response = FileResponse(file_handle, content_type='whatever')
            response['Content-Length'] = instance.file.size
            response['Content-Disposition'] = 'attachment; filename="%s"' % instance.file.name
            return response
        else:
            raise Http404("No file")

    @action(detail=False)
    def list(self, request, username):
        return super().list(request, username)

    @action(methods=["POST"], detail=False)
    def create(self, request, username):  # Inference
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        result = serializer.data
        result["thumbnail_uri"] = serializer.thumbnail_uri
        return Response(result, status=status.HTTP_201_CREATED, headers=headers)

    @action(methods=["POST"], detail=True)
    def destroy(self, request, username, name):
        return super().destroy(request, username, name)

    def get_serializer_class(self):
        if self.action == "retrieve":
            return Object3DRetrieve
        elif self.action == "create":
            return Object3DCreation
        else:  # default
            return self.serializer_class

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

    def perform_create(self, serializer):
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
        instance.file = File(infer_result.file, name="{}.glb".format(instance.name))
        instance.thumbnail = File(infer_result.thumbnail, name="{}.png".format(instance.name))
        instance.save()
        serializer.thumbnail_uri = resolve_url(instance.thumbnail.url)
