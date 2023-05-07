from django.shortcuts import get_list_or_404, get_object_or_404, Http404, resolve_url
from django.core.files import File
from django.http import FileResponse
from rest_framework.viewsets import GenericViewSet, mixins
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from django.contrib.auth.models import User

from .models import Object3D
from .serializers import Object3DSerializer, Object3DCreation, Object3DRetrieve

from .pytorch import inference


class Object3DModelViewSet(mixins.ListModelMixin, mixins.CreateModelMixin, GenericViewSet):

    queryset = Object3D.objects.all()
    serializer_class = Object3DSerializer

    def get_serializer_class(self):
        if self.action == "retrieve":
            return Object3DRetrieve
        elif self.action == "create":
            return Object3DCreation
        else:  # default
            return self.serializer_class

    def filter_queryset(self, queryset):
        if self.action == "list":
            return get_list_or_404(queryset, user__username=self.kwargs["username"])
        return super().filter_queryset(queryset)

    @action(detail=True)
    def retrieve(self, request, username, name) -> FileResponse:
        instance = get_object_or_404(self.get_queryset(), user__username=username, name=name)
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

    def perform_create(self, serializer):
        if Object3D.objects.filter(user__username=self.kwargs["username"], name=serializer.data["name"]).exists():
            raise ValidationError("Already exists")
        infer_result = inference(serializer.data["name"], serializer.data["text"])
        instance = Object3D(name=serializer.data["name"], description=serializer.data["description"])
        instance.user = get_object_or_404(User, username=self.kwargs["username"])
        instance.file = File(infer_result.file, name="{}.glb".format(instance.name))
        instance.thumbnail = File(infer_result.thumbnail, name="{}.png".format(instance.name))
        instance.save()
        serializer.thumbnail_uri = resolve_url(instance.thumbnail.url)
        print(serializer.thumbnail_uri)

    @action(methods=["POST"], detail=False)
    def create(self, request, username):  # Inference
        return super().create(request, username)
