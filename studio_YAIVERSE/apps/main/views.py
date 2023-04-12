from django.shortcuts import get_list_or_404, get_object_or_404, redirect, Http404
from django.core.files import File
from rest_framework.viewsets import GenericViewSet, mixins
from rest_framework.decorators import action
from django.contrib.auth.models import User

from .models import Object3D
from .serializers import Object3DSerializer, Object3DCreation

from .pytorch import inference


class Object3DModelViewSet(mixins.ListModelMixin, mixins.CreateModelMixin, GenericViewSet):

    queryset = Object3D.objects.all()
    serializer_class = Object3DSerializer
    create_serializer_class = Object3DCreation

    def get_serializer_class(self):
        if self.action == "retrieve":
            return
        elif self.action == "create":
            return Object3DCreation
        else:  # default
            return self.serializer_class

    def filter_queryset(self, queryset):
        if self.action == "list":
            return get_list_or_404(queryset, user__username=self.kwargs["username"])
        return super().filter_queryset(queryset)

    @action(detail=True)
    def retrieve(self, request, username, name):
        object_3d = get_object_or_404(
            self.get_queryset(),
            user__username=username,
            name=name
        )
        if object_3d.file:
            return redirect(object_3d.file.url)
        else:
            raise Http404("No file")

    @action(detail=False)
    def list(self, request, username):
        return super().list(request, username)

    def perform_create(self, serializer):
        inference_result = inference(serializer.data["name"])
        instance = Object3D(name=serializer.data["name"], description=serializer.data["description"])
        instance.user = get_object_or_404(User, username=self.kwargs["username"])
        instance.file = File(inference_result, name="{}.jpg".format(instance.name))
        # instance.thumbnail = File(io.BytesIO(b""), name="{}.jpg".format(instance.name))
        instance.save()

    @action(methods=["POST"], detail=False)
    def create(self, request, username):  # Inference
        return super().create(request, username)
