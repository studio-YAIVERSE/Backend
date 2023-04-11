from django.shortcuts import get_list_or_404, get_object_or_404, redirect, Http404
from rest_framework.viewsets import mixins, GenericViewSet
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Object3D
from .serializers import Object3DSerializer


class Object3DModelViewSet(
    mixins.ListModelMixin,
    GenericViewSet
):

    queryset = Object3D.objects.all()
    serializer_class = Object3DSerializer

    def get_serializer_class(self):
        if self.action == "retrieve":
            return
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

    @action(methods=["POST"], detail=False)
    def create(self, request, username):  # Inference
        return Response({})
