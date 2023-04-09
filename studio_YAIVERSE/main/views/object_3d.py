from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.exceptions import ParseError
from rest_framework.response import Response

from ..models import Object3D
from ..serializers import Object3DSerializer


class Object3DModelViewSet(viewsets.ModelViewSet):
    """actions: create, retrieve, update, partial_update, destroy, list"""
    queryset = Object3D.objects.all()
    serializer_class = Object3DSerializer

    # TODO: filter by user???
    # def get_queryset(self):
    #     queryset = self.queryset
    #     user = self.request.data.get('user', None)
    #     if user is None:
    #         raise ParseError('user is required')
    #     queryset = queryset.filter(user=user)
    #     return queryset


__all__ = ['Object3DModelViewSet']


@api_view(['GET'])
def get_object_3d_list(request):
    username = request.data.get('username', None)
    if username is None:
        raise ParseError('username is required')
    object_3d_list = Object3D.objects.filter(user__username=username).all()
    serializer = Object3DSerializer(object_3d_list, many=True)
    return Response(serializer.data)
