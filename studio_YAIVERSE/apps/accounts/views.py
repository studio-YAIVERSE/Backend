from rest_framework.viewsets import mixins, GenericViewSet
from .serializers import UserSerializer


class UserRegisterView(mixins.CreateModelMixin, GenericViewSet):
    """actions: create"""
    serializer_class = UserSerializer
