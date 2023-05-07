from rest_framework.viewsets import mixins, GenericViewSet
from rest_framework import status
from rest_framework.response import Response
from django.contrib.auth.models import User
from .serializers import UserSerializer


class UserRegisterView(mixins.CreateModelMixin, GenericViewSet):
    serializer_class = UserSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if User.objects.filter(username=serializer.initial_data['username']).exists():
            return Response({'username': request.data['username']}, status=status.HTTP_200_OK)
        return super().create(request, *args, **kwargs)
