from django.contrib.auth.models import User
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import ValidationError


# Create your views here.
@api_view(['POST'])
def register(request):
    check_users = User.objects.filter(username=request.data['username']).all()
    if len(check_users) > 0:
        raise ValidationError("User already exists")
    new_user = User(username='username')
    new_user.save()
    return Response({"username": new_user.username}, status=status.HTTP_200_OK)
