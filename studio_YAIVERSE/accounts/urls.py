from django.urls import path

from . import views as v

urlpatterns = [
    path('register/', v.register, name='register'),
]
