from django.urls import path
from . import views as v


urlpatterns = [
    path(
        "get/<str:username>/<str:name>/",
        v.Object3DModelViewSet.as_view({'get': 'retrieve'}),
        name="object_3d_list"),
    path(
        "list/<str:username>/",
        v.Object3DModelViewSet.as_view({'get': 'list'}),
        name="object_3d_list"
    ),
]
