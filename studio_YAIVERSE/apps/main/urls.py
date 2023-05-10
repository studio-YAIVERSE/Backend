from django.urls import path
from .views import Object3DModelViewSet


urlpatterns = [
    path(
        "get/<str:username>/<str:name>/",
        Object3DModelViewSet.as_view({'get': 'retrieve'}),
        name="object_3d_get"),
    path(
        "list/<str:username>/",
        Object3DModelViewSet.as_view({'get': 'list'}),
        name="object_3d_list"),
    path(
        "delete/<str:username>/<str:name>/",
        Object3DModelViewSet.as_view({'post': 'destroy'}),
        name="object_3d_delete"),
    path(
        "create/<str:username>/",
        Object3DModelViewSet.as_view({'post': 'create_initial'}),
        name="object_3d_create"),
    path(
        "toggle_effect/<str:username>/<str:name>/",
        Object3DModelViewSet.as_view({'get': 'toggle_effect'}),
        name="object_3d_toggle_effect"),
]
