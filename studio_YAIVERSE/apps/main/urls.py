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
        name="object_3d_list"),
    path(
        "delete/<str:username>/<str:name>/",
        v.Object3DModelViewSet.as_view({'post': 'destroy'}),
        name="object_3d_list"),

    path(
        "create/<str:username>/",
        v.Object3DModelCreationViews.as_view({'post': 'create_initial'}),
        name="object_3d_create"),
    path(
        "toggle_effect/<str:username>/<str:name>/",
        v.Object3DModelCreationViews.as_view({'get': 'toggle_effect'}),
        name="object_3d_create"),

]
