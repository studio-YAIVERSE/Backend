from rest_framework.routers import DefaultRouter

from . import views as v

router = DefaultRouter()
router.register(r'object3d', v.Object3DModelViewSet)

urlpatterns = [
    *router.urls
]
