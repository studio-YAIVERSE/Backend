from rest_framework.routers import DefaultRouter

from . import views as v

router = DefaultRouter()
router.register(r'register', v.UserRegisterView, 'User')
urlpatterns = router.get_urls()
