from django.contrib import admin
from django.conf import settings
from django.urls import path, re_path, include


def make_schema_view(patterns):
    from rest_framework.permissions import AllowAny
    from drf_yasg import openapi
    from drf_yasg.views import get_schema_view
    info = openapi.Info(
        title="studio-YAIVERSE API",
        default_version='v1',
        description="studio-YAIVERSE backend APIs",
        contact=openapi.Contact(email="dhakim@yonsei.ac.kr"),
        license=openapi.License(name="YAI: Yonsei AI @ Yonsei University"),
    )
    return get_schema_view(info, validators=['flex'], public=True, permission_classes=[AllowAny], patterns=patterns)


urlpatterns = [
    path('accounts/', include('studio_YAIVERSE.apps.accounts.urls')),
    path('main/', include('studio_YAIVERSE.apps.main.urls')),
]

schema_view_v1 = make_schema_view(urlpatterns)
urlpatterns += [
    re_path(r'^v1/swagger(?P<format>\.json|\.yaml)/$', schema_view_v1.without_ui(cache_timeout=0), name='schema-json'),
    path('v1/swagger/', schema_view_v1.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('v1/redoc/', schema_view_v1.with_ui('redoc', cache_timeout=0), name='schema-redoc-v1'),
]

urlpatterns += [
    path('admin/', admin.site.urls)
]

if settings.DEBUG:
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
