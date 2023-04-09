from django.contrib import admin
from django.urls import path, re_path, include
from .schema import make_schema_view

urlpatterns = [
    path('accounts/', include('studio_YAIVERSE.accounts.urls')),
    path('main/', include('studio_YAIVERSE.main.urls')),
]

schema_view_v1 = make_schema_view(urlpatterns)
urlpatterns += [
    re_path(r'^v1/swagger(?P<format>\.json|\.yaml)/$', schema_view_v1.without_ui(cache_timeout=0), name='schema-json'),
    re_path(r'^v1/swagger/$', schema_view_v1.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    re_path(r'^v1/redoc/$', schema_view_v1.with_ui('redoc', cache_timeout=0), name='schema-redoc-v1'),
]

urlpatterns += [
    path('admin/', admin.site.urls)
]
