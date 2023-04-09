from rest_framework.permissions import AllowAny
from drf_yasg.views import get_schema_view
from drf_yasg import openapi


# API document generation with drf-yasg
def make_schema_view(urlpatterns):
    return get_schema_view(
        openapi.Info(
            title="studio-YAIVERSE API",
            default_version='v1',
            description="studio-YAIVERSE backend APIs",
            contact=openapi.Contact(email="dhakim@yonsei.ac.kr"),
            license=openapi.License(name="YAI: Yonsei AI @ Yonsei University"),
        ),
        validators=['flex'],
        public=True,
        permission_classes=[AllowAny],
        patterns=urlpatterns,
    )
