from django.db import models
from django.contrib.auth.models import User
from .common import file_upload_path


class BaseModel(models.Model):
    class Meta:
        abstract = True

    created_at = models.DateTimeField(
        auto_now_add=True,
        blank=True,
        null=False,
        verbose_name="생성 일시",
        help_text="데이터가 생성된 날짜입니다."
    )

    updated_at = models.DateTimeField(
        auto_now=True,
        blank=True,
        null=False,
        verbose_name="수정 일시",
        help_text="데이터가 수정된 날짜입니다."
    )


class Object3D(BaseModel):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, verbose_name='계정'
    )
    name = models.CharField(
        max_length=32,
        verbose_name="3D Object 이름",
        help_text="3D Object 이름 입니다."
    )
    description = models.CharField(
        max_length=256,
        blank=True,
        verbose_name="3D Object 설명",
        help_text="3D Object 설명 입니다."
    )
    file = models.FileField(
        upload_to=file_upload_path,
        blank=True,
        verbose_name="3D Object 파일",
        help_text="3D Object 파일 입니다.",
    )
    thumbnail = models.ImageField(
        upload_to=file_upload_path,
        blank=True,
        verbose_name="3D Object 썸네일",
        help_text="3D Object 썸네일 입니다.",
    )

    class Meta:
        verbose_name = '3D 오브젝트'
        verbose_name_plural = '3D 오브젝트들'
        ordering = ['-created_at']
        constraints = [
            models.UniqueConstraint(fields=["user", "name"], name='user-name constraint')
        ]

    def __str__(self):
        return f"Object3D-{self.user.username}-{self.name}"


__all__ = ['Object3D']
