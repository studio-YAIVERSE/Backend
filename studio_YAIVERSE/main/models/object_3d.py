from django.db import models
from django.contrib.auth.models import User
from .base import BaseModel
from ..common import file_upload_path


class Object3D(BaseModel):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, verbose_name='계정'
    )
    name = models.CharField(
        max_length=32,
        unique=True,
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
        null=True,
        verbose_name="3D Object 파일",
        help_text="3D Object 파일 입니다.",
    )
    thumbnail = models.ImageField(
        upload_to=file_upload_path,
        null=True,
        verbose_name="3D Object 썸네일",
        help_text="3D Object 썸네일 입니다.",
    )

    class Meta:
        verbose_name = '3D 오브젝트'
        verbose_name_plural = '3D 오브젝트들'
        ordering = ['-created_at']

    def __str__(self):
        return f"Todo-{self.user.username}-{self.created_at}"


__all__ = ['Object3D']
