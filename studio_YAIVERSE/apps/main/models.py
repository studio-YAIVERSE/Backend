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
    toggle = models.BooleanField(
        default=False,
        verbose_name="3D Object 효과 적용 여부",
        help_text="3D Object 효과 적용 여부 입니다.",
    )

    with_effect_file = models.FileField(
        upload_to=file_upload_path,
        blank=True,
        verbose_name="3D Object with 효과 파일",
        help_text="3D Object with 효과 파일 입니다.",
    )
    with_effect_thumbnail = models.ImageField(
        upload_to=file_upload_path,
        blank=True,
        verbose_name="3D Object with 효과 썸네일",
        help_text="3D Object with 효과 썸네일 입니다.",
    )
    without_effect_file = models.FileField(
        upload_to=file_upload_path,
        blank=True,
        verbose_name="3D Object without 효과 파일",
        help_text="3D Object without 효과 파일 입니다.",
    )
    without_effect_thumbnail = models.ImageField(
        upload_to=file_upload_path,
        blank=True,
        verbose_name="3D Object without 효과 썸네일",
        help_text="3D Object without 효과 썸네일 입니다.",
    )

    file = property(lambda self: self.with_effect_file if self.toggle else self.without_effect_file)
    thumbnail = property(lambda self: self.with_effect_thumbnail if self.toggle else self.without_effect_thumbnail)

    file_uri = property(lambda self: self.file.url)
    thumbnail_uri = property(lambda self: self.thumbnail.url)

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
