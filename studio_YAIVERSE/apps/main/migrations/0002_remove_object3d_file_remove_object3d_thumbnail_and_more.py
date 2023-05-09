# Generated by Django 4.1.7 on 2023-05-10 02:11

from django.db import migrations, models
import studio_YAIVERSE.apps.main.common


class Migration(migrations.Migration):

    dependencies = [
        ("main", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="object3d",
            name="file",
        ),
        migrations.RemoveField(
            model_name="object3d",
            name="thumbnail",
        ),
        migrations.AddField(
            model_name="object3d",
            name="toggle",
            field=models.BooleanField(
                default=False,
                help_text="3D Object 효과 적용 여부 입니다.",
                verbose_name="3D Object 효과 적용 여부",
            ),
        ),
        migrations.AddField(
            model_name="object3d",
            name="with_effect_file",
            field=models.FileField(
                blank=True,
                help_text="3D Object with 효과 파일 입니다.",
                upload_to=studio_YAIVERSE.apps.main.common.file_upload_path,
                verbose_name="3D Object with 효과 파일",
            ),
        ),
        migrations.AddField(
            model_name="object3d",
            name="with_effect_thumbnail",
            field=models.ImageField(
                blank=True,
                help_text="3D Object with 효과 썸네일 입니다.",
                upload_to=studio_YAIVERSE.apps.main.common.file_upload_path,
                verbose_name="3D Object with 효과 썸네일",
            ),
        ),
        migrations.AddField(
            model_name="object3d",
            name="without_effect_file",
            field=models.FileField(
                blank=True,
                help_text="3D Object without 효과 파일 입니다.",
                upload_to=studio_YAIVERSE.apps.main.common.file_upload_path,
                verbose_name="3D Object without 효과 파일",
            ),
        ),
        migrations.AddField(
            model_name="object3d",
            name="without_effect_thumbnail",
            field=models.ImageField(
                blank=True,
                help_text="3D Object without 효과 썸네일 입니다.",
                upload_to=studio_YAIVERSE.apps.main.common.file_upload_path,
                verbose_name="3D Object without 효과 썸네일",
            ),
        ),
    ]
