# Generated by Django 4.1.7 on 2023-03-27 00:47

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
from studio_YAIVERSE.main import common


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Object3D',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True, help_text='데이터가 생성된 날짜입니다.', verbose_name='생성 일시')),
                ('updated_at', models.DateTimeField(auto_now=True, help_text='데이터가 수정된 날짜입니다.', verbose_name='수정 일시')),
                ('name', models.CharField(help_text='3D Object 이름 입니다.', max_length=32, unique=True, verbose_name='3D Object 이름')),
                ('description', models.CharField(blank=True, help_text='3D Object 설명 입니다.', max_length=256, verbose_name='3D Object 설명')),
                ('file', models.FileField(help_text='3D Object 파일 입니다.', null=True, upload_to=common.file_upload_path, verbose_name='3D Object 파일')),
                ('thumbnail', models.ImageField(help_text='3D Object 썸네일 입니다.', null=True, upload_to=common.file_upload_path, verbose_name='3D Object 썸네일')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL, verbose_name='계정')),
            ],
            options={
                'verbose_name': '3D 오브젝트',
                'verbose_name_plural': '3D 오브젝트들',
                'ordering': ['-created_at'],
            },
        ),
    ]
