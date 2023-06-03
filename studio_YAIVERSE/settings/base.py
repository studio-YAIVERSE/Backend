"""
Common settings shared by both development and production environments.
"""

import os as _os
import pathlib as _pathlib

# Base Project Directory Definition

BASE_DIR = _pathlib.Path(__file__).resolve().parent.parent.parent


# Pytorch Ops

TORCH_SETTINGS = {
    "BASE_DIR": BASE_DIR,
    "GET3D_PATH": BASE_DIR / 'GET3D',  # Path to GET3D (for import and model initialization)
    "TORCH_ENABLED": bool(int(_os.getenv("TORCH_ENABLED", 1))),  # 0 disables all torch operations
    "TORCH_LOG_LEVEL": int(_os.getenv("TORCH_LOG_LEVEL", 2)),  # 0: silent, 1: call, 2: 1 + process, 3: 2 + nada output
    "TORCH_WARM_UP_ITER": int(_os.getenv("TORCH_WARM_UP_ITER", 10)),  # Number of warm up iterations
    "TORCH_WITHOUT_CUSTOM_OPS_COMPILE": bool(int(_os.getenv("TORCH_WITHOUT_CUSTOM_OPS_COMPILE", 0))),  # without ninja
    "TORCH_DEVICE": _os.getenv("TORCH_DEVICE", "cuda:0"),  # Device to use
    "NADA_WEIGHT_DIR": _os.getenv("NADA_WEIGHT_DIR", BASE_DIR / "weights/get3d_nada"),  # Path of NADA weights
    "CLIP_MAP_PATH": _os.getenv("CLIP_MAP_PATH", BASE_DIR / "weights/clip_map/checkpoint_group.pt"),
    "MODEL_OPTS": {  # Model initialization kwargs which is compatible with script arguments
        'latent_dim': 512,
        'one_3d_generator': True,
        'deformation_multiplier': 1.,
        'use_style_mixing': True,
        'dmtet_scale': 1.,
        'feat_channel': 16,
        'mlp_latent_channel': 32,
        'tri_plane_resolution': 256,
        'n_views': 1,
        'render_type': 'neural_render',  # or 'spherical_gaussian'
        'use_tri_plane': True,
        'tet_res': 90,
        'geometry_type': 'conv3d',
        'data_camera_mode': 'shapenet_car',
        'n_implicit_layer': 1,
        'cbase': 32768,
        'cmax': 512,
        'fp32': False
    },
    "TORCH_SEED": 0,  # Random seed
    "TORCH_RESOLUTION": 1024  # Resolution of the output image
}


# Application definition

INSTALLED_APPS = [

    # Built-in Contrib Apps
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Rest Framework Apps and Extensions
    'rest_framework',
    'drf_yasg',
    'django_extensions',
    'corsheaders',

    # User-Apps
    'studio_YAIVERSE.apps.accounts',
    'studio_YAIVERSE.apps.main',

]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # CORS
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'studio_YAIVERSE.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'studio_YAIVERSE.wsgi.application'


# Password validation

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


# Internationalization

LANGUAGE_CODE = 'ko-kr'

TIME_ZONE = 'Asia/Seoul'

USE_I18N = True

USE_L10N = True

USE_TZ = False


# Static files (CSS, JavaScript, Images)

STATIC_URL = '/static/'

STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'

MEDIA_ROOT = BASE_DIR / 'attachment'

FILE_UPLOAD_MAX_MEMORY_SIZE = 128 << 20  # 100 MB


# Default primary key field type

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# CORS Setting

CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOW_CREDENTIALS = True


# Remove modules used
del _os, _pathlib
