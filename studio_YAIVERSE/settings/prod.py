"""
production settings module.
you should implement SECRET_KEY, ALLOWED_HOSTS, and DATABASES in `BASE_DIR / secret.json` file.

secret.json example using mysql database:
```
{
  "ALLOWED_HOSTS": [
    "{ipv4-address}",
    "{your-domain-name}"
  ],
  "SECRET_KEY" : "{secret-key}",
  "DATABASES": {
    "default": {
      "ENGINE": "django.db.backends.mysql",
      "NAME": "{database-name}",
      "USER": "{database-account-username}",
      "PASSWORD": "{database-account-password}",
      "HOST": "{database-host-address}",
      "PORT": "3306"
    }
  }
}
```
"""

from .base import *

try:

    with open(BASE_DIR / 'secret.json', 'r') as fd:

        import json
        globals().update(json.load(fd))

        try:
            DATABASES  # noqa
        except NameError:
            DATABASES = {
                'default': {
                    'ENGINE': 'django.db.backends.sqlite3',
                    'NAME': BASE_DIR / 'db.sqlite3',
                }
            }
            print("\nCannot resolve database, using default sqlite db...\n")

except FileNotFoundError as exc:

    raise RuntimeError(
        "secret.json is required for gunicorn production environment."
    ) from exc

del fd, json


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY: str  # type: str

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS: list  # type: list[str]


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES: dict  # type: dict[str, dict[str, str]]


# cors

CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOW_CREDENTIALS = True


# Session Settings

SESSION_COOKIE_AGE = 86400  # default is 1209600 (two weeks)

# SESSION_COOKIE_NAME = 'cookie-name'

SESSION_COOKIE_SECURE = True  # for https

CSRF_COOKIE_SECURE = True  # for https
