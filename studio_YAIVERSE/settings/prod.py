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


def fetch_secret():
    import json
    defaults = {
        'ALLOWED_HOSTS': ['*'],
        'DATABASES': {'default': {'ENGINE': 'django.db.backends.sqlite3',
                                  'NAME': str(BASE_DIR / 'db.sqlite3')}},
        'SECRET_KEY': 'warning!-overwrite-this-secret-key-to-your-own-value'
    }
    try:
        with open(BASE_DIR / 'secret.json', 'r') as fd:
            globals().update(json.load(fd))
            globals().setdefault('DATABASES', defaults['DATABASES'])
    except FileNotFoundError as exc:
        if input(
                "[Error] In Production mode, secret.json is required but not found. do you want to create it? [y/n]: "
        ).lower() in ('y', 'yes'):
            with open(BASE_DIR / 'secret.json', 'w') as fd:
                json.dump(defaults, fd, indent=2)
            globals().update(defaults)
        else:
            raise RuntimeError("secret.json is required for gunicorn production environment.") from exc


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY: str  # type: str

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS: list  # type: list[str]


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases

DATABASES: dict  # type: dict[str, dict[str, str]]
