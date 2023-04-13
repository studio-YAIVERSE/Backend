"""
WSGI with Gunicorn
==================
This module provides a WSGI application for use with Gunicorn.

If you run server with gunicorn, it will use production settings.
Otherwise, it will use local settings.

"""

import sys
import os

from django.core.wsgi import get_wsgi_application

if 'gunicorn' in sys.modules:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'studio_YAIVERSE.settings.prod')
else:
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'studio_YAIVERSE.settings')

application = get_wsgi_application()
