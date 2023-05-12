__version__ = "1.0.0"


def main():
    import os
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'studio_YAIVERSE.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line()
