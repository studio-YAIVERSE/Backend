from django.apps import AppConfig


class MainConfig(AppConfig):
    # Default Configs
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'studio_YAIVERSE.apps.main'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .pytorch import setup, construct_all
        setup()
        construct_all()