from django.apps import AppConfig


class MainConfig(AppConfig):
    # Default Configs
    default_auto_field = 'django.db.models.BigAutoField'
    name = __name__.rsplit('.', 1)[0]

    @staticmethod
    def ready():
        from django.conf import settings
        from .pytorch import init
        init(settings.TORCH_SETTINGS)
