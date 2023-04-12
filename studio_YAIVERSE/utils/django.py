def configure_settings(settings_mod):
    from django.conf import settings
    keys = filter(str.isupper, dir(settings_mod))
    options = {k: getattr(settings_mod, k) for k in keys}
    settings.configure(**options)
