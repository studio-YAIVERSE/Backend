def main():
    from django.core.management import execute_from_command_line
    from .utils.django import configure_settings
    from . import settings
    configure_settings(settings)
    execute_from_command_line()


if __name__ == '__main__':
    main()
