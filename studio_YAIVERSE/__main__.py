def main():
    from django.core.management import execute_from_command_line
    import os
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'studio_YAIVERSE.settings')
    execute_from_command_line()


if __name__ == '__main__':
    main()
