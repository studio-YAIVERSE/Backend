# Backend TODOs

<details>
<summary>펼쳐보기</summary>

### 4주차 Backend TODO

- [ ] Implement Text to Model View (have to change `MODEL_OPS`)
  - Val `studio_YAIVERSE.settings.MODEL_OPTS`
  - Fun `studio_YAIVERSE.apps.main.pytorch.functions.inference`
- [ ] Thumbnail Generation


### 3주차 Backend TODO

- [X] Finalize Warm-Up
- [X] Pretrained Weight Retrieve & Register in Settings
- [X] Test & Modification of Inference View (in GPU Server)
- [X] Modification of Object3D RUD View


### 1-2주차 Backend TODO

- [X] REPO Creation & Base Settings
- [X] base DB
- [X] base VIEW
- [X] schema

</details>

# How to run

## Requirements

* We need `cuda-11.1` and `cudnn-8.0.5` for **compiling GET3D extensions**.
* Manually install via [homepage](https://developer.nvidia.com/cuda-downloads), or use [Docker image](https://hub.docker.com/r/nvidia/cuda).


## Preparation

* Optional: make virtual environment. (recommended)

```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv --python=3.8
source venv/bin/activate
```

* We provide `setup.sh` for installing dependencies and model weight retrieval.

```bash
sh setup.sh
```


## How to run dev server

* `--noreload` is required: otherwise, model is loaded twice.
* you can use `python3 manage.py` interface with `python3 -m studio_YAIVERSE`.

```bash
python3 -m studio_YAIVERSE runserver --noreload
```


# Appendix. How to deploy with gunicorn & nginx

* `SERVER_NAME`, `SECRET_KEY`(optional) is required. alternate it to your server address.
```bash
export SERVER_NAME={your-server-address}
export SECRET_KEY={secret-key}
```

1. Write secret.json: implement SECRET_KEY, ALLOWED_HOSTS, and DATABASES
```bash
echo "{
  \"ALLOWED_HOSTS\": [\"$SERVER_NAME\"],
  \"SECRET_KEY\" : \"$SECRET_KEY\",
  \"DATABASES\": {
    \"default\": {
      \"ENGINE\": \"django.db.backends.sqlite3\",
      \"NAME\": \"$(pwd)/db.sqlite3\"
    }
  }
}
" > secret.json
```

2. Write gunicorn service file (gunicorn is already installed by `setup.sh`)
```bash
sudo echo "[Unit]
Description=studio-YAIVERSE gunicorn daemon
After=network.target

[Service]
User=$(whoami)
Group=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(which gunicorn) \\
        --workers 2 \\
        --bind unix:/tmp/studio-yaiverse-gunicorn.sock \\
        studio_YAIVERSE.wsgi:application

[Install]
WantedBy=multi-user.target" > /etc/systemd/system/studio-yaiverse-gunicorn.service
```

3. Install nginx, prepare static files, and configure your site.
```bash
sudo apt install nginx

TORCH_ENABLED=0 python -m studio_YAIVERSE collectstatic

cd /etc/nginx/sites-available/
sudo echo "server {
        listen 80;
        server_name $SERVER_NAME;

        location = /favicon.ico { access_log off; log_not_found off; }

        location /static {
                alias $(pwd)/staticfiles;
        }

        location /media {
                alias $(pwd)/attachment;
        }

        location / {
                include proxy_params;
                proxy_pass http://unix:/tmp/studio-yaiverse-gunicorn.sock;
        }
}" > /etc/nginx/sites-available/studio-yaiverse-site

cd /etc/nginx/sites-enabled/
if [ -f default ]; then
    sudo rm default
fi
sudo ln -s /etc/nginx/sites-available/studio-yaiverse-site studio-yaiverse-site
```

4. Enable and start gunicorn and nginx service
```bash
sudo systemctl enable studio-yaiverse-gunicorn
sudo systemctl start studio-yaiverse-gunicorn
sudo systemctl enable nginx
sudo systemctl restart nginx
```

5. Your site is now running at `http://$SERVER_NAME`!
