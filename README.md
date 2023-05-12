# Studio-YAIVerse Backend

> 2nd YAI-Con: Studio YAIVerse Team's backend repository

## Running Instructions

### Getting Dependencies: Python & CUDA

* Python 3.8: We highly recommend using `python3.8` for compatibility.
* CUDA and CUDNN: We need `cuda-11.1` and `cudnn-8.0.5` for compiling GET3D extensions.
  * Manually install via [homepage](https://developer.nvidia.com/cuda-downloads), or **<u>use our Dockerfile</u>**. (recommended)
  * Instead, set `TORCH_WITHOUT_CUSTOM_OPS_COMPILE=0` in environ or django settings, to disable GET3D extensions. (little bit slow inference speed)

### Runtime Preparation

* (Step 1) **<u>Recursively</u>** clone this repository (due to submodule dependency).

```bash
git clone --recursive https://github.com/studio-YAIVERSE/Backend studio-YAIVERSE-Backend
cd studio-YAIVERSE-Backend
# if you are updating an existing checkout
git submodule sync && git submodule update --init --recursive
```

* (Step 2) Make virtual environment. <u>You can use Dockerfile instead</u>. (see below)

```bash
python3 -m pip install virtualenv
python3 -m virtualenv venv --python=3.8
source venv/bin/activate
```

* (Step 3) We provide `setup.sh` for installing dependencies and model weight retrieval.

```bash
sh setup.sh
```

### Runtime preparation with Docker

<u>Using Docker image is optional but recommended</u>. Follow this instead of (Step 2) upon.

* Build Docker image

```bash
docker build -f Dockerfile -t studio-yaiverse:v1 .
```

* Start an interactive docker container

```bash
docker run --gpus device=all -it --rm -v $(pwd):/workspace -it studio-yaiverse:v1 bash
```

### How to run dev server

* `--noreload` is required: otherwise, model is loaded twice.
* you can use `python3 manage.py` interface with `python3 -m studio_YAIVERSE`.

```bash
python3 -m studio_YAIVERSE runserver --noreload
TORCH_DEVICE=cuda:0 python3 -m studio_YAIVERSE runserver --noreload  # to specify device
TORCH_ENABLED=0 python3 -m studio_YAIVERSE runserver  # to disable pytorch ops
```

### Appendix. How to deploy with gunicorn & nginx

<details>
<summary>View / Hide</summary>

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
   
   if [ -f /etc/nginx/sites-enabled/default ]; then
       sudo rm /etc/nginx/sites-enabled/default
   fi
   sudo ln -s /etc/nginx/sites-available/studio-yaiverse-site /etc/nginx/sites-enabled/studio-yaiverse-site
   ```
4. Enable and start gunicorn and nginx service
   ```bash
   sudo systemctl enable studio-yaiverse-gunicorn
   sudo systemctl start studio-yaiverse-gunicorn
   sudo systemctl enable nginx
   sudo systemctl restart nginx
   ```
5. Your site is now running at `http://$SERVER_NAME`!

</details>
