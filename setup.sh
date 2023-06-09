# Install dependencies
apt-get install -y xvfb || sudo apt-get install -y xvfb || echo "Failed to install xvfb"
python3 -m pip install -r requirements.txt

# Download weights
wget https://github.com/studio-YAIVERSE/Backend/releases/download/1.0.0/weights.z01
wget https://github.com/studio-YAIVERSE/Backend/releases/download/1.0.0/weights.z02
wget https://github.com/studio-YAIVERSE/Backend/releases/download/1.0.0/weights.zip
zip -s 0 weights.zip --output merge.zip && unzip merge.zip && rm merge.zip weights.{z01,z02,zip}

# Prepare local database
TORCH_ENABLED=0 python3 -m studio_YAIVERSE migrate

# Compile pytorch modules: will be automatically compiled while running first time
TORCH_ENABLED=1 python3 -m studio_YAIVERSE shell -c "exit(0)"
