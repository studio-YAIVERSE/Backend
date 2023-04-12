git clone https://github.com/nv-tlabs/GET3D.git

python3 -m pip install -r requirements.txt

mkdir weights
FILEID=18UdsemUdKo75GXmQLLVYdcOhNZ3zM215
FILENAME=weights/shapenet_car.pt
curl -sc ./cookie.txt "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ./cookie.txt`&id=${FILEID}" -o ${FILENAME}
FILEID=1gXwK3-Y16UBi1-KgTClKcuGv8EFXSeKo
FILENAME=weights/shapenet_chair.pt
curl -sc ./cookie.txt "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ./cookie.txt`&id=${FILEID}" -o ${FILENAME}
FILEID=1XNWQLCwJ6V_wr2G0dDRsyWyT05b1lWmK
FILENAME=weights/shapenet_motorbike.pt
curl -sc ./cookie.txt "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ./cookie.txt`&id=${FILEID}" -o ${FILENAME}
FILEID=1msJs8HUR_fjhAJJHrWQkgbwgRW_gFSFq
FILENAME=weights/shapenet_table.pt
curl -sc ./cookie.txt "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
curl -Lb ./cookie.txt "https://drive.google.com/uc?export=download&confirm=`awk '/_warning_/ {print $NF}' ./cookie.txt`&id=${FILEID}" -o ${FILENAME}
rm cookie.txt

TORCH_ENABLED=0 python3 -m studio_YAIVERSE migrate
