# 3주차 Backend TODO

- [ ] Finalize Warm-Up
  - Fun `studio_YAIVERSE.main.apps._warm_up`
- [X] Pretrained Weight Retrieve & Register in Settings
- [ ] Test & Modification of Inference View (in GPU Server)
  - Fun `studio_YAIVERSE.main.views.inference.inference`
- [ ] Modification of Object3D RUD View
  - Cls `studio_YAIVERSE.main.views.object_3d.Object3DModelViewSet`
  - Cls `studio_YAIVERSE.main.views.object_3d.get_object_3d_list`
- [ ] Implement Text to Model View (have to change `MODEL_OPS`)
  - Val `studio_YAIVERSE.config.settings.MODEL_OPTS`
  - Fun `studio_YAIVERSE.main.views.inference.inference`


# 1-2주차 Backend TODO

- [X] REPO Creation & Base Settings
- [X] base DB
- [X] base VIEW
- [X] schema

## DB

- User Table
    
| userid   | identifier | USER 고유 번호                            |
|----------|------------|---------------------------------------|
| username | string     | ID 개념                                 |
| password | string     | hash value of user password (아직은 명목상) |

- 3D Object Table

| object_id            | identifier | OBJECT 고유 번호       |
|----------------------|------------|--------------------|
| object_name          | string     | OBJECT 이름          |
| object_description   | string     | OBJECT 설명          |
| object_uri           | string     | 3D OBJECT FILE URI |
| object_thumbnail_uri | string     | THUMBNAIL URI      |
| (기타 필요한 metadata)    |            |                    |

## VIEW
1. `{"username": "$USERNAME"}` 형태로 전달 시 유저 생성
2. `{"username": "$USERNAME"}` 형태로 전달 시 object list 반환
    - Return 형식: **json(list)**
            
        ```json
        [
            {"object_id": "", "object_name": "", "object_thumbnail_uri": ""},
            {"object_id": "", "object_name": "", "object_thumbnail_uri": ""},
            {"object_id": "", "object_name": "", "object_thumbnail_uri": ""}
        ]
        ```
            
3. `{"username": "$USERNAME", "object_id": "$OBJECT_ID"}` 형태로 전달 시 3d file 반환
    - Return 형식: **Binary File**

## **Schema**

`127.0.0.1:8000/`
