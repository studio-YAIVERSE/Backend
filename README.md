# 1주차 Backend TODO

- [X] 환경 설정 및 REPO 만들기
- [ ] DB 구현
- [ ] VIEW 구현
- [ ] schema

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

https://api.partyone.kr/schema/swagger-ui
