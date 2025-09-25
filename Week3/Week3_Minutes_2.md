
# 캡스톤 디자인 4주차 업무

**업무 기간:** 2025.09.25 ~ 2025.10.01

---

## 1. 데이터셋 전처리 담당자별 업무

-   [ ] **범석 (폭행)**: 폭행 영상 제작 및 시간 기록
-   [ ] **왕재 (절도)**: 절도 영상 제작 및 시간 기록
-   [ ] **서진 (싸움)**: 싸움 영상 제작 및 시간 기록
-   [ ] **종은 (실신)**: 실신 영상 제작 및 시간 기록

---

## 2. 데이터셋 전처리 과정

1.  **AI Hub 데이터셋 다운로드**
    -   [AI Hub CCTV 데이터셋 링크](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=171)로 이동하여 할당된 데이터셋을 다운로드합니다.
    -   <img width="797" height="924" alt="Image" src="https://github.com/user-attachments/assets/5ec52a27-a84e-47c6-9916-6c0850a252c1" />
    -   본인이 사용하려는 영상 파일을 선택하여 다운로드합니다.

2.  **영상 편집**
    -   압축 해제 후, `Microsoft Clipchamp`으로 영상을 업로드합니다.
    -   <img width="1404" height="769" alt="Image" src="https://github.com/user-attachments/assets/cfe49570-a206-40d4-a9e2-332479dbe1ed" />
    -   영상 앞/뒤로 **정상 행동 10초**, **비정상 행동 10초**가 포함되도록 영상을 자릅니다.

3.  **저장 및 파일 정리**
    -   `내보내기`를 클릭하여 **1080p** 화질로 저장합니다.
    -   파일 제목을 `(행동명_번호)` 형식으로 수정합니다. (예: `절도_1.mp4`)

4.  **시간 기록 (Annotation용)**
    -   영상에 해당하는 JSON 파일을 생성하여 행동별 시간을 기록합니다.
    -   *추후 annotation 파일 생성을 위해 단순 초 단위 기록도 가능합니다.*

### 예시: `절도_1.json`
<img width="467" height="358" alt="Image" src="https://github.com/user-attachments/assets/ecb52890-cfce-45c3-9084-abc1a0d75363" />


