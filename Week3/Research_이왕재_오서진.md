# Week3_행동 의도 탐지에 적합한 딥러닝 모델 아키텍처 조사 및 관련 선행 연구 논문 및 실제 적용 사례 조사

**문서 목적**: 행동 의도 및 이상행동 탐지에 적합한 딥러닝 모델 아키텍처를 조사하고, 이에 관련된 선행 연구 논문 및 실제 적용 사례를 분석을 통해 최신 동향을 파악하고, 향후 시스템 설계 및 연구 방향 설정을 위한 기초 자료를 제공한다.

---

## 1. 행동 의도 탐지에 적합한 딥러닝 모델 아키텍처 분석 

### 1.1. 스켈레톤 기반 시퀀스 모델 (Skeletal Sequence-based Models)
1) **Bi-LSTM (Bidirectional Long Short-Term Memory)**
   - LSTM은 시계열 데이터를 다루는 데 특화된 RNN(Recurrent Neural Network)의 한 종류
   - 일반적인 LSTM이 과거의 정보만으로 미래를 예측하는 것과 달리, Bi-LSTM은 순방향과 역방향 두 가지 방향으로 데이터를 처리하여 현재 시점의 행동을 예측할 때 **과거와 미래의 문맥 정보**를 모두 활용함
   - 예를 들어, '주머니에 손을 넣는다'는 행동이 **이후에 '물건을 꺼내는 행동'** 으로 이어질 것인지, **이전에 '주위를 살피는 행동'** 이 있었는지와 같은 맥락을 더 정확하게 파악할 수 있음

2) **Transformer**
   - 자연어 처리 분야에서 혁신을 일으킨 Transformer는 **Attention 메커니즘**을 핵심으로 사용함
   - Attention은 시퀀스의 모든 요소 간의 상호 관계를 한 번에 계산하여, 특정 시점의 행동이 전체 시퀀스에서 얼마나 중요한지를 가중치로 부여함
   - 이 덕분에 길이가 긴 행동 시퀀스에서도 중요한 순간들을 놓치지 않고 분석할 수 있음
   - 예를 들어, 길게 이어지는 '배회' 행동 시퀀스에서, 갑자기 '주머니에 손을 넣는' 미세한 변화를 중요한 신호로 인식하는 데 탁월함
  
### 1.2. 스켈레톤+GCN 기반 모델 (Skeletal-GCN Based Models)
- **Graph Convolutional Network (GCN)**
  
  - GCN은 비유클리드 데이터(non-Euclidean data)인 그래프 데이터를 처리하는 데 최적화된 딥러닝 모델
  - 인체 스켈레톤 데이터를 그래프로 변환하면, 관절 간의 **공간적 관계** (ex. 팔과 몸통의 거리)와 **시간적 변화** (ex. 손이 올라갔다 내려오는 움직임)를 동시에 학습할 수 있음
  - 이 방식은 다양한 시점과 각도에서 촬영된 영상에서도 인체의 복잡한 자세 변화를 효과적으로 분석하는 데 매우 강력함
 
### 1.3. 멀티모달 퓨전 모델 (Multimodal Fusion Models)
- **CNN+RNN (Convolutional Neural Network + Recurrent Neural Network)**
  
  - **CNN**: 비디오의 각 프레임에서 시각적 특징(ex. 배경, 옷의 색깔, 객체의 모양)을 추출하는 역할을 함
  - **RNN**: CNN이 추출한 특징 벡터들의 **시간적 순서**를 학습하여 행동 패턴을 인식함
  - 이 모델은 사람의 행동뿐만 아니라, 행동이 발생하는 **환경 정보(Contextual information)** 까지 함께 고려할 수 있어, 행동 의도 탐지의 정확도를 높이는 데 기여함
  - 예를 들어, '주머니에 손을 넣는' 행동이 **은행**에서 발생하는지, **골목길**에서 발생하는지에 따라 위험 확률을 다르게 판단할 수 있음

---

## 2. 행동 의도 및 이상행동 탐지에 관한 선행 연구 논문

| 주제 | 논문 이름 | 목표 | 접근 방법 | 링크 | 
| :--- | :--- | :--- | :--- | :--- |
| CCTV 기반 비정상 행동 예측 모델 | **다중 CCTV 연동 기반 비정상 행동 예측모델** | 다수 CCTV에서 수집한 객체 정보를 **연동·추적**하고, 객체 간 **관계/상호작용**을 분석해 **범죄 전 비정상 징후**를 예측 | 객체 인식 → 다중 카메라 연동 추적 → 시간적 행동 패턴 분석(관계 중심) | **[Paper](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NPAP12684279&utm_source=chatgpt.com)** | 
| 비디오 기반 폭력 행동 탐지 | **Human skeletons and change detection for efficient violence recognition in surveillance videos (CVIU, 2023)** | 감시 영상에서 **경량** 구조로 폭력/비폭력을 자동 탐지 | 사람 **스켈레톤**과 **변화 감지(모션/프레임 변화)** 를 결합해 시간적 움직임 패턴을 효율적으로 포착(실전 감시 환경에서의 속도·효율 중시) | **[Paper](https://www.sciencedirect.com/science/article/pii/S1077314223001194?utm_source=chatgpt.com)** / **[GitHub](https://www.sciencedirect.com/science/article/pii/S1077314223001194?utm_source=chatgpt.com)** | 
| 비디오 기반 폭력 행동 탐지 | **Human Interaction Learning on 3D Skeleton Point Clouds for Video Violence Recognition (ECCV, 2020)** | **사람 간 상호작용**을 중심으로 폭력 여부를 정밀 인식 | 비디오에서 추출한 스켈레톤 시퀀스를 **3D 포인트클라우드**로 구성하고, **Skeleton Points Interaction Learning(SPIL)** 로 포인트 간 관계를 학습(지역/전역 상호작용, 멀티헤드 메커니즘) | **[ECCV 공식](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490069.pdf?utm_source=chatgpt.com)** / **[확장·보강판(arXiv, 2023)](https://arxiv.org/abs/2308.13866?utm_source=chatgpt.com)** | 
| 실시간 군중 이상행동 탐지 | **An enhanced framework for real-time dense crowd anomaly detection (CADF) (Springer, 2025)** | 공항·역 등 **혼잡 환경**에서 **실시간 이상행동**(갑작스런 분산·도주 등) 탐지 | **검출+추적(YOLOv8/DeepSORT 등)** 로 군중 동태를 추적하고, **CNN/RNN/3D-CNN**으로 시공간 패턴을 분석(소프트-NMS 등 실시간 최적화) | **[CADF(Springer)](https://link.springer.com/article/10.1007/s10462-025-11206-w?utm_source=chatgpt.com)** / **[실시간 군중 이상탐지 서베이(ACM/PUC)](https://dl.acm.org/doi/abs/10.1007/s00779-021-01586-5?utm_source=chatgpt.com)** / **[YOLO 기반 실시간 군중 검출(PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9885395/?utm_source=chatgpt.com)** | 
| 그래프 기반 행동 인식 모델 | **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (AAAI, 2018)** | 스켈레톤(관절 좌표)을 **그래프(노드/엣지)** 로 모델링해 **공간+시간** 관계를 동시에 학습, 행동 인식 정확도 **대폭 향상** | **시공간 그래프 합성곱**(ST-GCN)으로 관절 간 상호작용을 계층적으로 추출 → 스켈레톤 기반 라인의 **표준 베이스라인** | **[Paper](https://arxiv.org/abs/1801.07455?utm_source=chatgpt.com)** / **[GitHub](https://github.com/yysijie/st-gcn?utm_source=chatgpt.com)** |

---

## 3.  행동 의도 및 이상행동 탐지의 실제 적용 사례

### 3.1. [서울특별시 서초구 — AI 기반 범죄·사고 예방](https://www.munhwa.com/article/11444245)

- **도입 범위:** 서울특별시 서초구 전역  
- **카메라 규모:** CCTV 4,100여 대  
- **목표:** 폭력, 쓰러짐, 화재 등 10가지 이상의 위험 상황을 실시간 자동 감지  

- **실제 효과 및 사례**
  * **만취자 구조 (2023.01, 방배동)**  
    - 도로에서 만취자가 쓰러져 움직이지 않는 상황을 AI가 '쓰러짐'으로 감지  
    - 관제센터 팝업 → 관제 요원 신고 → 경찰 출동 및 구조  
  * **실시간 범인 검거 (다수 사례)**  
    - 관제 요원이 수상한 행동을 하는 인물을 수동으로 지정  
    - AI가 이동 경로를 예측하고 인접 CCTV로 자동 연계하여 추적  
    - 경찰과 공조하여 현장에서 검거  

### 3.2. [서울특별시 한강교량 — 투신 예방 예측 시스템](https://blog.naver.com/haechiseoul/222412624695)

- **시스템 개요**  
  * 목적: 교량 구간에서 **투신 전조 행동**을 조기 인지하고 관제에 **선별 알림** 전송  

- **핵심 아이디어 (What)**  
  * 과거 **투신 시도자 행동 패턴**을 학습  
    - 예: 교량에서 **앞뒤 반복 보행**, **난간 아래 내려다보기**, **난간 잡는 동작** 등  
  * 실시간 CCTV에서 **유사 행동**이 감지되면, 해당 **영상 일부를 관제요원에게 선별 전송**  

- **데이터 활용**  
  * 1년간 소방재난본부 출동 정보  
  * CCTV 동영상  
  * 감지 센서 데이터 (교량 줄 장력, 레이저 센서)  
  * 신고 이력 및 통화 내용 등 **정형·비정형 데이터 통합**  

- **기술/운영 방식**  
  * AI 딥러닝 기반 영상 분석 → 과거 투신자 행동 추출 → 이상 행동 패턴 유사도 탐지  
  * 교량 말단 줄 센서(장력) 및 레이저 감지 센서 병합으로 **물리적 징후도 함께 활용**  

### 3.3. [경기도 오산시 — 지능형 CCTV 위험징후 자동 인식](https://www.joongang.co.kr/article/23897881)

- **도입 범위:** 경기도 **오산시 전역(관제센터 연계)**
- **시스템 개요:** 길에 **쓰러진 사람**, **인파 밀집**, **휠체어 이동**, **연기(화재 징후)** 등 **위험 징후**를 자동 인식하여
  관제센터 **메인 화면 중앙에 자동 표출**하는 지능형 CCTV 운영

- **적용 기술 (How)**
  * **위험징후 판단 모델:** 과거 **신고 데이터** 및 **관제 경험**을 바탕으로 학습된 모델을 사용
  * **객체·행동 인지:** 영상에서 **사람/사물의 움직임**과 **군중 밀집 패턴**을 감지
  * **즉시 표출:** 위험 징후 감지 시 해당 카메라 영상이 **메인 화면 중앙**에 자동 팝업되어 요원 주의를 집중
  * **운영 정책:** 시간대·장소별 **임계치/지속시간** 조정으로 오탐을 최소화

---

## 4. 결론 및 최종 전략

---

## 요약
