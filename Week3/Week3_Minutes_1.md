# Week3_캡스톤 디자인 팀 회의록 
**1. 일시**: 2025년 9월 23일 22시  
**2. 장소**: 공학 1관 354  
**3. 참석자**: 이왕재, 양범석, 오서진, 원종은  

# 회의 안건

**1.** 사용할 Dataset, Model 선정   
**2.** 다음주 계획, 모델링 시작 시점  
**3.** 물품 구매   

---

# 회의 결과
## 1. 사용할 Dataset, Model 선정

#### 데이터셋 구성
| 순위 | 데이터셋 | 역할 | 사유 |
|------|----------|------|------|
| 1 | **[AI Hub 이상행동](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=171)** | 메인 데이터셋 | 이상행동 CCTV 영상 717시간, 12개 라벨 |
| 2 | **[UCF-Official](https://www.crcv.ucf.edu/projects/real-world/) / [UCA](https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset) / [w. BBox](https://www.kaggle.com/datasets/vulamnguyen/ucfcrime2local-with-ground-truth-bounding-boxes)** | 서브 데이터셋 | 실제 범죄 영상으로 일반화 성능 검증 |
| 3 | **[Real Life Violence Situations](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)** | 추가 데이터셋 | 정상 상태 데이터 보강 |

**선정 근거**
- AI Hub 데이터셋 내 정상 상태 데이터 부족
- CRxK-6 연구에서 정상 상태 감지 최대 정밀도 0.750에 그침
- 정확도 향상을 위한 정상 데이터 증강 필요

#### 모델 아키텍처
- **메인 모델**: ResNet-18
- **추가 검토 예정 모델**:
  - BiLSTM (시계열 행동 패턴 학습)
  - Transformer (Attention 기반 행동 의도 분석)  
  - GCN (Graph Convolutional Network, 스켈레톤 관절 관계 학습)
  - 3D CNN (공간-시간 특징 동시 추출)
  - ST-GCN (Spatial-Temporal Graph CNN, 스켈레톤 기반)


#### 팀 구성
| 팀명 | 멤버 | 접근 방식 | 모델 선택 |
|------|------|-----------|-----------|
| **뼈팀** | 이왕재, 원종은 | 스켈레톤 기반 | 모델 변경 가능성 있음 |
| **순살팀** | 양범석, 오서진 | RGB 영상 기반 | 모델 변경 가능성 있음 |

---

# 2. 다음주 계획, 모델링 시작 시점
- **데이터셋 다운로드**: 9월 24일 내로 사용 웹하드 선정 후 AI Hub 이상행동 데이터셋(~5TB) 웹하드 다운로드 시작
- 각 팀별 최종 데이터셋 및 모델 확정
- 대략적인 모델링 시작
- 스켈레톤 유무에 따른 차별화된 접근법 구체화

---

# 3. 물품 구매 
- **상태**: 보류
- **추후 결정 예정**

---

# 연구 차별점

### 핵심 차별화 요소
1. **의심 단계 탐지**: 범죄 발생 전 전조 행동 감지
2. **스켈레톤 기반 분석**: 프라이버시 보장 및 행동 의도 집중 분석
3. **실시간 탐지**: 즉시 경고 시스템 구현

### 추가 연구 계획 (시간 여유 시)
- **군중 탐지**: 다수 인원 상황에서의 이상행동 식별
- **가해자 특성 분석**: 성별, 키, 상의 색상 등 인적사항 추출

---

# Next Action Items
- [x] 전체: AI Hub 데이터셋(~5TB) 웹하드 다운로드 
- [ ] 뼈팀: 스켈레톤 추출 도구 및 모델 아키텍처 확정
- [ ] 순살팀: RGB 기반 전처리 파이프라인 및 모델 구조 설계
- [ ] 전체: 정상 데이터 수집 및 라벨링 방안 구체화
- [ ] 다음 회의: 각 팀별 모델링 진행 상황 공유
