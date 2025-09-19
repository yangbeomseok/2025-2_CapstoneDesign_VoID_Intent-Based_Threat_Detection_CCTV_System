# 3주차 보고서 초안,, CRxK-6 논문 기반

## 1. 조사 주제
- 행동 의도 탐지에 적합한 딥러닝 모델 아키텍처 조사
- 활용 가능한 공개 데이터셋 목록 정리 및 특성 분석

## 2. 딥러닝 모델 아키텍처 조사

1. **2D CNN 기반 모델**
   - **ResNet-18**
     - 장점: 경량화된 구조로 빠른 학습 및 추론 가능, CRxK 논문에서 이진 분류에 성공적으로 활용[1]
     - 성능: 98% 이진 분류 정확도 달성[1]
   - **EfficientNet-B0**
     - 장점: 컴퓨팅 자원 대비 높은 성능, 전이 학습으로 소량 데이터에도 적응 용이[1]
     - 성능: 다중 분류에서 94% F1-score 기록[1]

2. **3D CNN 기반 모델**
   - **I3D (Inflated 3D ConvNet)**
     - 설명: 공간-시간 특징을 동시에 학습, Kinetics 사전학습을 통해 미세한 행동 패턴 포착에 효과적
   - **SlowFast Network**
     - 설명: Slow 경로로 공간 정보, Fast 경로로 시간 정보 처리, 두 경로 병합으로 세밀한 동작 분석

3. **Transformer 기반 영상 모델**
   - **TimeSformer**
     - 설명: 분리된 공간·시간 Self-Attention, 병렬 처리로 대규모 영상 데이터 학습 효율성 확보

4. **Skeleton 기반 모델**
   - **ST-GCN (Spatial-Temporal Graph Convolutional Network)**
     - 설명: 관절 키포인트를 그래프 형태로 분석, 프라이버시 보호 가능, 의도 기반 행동 탐지에 적합

## 3. 공개 데이터셋 목록 및 특성 분석

| 데이터셋명      | 특성                                     | 활용 방안                          | 출처             |
|---------------|----------------------------------------|-----------------------------------|----------------|
| [CRxK](https://github.com/dxlabskku/CRxK-6)    | 6개 범죄 유형, 총 51,000 프레임 샘플링[1]  | 핵심 범죄 행동 프레임 학습         | Sci. Rep. 2025[1] |
| [AI Hub 이상행동](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=171) | 8,436편(717시간), 12개 이상행동 라벨[61]   | 정상 vs 의심 행동 이진 분류        | AI Hub[61]      |
| HR-Crime  | UCF-Crime 데이터의 단점을 보완하여 정제 | 이상행동 탐지 일반화 성능 검증    | UCSB CV Lab[3]  |
| [UNI-Crime](https://link.springer.com/chapter/10.1007/978-3-030-19823-7_23) | HR-Crime과 마찬가지, 10초 길이로 정제 후 유튜브 영상을 추가 수집, 1001개의 정상 420개의 범죄 | 이상행동 탐지 일반화 성능 검증 | Sci. Rep. 2025[1] |
| [XD-Violence](https://roc-ng.github.io/XD-Violence/)   | 4,000편 폭력·위협 행동 비디오             | 폭력·위협 세부 분류                | Bilibili AI[4]  |
| [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html)  | 13개 장면, 이상행동 라벨                  | 이상행동(Anomaly) 탐지            | SJTU[5]         |
| [VIRAT](https://gitlab.kitware.com/viratdata/viratannotations)         | 10시간 이상 공공장소 활동·침입 라벨       | 이동 및 침입 이벤트 감지          | NIST[6]         |
| [Avenue](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)        | 16편 영상, 1,561 이상행동 이벤트          | 군중 이상행동 분석                | UCSD[7]         |

## 4. 분석 요약 및 제언
- **모델 아키텍처**: 초기 프로토타입으로 ResNet-18/EfficientNet-B0 사용 후, 추가로 I3D 또는 SlowFast 도입하여 시간 정보 학습 강화 권장.
- **Skeleton 모델**: ST-GCN은 프라이버시 보호 환경에서 행동 의도 탐지에 유리.
- **데이터셋 전략**: CRxK 및 AI Hub 데이터로 기본 학습, UCF-Crime과 XD-Violence로 일반화 검증, Anomaly 데이터셋(ShanghaiTech 등)으로 이상행동 탐지 확장.

---

### References
1. Chaehee An et al., "CRxK dataset: a multi-view surveillance video dataset for re-enacted crimes in Korea," *Sci. Rep.* 2025. [file:456a7cc4-4bbf-4adf-aa5f-cfaf2d47a0e0]
3. UCF-Crime dataset overview, UCSB Computer Vision Lab.
4. XD-Violence dataset, Bilibili AI Research.
5. ShanghaiTech Campus anomaly dataset, Shanghai Jiao Tong University.
6. VIRAT Video Dataset, National Institute of Standards and Technology.
7. Avenue dataset, UCSD Machine Vision Group.
61. AI Hub ‘이상행동 CCTV 영상’ dataSetSn=139. [web:61]
