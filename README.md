# 2025-2_CapstoneDesign_VOID_crime-detection-system 
2025년 3학년 2학기 캡스톤디자인_행동 의도 기반 위험 탐지 CCTV

# CRxK-6 프레임워크 사용 방법에 대해 ! 
## 목차
1. [CRxK-6 데이터 구성]
2. [실제 데이터 현황]
3. [AI Hub 원본 데이터 다운로드 방법]
4. [학습용 vs 추론용 코드 차이점]
5. [완전한 사용 가이드]
6. [프로젝트 적용 방안]

---

## 1. CRxK-6 데이터 구성

### GitHub Repository 구조
```
CRxK-6/
├── data/frame_data/          # ⚠️ 프레임 추출된 JPG 이미지들 (영상 아님)
│   ├── assault_frame/        # 폭행: frame_0000.jpg ~ frame_0099.jpg (100개)
│   ├── burglary_frame/       # 절도: frame_0000.jpg ~ frame_0099.jpg (100개)
│   ├── kidnap_frame/         # 납치: frame_0000.jpg ~ frame_0099.jpg (100개)
│   ├── robbery_frame/        # 강도: frame_0000.jpg ~ frame_0099.jpg (100개)
│   ├── swoon_frame/          # 실신: frame_0000.jpg ~ frame_0099.jpg (100개)
│   └── normal_frame/         # 정상: frame_0000.jpg ~ frame_0099.jpg (100개)
├── models/                   # 학습용 파이썬 스크립트들
│   ├── bin_*.py              # 이진 분류 (범죄 vs 정상)
│   ├── multi_*.py            # 다중 분류 (5개 범죄 유형)
│   └── bin_multi_*.py        # 2단계 파이프라인
├── preprocess/               # 데이터 전처리 도구
│   ├── datautils.py          # Dataset 클래스 및 평가 지표
│   ├── frame_crop.py         # 영상→프레임 추출
│   ├── make_annotation.py    # 어노테이션 생성
│   └── video_crop.py         # 영상 전처리
└── README.md
```

### 🔍 핵심 발견사항

**1. 데이터 형태: 프레임 추출된 이미지**
- ❌ 원본 영상 파일 없음
- ✅ JPG 형식 프레임 이미지만 존재
- 📊 총 600개 샘플 프레임 (각 카테고리 100개씩)

**2. 논문 vs GitHub 데이터량 차이**
| 구분 | 논문 언급 | GitHub 실제 |
|------|-----------|-------------|
| 총 프레임 수 | 2,054,013개 | 600개 |
| 학습 데이터 | 각 카테고리 8,500개 | 각 카테고리 100개 |
| 파일 형식 | 원본 영상 + 추출 프레임 | 샘플 프레임만 |

---

## 2. 실제 데이터 현황

### ⚠️ 치명적 한계점

**GitHub 데이터로는 실용적 모델 학습 불가능**
- 600개 프레임으로는 딥러닝 모델 학습에 절대적으로 부족
- 각 카테고리별 100개 샘플은 "코드 테스트용" 수준
- 과적합(Overfitting) 발생 확실

### 📈 필요한 최소 데이터량
```
실용적 학습을 위한 권장 데이터:
- 각 카테고리별 최소 1,000개 이상 프레임
- 전체 최소 6,000개 프레임 (6개 카테고리)
- 논문 수준: 각 카테고리 8,500개 = 총 51,000개 프레임
```

### 🎯 GitHub 코드의 실제 목적
- **프레임워크 제공**: 코드 구조 및 실행 방법 시연
- **데모 실행**: 작은 데이터셋으로 파이프라인 테스트
- **연구 재현**: 실제 데이터는 별도 확보 필요

---

## 3. AI Hub 원본 데이터 다운로드 방법

### 🔗 공식 다운로드 경로

**AI Hub 이상행동 CCTV 영상 데이터셋**
- **URL**: https://aihub.or.kr/aidata/139
- **대체 URL**: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=171

### 📊 원본 데이터 사양
```
총 용량: 약 5TB
총 영상 수: 8,436개 MP4 파일
총 시간: 717시간
해상도: 3840×2160 (4K)
범죄 유형: 12가지 (폭행, 싸움, 절도, 기물파손, 실신, 배회, 침입, 투기, 강도, 데이트폭력, 납치, 주취)
파일 구성: MP4 영상 + XML 라벨 파일
```

### 📋 다운로드 절차
1. **AI Hub 회원가입** (https://aihub.or.kr)
2. **데이터 신청서 제출** 및 승인 대기 (1-2일)
3. **AI Hub 전용 다운로드 프로그램 설치**
4. **선택적 다운로드** (전체 5TB 또는 필요한 부분만)
5. **GitHub 코드로 전처리 실행**

### 💡 부분 다운로드 전략
```bash
# 메모리/저장공간 절약을 위한 선택적 다운로드
- 각 범죄 유형별 100-200개 영상만 선택
- 약 500GB-1TB 정도로 용량 절약
- 여전히 수만 개 프레임 확보 가능
```

---

## 4. 학습용 vs 추론용 코드 차이점

### 🔍 현재 GitHub 코드 분석: `bin_cnn.py`

**학습용 코드의 특징**:
```python
# ✅ 학습 구성 요소들
data = NormalDataset(annotations_file="./train_annotation.csv")  # 학습 데이터
trainloader = DataLoader(trainset, batch_size=128, shuffle=True) # 학습 로더
optimizer = Adam(model.parameters(), lr=learning_rate)           # 옵티마이저
loss_function = nn.CrossEntropyLoss()                           # 손실함수

def train(model, params):  # 🎯 학습 함수
    for epoch in range(num_epochs):
        model.train()                    # 학습 모드
        # 역전파 및 가중치 업데이트
        torch.save(model.state_dict(), './weight_cnn_bin')  # 모델 저장
```

### ❌ 현재 GitHub에 없는 것들
- **추론(Inference) 코드**: 학습된 모델로 새 영상 분석
- **실시간 CCTV 연결**: RTSP 스트림 처리
- **웹/앱 인터페이스**: 영상 업로드 UI
- **배포용 API**: REST API 서버

### 🚀 추론용 코드 예시 (직접 구현 필요)

```python
# 학습된 모델로 새 영상 분석하는 코드 (GitHub에 없음)
import torch
import cv2
from torchvision import transforms

class CrimeDetector:
    def __init__(self, model_path):
        # 1. 학습된 모델 로드
        self.model = CNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 추론 모드
        
        # 2. 전처리 파이프라인
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    
    def predict_frame(self, frame):
        """단일 프레임 범죄 예측"""
        image = self.preprocess(frame).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities).item()
        
        labels = ['Crime', 'Normal']
        return labels[prediction], probabilities[0][prediction].item()
    
    def analyze_video(self, video_path):
        """전체 영상 분석"""
        cap = cv2.VideoCapture(video_path)
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            prediction, confidence = self.predict_frame(frame)
            results.append({
                'frame': len(results),
                'prediction': prediction,
                'confidence': confidence
            })
            
            # 실시간 결과 표시
            color = (0,0,255) if prediction == 'Crime' else (0,255,0)
            cv2.putText(frame, f'{prediction}: {confidence:.2f}', 
                       (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Crime Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return results

# 사용 예시
detector = CrimeDetector('./weight_cnn_bin')
results = detector.analyze_video('업로드한영상.mp4')
```

---

## 5. 완전한 사용 가이드

### 📋 전체 워크플로우

#### 1단계: 환경 설정
```bash
# 필수 패키지 설치
pip install torch torchvision opencv-python pandas numpy tqdm matplotlib
pip install moviepy imageio-ffmpeg beautifulsoup4 lxml

# Repository 클론
git clone https://github.com/dxlabskku/CRxK-6.git
cd CRxK-6
```

#### 2단계: 데이터 준비
```bash
# Option A: AI Hub 원본 데이터 사용 (권장)
# 1. AI Hub에서 데이터 다운로드
# 2. ./data/videos/ 폴더에 MP4 파일들 배치
# 3. 프레임 추출
python preprocess/frame_crop.py --path ./data --category assault
python preprocess/frame_crop.py --path ./data --category burglary
# ... 각 카테고리별 실행

# Option B: 샘플 데이터로 테스트 (현재 GitHub)
# 이미 frame_data 폴더에 600개 샘플 준비됨
```

#### 3단계: 어노테이션 생성
```bash
python preprocess/make_annotation.py --path ./data
# → train_annotation.csv 파일 생성
```

#### 4단계: 모델 학습
```bash
# 이진 분류 (범죄 vs 정상)
python models/bin_res18.py     # ResNet-18 (권장)
python models/bin_eff.py       # EfficientNet-B0
python models/bin_cnn.py       # Custom CNN

# 다중 분류 (5개 범죄 유형)
python models/multi_res18.py   # ResNet-18
python models/multi_eff.py     # EfficientNet-B0

# 2단계 파이프라인
python models/bin_multi_res18.py
```

#### 5단계: 추론 시스템 구축 (직접 구현)
```python
# inference.py (새로 작성 필요)
from crime_detector import CrimeDetector

detector = CrimeDetector('./weight_cnn_bin')
results = detector.analyze_video('test_video.mp4')
```

### 🎯 성능 최적화 방안

```python
# GPU 가속 활용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 최적화
# 1. TensorRT 변환으로 추론 속도 향상
# 2. Mixed Precision (FP16) 사용으로 메모리 절약
# 3. 배치 처리로 처리량 증대

# 실시간 처리 최적화
# - 프레임 스킵 (매 3프레임마다 분석)
# - 다중 스레드 처리
# - GPU 메모리 효율적 사용
```

---

## 6. 실무 적용 방안

### 🏢 CCTV 실시간 모니터링 시스템

```python
# rtsp_monitor.py (실제 CCTV 연결)
import cv2

class RTSPCrimeMonitor:
    def __init__(self, rtsp_url, model_path):
        self.rtsp_url = rtsp_url
        self.detector = CrimeDetector(model_path)
        self.cap = cv2.VideoCapture(rtsp_url)
    
    def start_monitoring(self):
        """실시간 CCTV 모니터링 시작"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("스트림 연결 실패, 재시도...")
                continue
            
            prediction, confidence = self.detector.predict_frame(frame)
            
            if prediction == 'Crime' and confidence > 0.8:
                self.send_alert(frame, confidence)
            
            self.display_frame(frame, prediction, confidence)
    
    def send_alert(self, frame, confidence):
        """위험 상황 알림"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"🚨 ALERT: 범죄 의심 행동 감지! ({confidence:.1%}) - {timestamp}")
        
        # 이미지 저장
        cv2.imwrite(f'alert_{timestamp}.jpg', frame)
        
        # 관리자에게 알림 전송 (이메일, SMS, 웹훅 등)
        # send_notification(frame, confidence, timestamp)

# 사용 예시
monitor = RTSPCrimeMonitor(
    rtsp_url='rtsp://admin:password@192.168.1.100:554/stream1',
    model_path='./weight_cnn_bin'
)
monitor.start_monitoring()
```

### 📱 웹 애플리케이션 구축

```python
# flask_app.py (웹 인터페이스)
from flask import Flask, request, render_template
import os

app = Flask(__name__)
detector = CrimeDetector('./weight_cnn_bin')

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video_file = request.files['video']
        
        # 임시 저장
        temp_path = f'./temp/{video_file.filename}'
        video_file.save(temp_path)
        
        # 분석 실행
        results = detector.analyze_video(temp_path)
        
        # 결과 반환
        return render_template('results.html', results=results)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
```

### 💡 확장 가능한 아키텍처

```
                    📹 CCTV Cameras
                         │
                    🌐 RTSP Streams
                         │
              ┌──────────┼──────────┐
              │          │          │
         🖥️ GPU Server  🖥️ GPU Server  🖥️ GPU Server
         (AI 모델 실행)  (AI 모델 실행)  (AI 모델 실행)
              │          │          │
              └──────────┼──────────┘
                         │
                  📊 Central Database
                     (MongoDB)
                         │
              ┌──────────┼──────────┐
              │          │          │
         💻 Web Dashboard  📱 Mobile App  🚨 Alert System
         (관리자 모니터링)  (현장 담당자)   (자동 신고)
```

---

## 📋 체크리스트 및 주의사항

### ✅ 성공적 구축을 위한 체크리스트

**데이터 준비**:
- [ ] AI Hub 계정 생성 및 데이터 신청
- [ ] 최소 1TB 이상 저장공간 확보
- [ ] GPU 환경 준비 (NVIDIA RTX 3070 이상 권장)

**코드 구현**:
- [ ] GitHub 학습 코드 실행 테스트
- [ ] 추론 파이프라인 직접 구현
- [ ] 실시간 처리 최적화

**시스템 통합**:
- [ ] CCTV RTSP 연결 테스트
- [ ] 알림 시스템 구축
- [ ] 웹 인터페이스 개발

### ⚠️ 주요 주의사항

1. **데이터 부족 문제**: GitHub 샘플만으로는 실용 불가
2. **추론 코드 부재**: 학습 후 별도 구현 필요
3. **실시간 성능**: GPU 없이는 실시간 처리 어려움
4. **정확도 한계**: 재연 데이터와 실제 환경의 차이
5. **개인정보 보호**: 얼굴 마스킹 및 데이터 암호화 필수

---

## 🎯 결론 및 권장사항

### 현실적 구현 전략

**단계별 접근**:
1. **Phase 1**: GitHub 코드로 학습 파이프라인 이해
2. **Phase 2**: AI Hub 부분 데이터로 프로토타입 구축
3. **Phase 3**: 전체 데이터로 실용 모델 학습
4. **Phase 4**: 실시간 시스템 구축 및 배포

**권장 모델**:
- **개발 단계**: ResNet-18 (빠른 실험)
- **배포 단계**: EfficientNet-B0 (속도-성능 균형)
- **고성능 필요시**: ResNet-50 (높은 정확도)

**현실적 한계 인지**:
- CRxK-6 GitHub = 학습 프레임워크 제공
- 실제 사용 = AI Hub 데이터 + 추가 개발 필요
- 상용화 = 대량 실제 데이터로 재학습 권장

이 가이드를 통해 CRxK-6를 활용한 완전한 범죄 감지 시스템을 구축할 수 있습니다. 단계별로 차근차근 진행하면서 각 단계의 한계와 요구사항을 명확히 이해하는 것이 성공의 열쇠입니다.
