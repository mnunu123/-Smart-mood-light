# Smart Mood Light — 손동작 인식 무드등 제어

웹캠으로 손동작을 인식해 무드등의 색상/밝기를 제어하는 프로젝트입니다.
MediaPipe로 손 관절(landmark) 좌표를 추출하고, 관절 사이의 각도를 특징으로 삼아
Transformer 기반 시퀀스 분류 모델로 5가지 제스처를 인식합니다.

- **인식 제스처(5종)**: `red` / `green` / `blue` (색상 전환), `up` / `down` (밝기 조절)
- **Stack**: Python, OpenCV, MediaPipe Hands, TensorFlow/Keras (Transformer Encoder)

> ⚠️ **실제 구동 화면 캡처 불가 안내**: 이 프로젝트는 물리적 웹캠 입력이 필수라, 카메라 장치가 없는 서버/에이전트 환경에서는
> 실행할 수 없습니다. 대신 `train.ipynb`에 이미 기록된 실제 학습 로그(검증 정확도 100%, 50 epoch)가
> 저자 본인 PC에서 실제로 실행한 결과이니, 모델 동작 확인은 해당 노트북 셀 출력을 참고해주세요.

## 파이프라인

```
웹캠 프레임
  → MediaPipe Hands로 21개 손 관절 좌표(x,y,z,visibility) 추출
  → 관절 간 벡터 → 각도(15개) 계산 (arccos of dot product)
  → 30프레임 시퀀스로 묶어 시계열 데이터 구성
  → Transformer Encoder(Self-Attention) → GlobalAveragePooling → Dense
  → 5개 제스처 클래스 중 하나로 분류 (연속 3프레임 이상 같은 예측일 때만 확정)
```

## 파일 구성

> ⚠️ 원본 파일들이 확장자 없이 커밋되어 있습니다. 아래는 실제로는 모두 Python(`.py`) / Jupyter(`.ipynb`) 스크립트입니다.

| 파일 | 역할 |
| --- | --- |
| `mediapipe` | MediaPipe Hands로 엄지-검지 거리를 측정해 특정 제스처(핀치)를 감지하는 초기 프로토타입 |
| `dataset code(seq)` | 웹캠으로 5개 제스처(`red/green/blue/down/up`)를 30초씩 촬영해 관절 각도 시퀀스(길이 30) 데이터셋(.npy)으로 저장 |
| `dataset code(vector)` | 마우스 클릭으로 한 프레임씩 라벨링해 관절 각도 벡터를 CSV로 누적 저장하는 대안 수집 방식 |
| `train.ipynb` | 수집된 시퀀스 데이터를 로드 → train/val 분할(9:1) → Transformer Encoder 모델 구성/학습 → `transformer_model.keras`로 저장 |
| `transformer test code` | 학습된 모델을 로드해 웹캠 실시간 입력으로 제스처를 추론(inference)하는 데모 스크립트 |
| `dataset.csv.zip` | 수집된 원본 시퀀스 데이터셋 (`seq_*.npy` 압축본) |

## 모델 구조

- 입력: `(30 프레임, 99 특징)` — 21개 관절 × (x,y,z,visibility) + 관절 간 각도
- Transformer Encoder 1개 블록 (Multi-Head Attention 4-head, head_size=64, ff_dim=64) + residual + LayerNorm
- `GlobalAveragePooling1D` → `Dense(32, relu)` → `Dense(5, softmax)`
- 검증 정확도 100% (자체 수집 데이터셋 기준, `train.ipynb` 실행 결과)

## 실행

```bash
pip install opencv-python mediapipe numpy tensorflow scikit-learn

# 1) 제스처 시퀀스 데이터 수집 (5종 제스처를 각 30초씩 촬영)
python "dataset code(seq).py"

# 2) train.ipynb 실행 — 모델 학습 후 transformer_model.keras 저장

# 3) 실시간 추론 데모
python "transformer test code.py"
```

> 스크립트 내 데이터/모델 경로가 원 개발 환경(`C:\Users\...\스마트무드등\...`) 기준 절대경로로 하드코딩되어 있어, 실행 전 각자 환경에 맞게 경로를 수정해야 합니다.
