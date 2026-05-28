# Tunnel Method Recommender

터널 조건 데이터를 기반으로 `NATM`, `EPB TBM`, `Slurry TBM`, `개착식`, `침매식` 중 적합한 터널 공법을 추천하는 Python 프로젝트입니다.

정규화된 과거 시공 사례를 학습 데이터로 사용하고, XGBoost 기반 공법별 점수 모델을 통해 입력 조건에 대한 공법별 적합도를 계산합니다. 이후 규칙 기반 필터링, 엑셀 결과 저장, Leave-One-Out 방식의 백테스트, 시방서 RAG 기반 최종 보고서 생성까지 수행할 수 있습니다.

---

## 1. 주요 기능

### 1) 데이터 정규화 및 가중치 분석

`scripts/run_weighting.py`는 원본 사례 데이터를 불러와 모델 입력용 데이터로 정리하고, 변수 중요도를 계산합니다.

수행 내용은 다음과 같습니다.

- 필수 컬럼 검증
- 결측치 처리
- 숫자형 변수 변환
- 범주형 변수 인코딩
- Mutual Information 기반 변수 가중치 계산
- XGBoost gain 기반 변수 중요도 계산
- 결과 엑셀 저장

생성 파일:

```text
data/interim/normalized_training_data.xlsx
data/interim/encoded_training_data.xlsx
data/interim/category_maps.json
data/outputs/weight_analysis.xlsx
```

---

### 2) 입력 조건 기반 공법 추천

`scripts/run_scoring.py`는 `data/input/input_case.xlsx`에 입력된 1개 프로젝트 조건을 읽고, 공법별 추천 점수를 계산합니다.

수행 흐름은 다음과 같습니다.

1. `data/raw/조사 자료 정규화.xlsx` 학습 데이터 로드
2. `config/feature_schema.yaml` 기준으로 사용할 변수와 공법 목록 로드
3. 학습 데이터 정규화
4. 공법별 이진 XGBoost 모델 학습
5. 입력 조건에 대해 각 공법의 raw score 계산
6. calibration table을 이용해 100점 만점 점수로 변환
7. 적용 가능성 규칙 반영
8. 결과를 `data/outputs/result.xlsx`로 저장

생성 파일:

```text
data/outputs/result.xlsx
```

`result.xlsx`에는 다음 시트가 생성됩니다.

| 시트명 | 내용 |
|---|---|
| `input_data` | 사용자가 입력한 프로젝트 조건 |
| `result` | 공법별 raw score, 100점 환산 점수, 판정, 비고 |

---

### 3) 적용 가능성 판정 규칙

공법별 기본 판정은 100점 환산 점수를 기준으로 결정됩니다.

| 점수 구간 | 판정 |
|---:|---|
| 85점 이상 | 적합 |
| 60점 이상 85점 미만 | 가능 |
| 60점 미만 | 추천하지 않음 |

추가 규칙:

- `침매식`은 입력 조건의 `시공 위치_정규화`가 수중 조건이 아닐 경우 `적용 곤란`으로 처리됩니다.
- 수중 조건으로 인정되는 값은 다음과 같습니다.

```text
수중, 해저, 하천, 호수, 수중구간
```

즉, 침매식 점수가 높게 나와도 시공 위치가 수중 계열이 아니면 최종 점수는 0점으로 조정됩니다.

---

### 4) Leave-One-Out 백테스트

`scripts/backtest.py`는 전체 사례 데이터에서 하나의 사례를 테스트 데이터로 빼고, 나머지 데이터로 모델을 학습한 뒤 뺀 사례의 실제 공법을 맞히는 방식으로 검증합니다.

예를 들어 전체 데이터가 56개라면 다음 과정을 56번 반복합니다.

1. 1개 사례 제외
2. 나머지 55개 사례로 모델 학습
3. 제외한 1개 사례의 조건으로 공법 추천
4. 실제 적용 공법과 예측 결과 비교
5. 전체 정답률 계산

정답 인정 기준은 다음과 같습니다.

| 기준 | 인정 여부 |
|---|---|
| 실제 공법이 1순위 예측 공법과 같음 | 정답 |
| 실제 공법이 2순위이고, 1순위와 2순위 점수가 같음 | 정답 인정 |
| 실제 공법이 2순위이고, 1순위와 2순위가 모두 `적합` 판정 | 정답 인정 |
| 실제 공법이 3순위이고, 3순위 판정이 `적합` 또는 `가능` | 정답 인정 |
| 그 외 | 오답 |

생성 파일:

```text
data/outputs/backtest_result.csv
```

현재 포함된 `backtest_result.csv` 기준 결과는 다음과 같습니다.

```text
전체 개수: 56
정답률: 87.50%
```

---

### 5) RAG 기반 최종 보고서 생성

`scripts/run_llm_report.py`는 `data/outputs/result.xlsx`의 정량 평가 결과와 `data/knowledge/specs_selected_txt/`에 있는 터널 기준/시방서 TXT 파일을 함께 사용해 최종 기술검토 보고서를 생성합니다.

생성 파일:

```text
data/outputs/llm_report/final_report.md
data/outputs/llm_report/final_report.json
```

보고서에는 다음 내용이 포함됩니다.

- 프로젝트 입력 조건 요약
- 정량 평가 결과 해석
- 상위 공법 비교
- 시방서/기준 기반 검토사항
- 최종 추천 공법
- 추가 확인 필요사항
- 보고서 한계

침매식 관련 특수 로직도 포함되어 있습니다.

- 침매식이 정량 결과 기준 1위 또는 2위일 때만 침매공법 관련 TXT 문서를 강제로 읽습니다.
- 침매식이 하위권이면 침매 문서를 억지로 보고서에 반영하지 않습니다.

---

## 2. 프로젝트 구조

```text
Tunnel Method Recommender/
├─ config/
│  ├─ feature_schema.yaml          # 사용할 변수, 공법 목록, 필수 컬럼 정의
│  └─ settings.yaml                # 일부 경로/설정값
│
├─ data/
│  ├─ raw/
│  │  └─ 조사 자료 정규화.xlsx      # 학습용 정규화 사례 데이터
│  ├─ input/
│  │  └─ input_case.xlsx            # 추천할 신규 프로젝트 조건
│  ├─ interim/
│  │  ├─ normalized_training_data.xlsx
│  │  ├─ encoded_training_data.xlsx
│  │  └─ category_maps.json
│  ├─ outputs/
│  │  ├─ result.xlsx
│  │  ├─ weight_analysis.xlsx
│  │  ├─ backtest_result.csv
│  │  └─ llm_report/
│  │     ├─ final_report.md
│  │     └─ final_report.json
│  └─ knowledge/
│     └─ specs_selected_txt/        # RAG 보고서 생성에 사용하는 기준/시방서 TXT
│
├─ scripts/
│  ├─ run_weighting.py              # 데이터 정규화 및 변수 중요도 분석
│  ├─ run_scoring.py                # 입력 조건 기반 공법 추천
│  ├─ backtest.py                   # Leave-One-Out 백테스트
│  └─ run_llm_report.py             # RAG 기반 최종 보고서 생성
│
├─ src/
│  ├─ preprocessing/normalizer.py   # 데이터 로딩, 정제, 인코딩
│  ├─ weighting/weight_calculator.py# MI/XGB 변수 중요도 계산
│  ├─ inference/model_trainer.py    # 공법별 XGBoost 모델 학습
│  ├─ inference/scorer.py           # 입력 조건 점수 계산
│  ├─ inference/calibration.py      # raw score를 100점 점수로 변환
│  ├─ inference/rule_filter.py      # 적용 가능성 규칙 반영
│  ├─ reporting/export_excel.py     # 가중치 분석 엑셀 저장
│  ├─ reporting/export_score_excel.py# 추천 결과 엑셀 저장
│  └─ llm/rag_reporter.py           # OpenAI file_search 기반 보고서 생성
│
├─ requirements.txt
└─ README.md
```

---

## 3. 설치 방법

Python 3.10 이상 사용을 권장합니다.

```bash
pip install -r requirements.txt
```

필요 패키지:

```text
pandas
openpyxl
scikit-learn
pyyaml
xgboost
```

RAG 보고서 생성까지 사용할 경우 `openai` 패키지도 필요합니다.

```bash
pip install openai
```

---

## 4. 실행 순서

프로젝트 루트 폴더에서 아래 명령어를 실행합니다.

### 1) 정규화 및 변수 중요도 분석

```bash
python scripts/run_weighting.py
```

결과:

```text
data/interim/normalized_training_data.xlsx
data/interim/encoded_training_data.xlsx
data/interim/category_maps.json
data/outputs/weight_analysis.xlsx
```

---

### 2) 신규 조건 공법 추천

먼저 `data/input/input_case.xlsx`에 추천할 프로젝트 조건을 입력합니다.

필요 컬럼:

```text
심도(m)_정규화
구경(m)_정규화
터널 길이(km)_정규화
지반 조건_정규화
시공 위치_정규화
주변 민감도_정규화
요구사항_주요정규화
요구사항_보조정규화
```

실행:

```bash
python scripts/run_scoring.py
```

결과:

```text
data/outputs/result.xlsx
```

---

### 3) 백테스트 실행

```bash
python scripts/backtest.py
```

결과:

```text
data/outputs/backtest_result.csv
```

터미널에는 전체 개수, 정답 개수, 오답 개수, 정답률이 `%` 단위로 출력됩니다.

---

### 4) RAG 최종 보고서 생성

먼저 OpenAI API 키를 환경변수로 설정합니다.

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="본인_API_KEY"
```

macOS/Linux:

```bash
export OPENAI_API_KEY="본인_API_KEY"
```

그 다음 실행합니다.

```bash
python scripts/run_llm_report.py
```

기본 입력:

```text
data/outputs/result.xlsx
data/knowledge/specs_selected_txt/*.txt
```

기본 출력:

```text
data/outputs/llm_report/final_report.md
data/outputs/llm_report/final_report.json
```

모델을 바꾸고 싶으면 다음처럼 실행할 수 있습니다.

```bash
python scripts/run_llm_report.py --model gpt-4.1-mini
```

---

## 5. 설정 파일 설명

### `config/feature_schema.yaml`

모델이 사용할 핵심 설정 파일입니다.

현재 target은 다음과 같습니다.

```yaml
target: 적용 공법
```

현재 추천 대상 공법은 다음 5개입니다.

```yaml
methods:
  - NATM
  - EPB TBM
  - Slurry TBM
  - 개착식
  - 침매식
```

숫자형 변수:

```yaml
numeric_features:
  - 심도(m)_정규화
  - 구경(m)_정규화
  - 터널 길이(km)_정규화
```

범주형 변수:

```yaml
categorical_features:
  - 지반 조건_정규화
  - 시공 위치_정규화
  - 주변 민감도_정규화
  - 요구사항_주요정규화
  - 요구사항_보조정규화
```

`required_columns`에 있는 컬럼이 학습 데이터에 없으면 실행 중 오류가 발생합니다.

---

## 6. 모델 및 점수 산정 방식

이 프로젝트는 하나의 다중분류 모델이 아니라, 공법별 이진 분류 모델을 따로 학습합니다.

예를 들어 다음과 같은 방식입니다.

| 모델 | 학습 방식 |
|---|---|
| NATM 모델 | 실제 공법이 NATM이면 1, 아니면 0 |
| EPB TBM 모델 | 실제 공법이 EPB TBM이면 1, 아니면 0 |
| Slurry TBM 모델 | 실제 공법이 Slurry TBM이면 1, 아니면 0 |
| 개착식 모델 | 실제 공법이 개착식이면 1, 아니면 0 |
| 침매식 모델 | 실제 공법이 침매식이면 1, 아니면 0 |

각 모델은 입력 조건에 대해 `predict_proba` 값을 출력하고, 이 값이 `raw_score`가 됩니다.

단, raw score를 그대로 쓰지 않고 3-fold OOF 점수 분포를 기준으로 calibration table을 만든 뒤 100점 만점 점수로 변환합니다.

점수 변환 기준:

| raw score 위치 | 환산 점수 |
|---|---:|
| p10 이하 | 10 |
| p25 이하 | 30 |
| p50 이하 | 50 |
| p75 이하 | 70 |
| p90 이하 | 85 |
| p90 초과 | 95 |

---

## 7. 주의사항

- 학습 데이터 파일명은 코드 기준으로 `data/raw/조사 자료 정규화.xlsx`여야 합니다.
- 엑셀 컬럼명은 `config/feature_schema.yaml`의 컬럼명과 정확히 일치해야 합니다.
- `input_case.xlsx`는 현재 첫 번째 행만 읽어서 추천합니다.
- 데이터 수가 적으면 XGBoost 결과와 백테스트 정답률이 불안정할 수 있습니다.
- 백테스트는 Leave-One-Out 방식이라 데이터가 많아질수록 실행 시간이 길어집니다.
- RAG 보고서 생성은 OpenAI API 키와 인터넷 연결이 필요합니다.
- `run_llm_report.py`는 내부적으로 vector store를 생성하고 TXT 시방서 파일을 업로드합니다.
- 현재 `scripts/run_llm_report.py`에서는 `reuse_existing_vector_store=False`로 고정되어 있어 실행할 때마다 새 벡터스토어를 생성합니다.

---

## 8. 자주 발생할 수 있는 오류

### 1) 학습 데이터 파일을 찾을 수 없음

오류 예시:

```text
FileNotFoundError: 입력 파일을 찾을 수 없습니다: data/raw/조사 자료 정규화.xlsx
```

해결:

- `data/raw/` 폴더 안에 `조사 자료 정규화.xlsx` 파일이 있는지 확인합니다.
- 파일명이 다르면 코드에서 지정한 이름과 동일하게 변경합니다.

---

### 2) 필수 컬럼이 없다는 오류

오류 예시:

```text
ValueError: 필수 컬럼이 없습니다: [...]
```

해결:

- 학습 데이터 엑셀의 컬럼명이 `config/feature_schema.yaml`의 `required_columns`와 일치하는지 확인합니다.
- 공백, 괄호, `_정규화` 표기 차이도 오류 원인이 될 수 있습니다.

---

### 3) RAG 보고서 생성 시 API 키 오류

오류 원인:

- `OPENAI_API_KEY` 환경변수가 설정되지 않았습니다.

해결:

```bash
export OPENAI_API_KEY="본인_API_KEY"
```

또는 Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="본인_API_KEY"
```

---

## 9. 전체 실행 예시

```bash
# 1. 패키지 설치
pip install -r requirements.txt
pip install openai

# 2. 정규화 및 변수 중요도 분석
python scripts/run_weighting.py

# 3. 입력 조건 기반 공법 추천
python scripts/run_scoring.py

# 4. 백테스트
python scripts/backtest.py

# 5. RAG 최종 보고서 생성
python scripts/run_llm_report.py
```

---

## 10. 현재 프로그램 요약

이 프로그램은 단순히 점수가 높은 공법을 뽑는 코드가 아니라, 다음 흐름을 가진 터널 공법 추천 시스템입니다.

```text
정규화된 과거 사례 데이터
        ↓
전처리 및 범주형 인코딩
        ↓
공법별 XGBoost 모델 학습
        ↓
신규 프로젝트 조건 입력
        ↓
공법별 raw score 계산
        ↓
calibration 기반 100점 점수 변환
        ↓
수중 여부 등 적용 가능성 규칙 반영
        ↓
공법 추천 결과 엑셀 저장
        ↓
시방서 TXT + RAG 기반 최종 보고서 생성
```

최종적으로 사용자는 `input_case.xlsx`에 현장 조건을 입력하고 `run_scoring.py`를 실행하면, `result.xlsx`에서 공법별 추천 순위와 판정을 확인할 수 있습니다.
