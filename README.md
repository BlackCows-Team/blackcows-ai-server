# BlackCows AI 예측 서버

젖소 착유량 및 유방염 예측을 위한 경량화된 AI 모델 서버입니다.

## 🗂️ 구현된 AI 예측 API 요약

| Method | Endpoint | 설명 | 필수 필드 | 옵션 필드 | 응답 주요 필드 |
|--------|----------|------|----------|-----------|----------------|
| POST | /ai/milk-yield/predict | 개별 젖소 착유량 예측 | milking_frequency, conductivity, temperature, fat_percentage, protein_percentage, concentrate_intake, milking_month, milking_day_of_week | cow_id, prediction_date, notes | predicted_milk_yield, confidence, input_features, model_version, prediction_time |
| POST | /ai/mastitis/predict | 개별 젖소 유방염 예측 | milk_yield, conductivity, fat_percentage, protein_percentage, lactation_number | cow_id, prediction_date, notes | prediction_class, prediction_class_label, confidence, input_features, model_version |
| POST | /ai/mastitis/predict-by-scc | 체세포수 기반 유방염 예측 | somatic_cell_count | cow_id, measurement_date, notes | prediction_class, prediction_class_label, confidence, description, input_features, classification_criteria |
| POST | /ai/milk-yield/batch-predict | 다중 젖소 착유량 예측 | predictions (MilkYieldPredictionRequest 배열) | batch_name | predictions, batch_id, total_predictions |
| POST | /ai/mastitis/batch-predict | 다중 젖소 유방염 예측 | predictions (MastitisPredictionRequest 배열) | batch_name | predictions, batch_id, total_predictions |
| POST | /ai/mastitis/batch-predict-by-scc | 다중 젖소 체세포수 기반 유방염 예측 | predictions (SomaticCellCountPredictionRequest 배열) | batch_name | predictions, batch_id, total_predictions |
| GET | /ai/mastitis/scc-classification-info | 체세포수 분류 기준 정보 | 없음 | 없음 | criteria, notes, references |
| GET | /ai/model-health | 모델 상태 확인 | 없음 | 없음 | status, checks, model_info |

---

## 📌 각 엔드포인트별 요청/응답 예시

### 1. 개별 젖소 착유량 예측
- **POST** `/ai/milk-yield/predict`
- **요청 예시**
```json
{
  "cow_id": "cow_123",
  "milking_frequency": 2,
  "conductivity": 7.5,
  "temperature": 38.5,
  "fat_percentage": 3.8,
  "protein_percentage": 3.2,
  "concentrate_intake": 3.5,
  "milking_month": 6,
  "milking_day_of_week": 1,
  "prediction_date": "2024-06-01",
  "notes": "테스트"
}
```
- **응답 예시**
```json
{
  "prediction_id": "uuid-123",
  "cow_id": "cow_123",
  "predicted_milk_yield": 25.5,
  "confidence": 85.2,
  "input_features": {
    "착유횟수": 2,
    "전도율": 7.5,
    "온도": 38.5,
    "유지방비율": 3.8,
    "유단백비율": 3.2,
    "농후사료섭취량": 3.5,
    "착유기측정월": 6,
    "착유기측정요일": 1
  },
  "model_version": "v2.0.0",
  "prediction_time": "2024-06-01T10:30:00",
  "processing_time_ms": 45.2
}
```

---

### 2. 개별 젖소 유방염 예측
- **POST** `/ai/mastitis/predict`
- **요청 예시**
```json
{
  "cow_id": "cow_123",
  "milk_yield": 25.5,
  "conductivity": 7.5,
  "fat_percentage": 3.8,
  "protein_percentage": 3.2,
  "lactation_number": 2,
  "prediction_date": "2024-06-01",
  "notes": "테스트"
}
```
- **응답 예시**
```json
{
  "prediction_id": "uuid-456",
  "cow_id": "cow_123",
  "prediction_class": 1,
  "prediction_class_label": "주의",
  "confidence": 92.1,
  "input_features": {
    "착유량": 25.5,
    "전도율": 7.5,
    "유지방비율": 3.8,
    "유단백비율": 3.2,
    "산차수": 2
  },
  "model_version": "mastitis_rf_v1",
  "prediction_time": "2024-06-01T10:31:00",
  "processing_time_ms": 38.7
}
```

---

### 3. 체세포수 기반 유방염 예측
- **POST** `/ai/mastitis/predict-by-scc`
- **요청 예시**
```json
{
  "cow_id": "cow_123",
  "somatic_cell_count": 150,
  "measurement_date": "2024-06-01",
  "notes": "테스트"
}
```
- **응답 예시**
```json
{
  "prediction_id": "uuid-789",
  "cow_id": "cow_123",
  "prediction_method": "somatic_cell_count",
  "prediction_class": 1,
  "prediction_class_label": "주의",
  "confidence": 95.0,
  "description": "체세포수가 약간 증가한 상태입니다. 주의 깊은 관찰이 필요합니다.",
  "input_features": {
    "체세포수": 150,
    "단위": "개/ml"
  },
  "classification_criteria": {
    "정상": "≤ 100개/ml",
    "주의": "101-300개/ml",
    "염증_가능성": "> 300개/ml"
  },
  "prediction_time": "2024-06-01T10:32:00",
  "processing_time_ms": 5.1
}
```

---

### 4. 다중 젖소 착유량 예측
- **POST** `/ai/milk-yield/batch-predict`
- **요청 예시**
```json
{
  "predictions": [
    {
      "cow_id": "cow_123",
      "milking_frequency": 2,
      "conductivity": 7.5,
      "temperature": 38.5,
      "fat_percentage": 3.8,
      "protein_percentage": 3.2,
      "concentrate_intake": 3.5,
      "milking_month": 6,
      "milking_day_of_week": 1
    },
    {
      "cow_id": "cow_456",
      "milking_frequency": 3,
      "conductivity": 8.0,
      "temperature": 38.2,
      "fat_percentage": 4.0,
      "protein_percentage": 3.5,
      "concentrate_intake": 4.0,
      "milking_month": 6,
      "milking_day_of_week": 2
    }
  ],
  "batch_name": "테스트 배치"
}
```
- **응답 예시**
```json
{
  "batch_id": "uuid-batch-1",
  "total_predictions": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "predictions": [
    { /* 개별 예측 결과 */ },
    { /* 개별 예측 결과 */ }
  ],
  "batch_created_at": "2024-06-01T10:33:00",
  "total_processing_time_ms": 90.5
}
```

---

### 5. 다중 젖소 유방염 예측
- **POST** `/ai/mastitis/batch-predict`
- **요청 예시**
```json
{
  "predictions": [
    {
      "cow_id": "cow_123",
      "milk_yield": 25.5,
      "conductivity": 7.5,
      "fat_percentage": 3.8,
      "protein_percentage": 3.2,
      "lactation_number": 2
    },
    {
      "cow_id": "cow_456",
      "milk_yield": 28.0,
      "conductivity": 8.0,
      "fat_percentage": 4.0,
      "protein_percentage": 3.5,
      "lactation_number": 3
    }
  ],
  "batch_name": "유방염 배치"
}
```
- **응답 예시**
```json
{
  "batch_id": "uuid-batch-2",
  "total_predictions": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "predictions": [
    { /* 개별 예측 결과 */ },
    { /* 개별 예측 결과 */ }
  ],
  "batch_created_at": "2024-06-01T10:34:00",
  "total_processing_time_ms": 80.2
}
```

---

### 6. 다중 젖소 체세포수 기반 유방염 예측
- **POST** `/ai/mastitis/batch-predict-by-scc`
- **요청 예시**
```json
{
  "predictions": [
    {
      "cow_id": "cow_123",
      "somatic_cell_count": 150
    },
    {
      "cow_id": "cow_456",
      "somatic_cell_count": 350
    }
  ],
  "batch_name": "체세포수 배치"
}
```
- **응답 예시**
```json
{
  "batch_id": "uuid-batch-3",
  "prediction_method": "somatic_cell_count_batch",
  "total_predictions": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "predictions": [
    { /* 개별 예측 결과 */ },
    { /* 개별 예측 결과 */ }
  ],
  "batch_created_at": "2024-06-01T10:35:00",
  "total_processing_time_ms": 12.3,
  "average_processing_time_ms": 6.15
}
```

---

### 7. 체세포수 분류 기준 정보
- **GET** `/ai/mastitis/scc-classification-info`
- **응답 예시**
```json
{
  "classification_method": "somatic_cell_count",
  "unit": "개/ml",
  "criteria": {
    "정상": {
      "range": "≤ 100",
      "class": 0,
      "description": "체세포수가 정상 범위로 건강한 상태",
      "color": "green",
      "action": "정기 모니터링 지속"
    },
    "주의": {
      "range": "101-300",
      "class": 1,
      "description": "체세포수가 약간 증가한 상태로 주의 필요",
      "color": "yellow",
      "action": "위생 관리 강화 및 모니터링"
    },
    "염증_가능성": {
      "range": "> 300",
      "class": 2,
      "description": "체세포수가 높아 유방염 의심",
      "color": "red",
      "action": "즉시 수의사 진료 필요"
    }
  },
  "notes": [
    "체세포수는 우유 1ml당 체세포의 개수를 나타냅니다",
    "체세포수가 높을수록 유방염 가능성이 증가합니다",
    "이 기준은 일반적인 가이드라인이며, 수의사의 전문적인 진단이 필요합니다",
    "개체별, 환경별 차이를 고려하여 종합적으로 판단해야 합니다"
  ],
  "references": [
    "대한수의사회 유방염 진단 가이드라인",
    "낙농진흥회 우유 품질 관리 기준"
  ]
}
```

---

### 8. 모델 상태 확인
- **GET** `/ai/model-health`
- **응답 예시**
```json
{
  "status": "healthy",
  "message": "AI 예측 서비스가 정상 작동 중입니다",
  "timestamp": "2024-06-01T10:36:00",
  "response_time_ms": 12.5,
  "checks": {
    "model_file_exists": true,
    "scaler_file_exists": true,
    "model_load_success": true,
    "prediction_test_success": true,
    "cache_loaded": true
  },
  "model_info": {
    "version": "v2.0.0",
    "cached": true,
    "available": true
  }
}
```

---

## 📋 프로젝트 개요

이 서버는 **AI 모델 예측만을 담당하는 전용 서버**로, 다음과 같은 특징을 가집니다:

- **목적**: 젖소 착유량 예측 AI 모델 제공
- **인프라**: 로컬 PC (Ryzen 7 5800X / 32GB RAM / RTX 4080 Super)
- **접근**: Cloudflare Tunnel을 통한 `ai.blackcowsdairy.com` 도메인
- **보안**: AWS EC2 메인 서버(`api.blackcowsdairy.com`)에서만 접근 가능
- **특징**: 인증/DB 없는 순수 모델 추론 서버

## 🏗️ 아키텍처

```
외부 사용자 → AWS EC2 (api.blackcowsdairy.com) → Cloudflare Tunnel → 로컬 AI 서버 (ai.blackcowsdairy.com)
```

- **메인 서버**: AWS EC2에서 운영 중인 FastAPI 서버
- **AI 서버**: 로컬 PC에서 운영하는 AI 모델 전용 서버
- **통신**: EC2에서만 AI 서버로 요청 전송

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 파일 준비

`models/` 디렉토리에 다음 파일들이 필요합니다:
- `1_random_forest.pkl` - 학습된 Random Forest 모델
- `1_scaler.pkl` - 특성 스케일링 모델

### 3. 서버 실행

```bash
python main.py
```

서버가 `http://localhost:8000`에서 실행됩니다.

## 📚 API 엔드포인트

### 🤖 AI 예측 API

#### 1. 개별 젖소 착유량 예측
```http
POST /ai/milk-yield/predict
```

**요청 본문:**
```json
{
  "cow_id": "cow_123",
  "milking_frequency": 2,
  "conductivity": 7.5,
  "temperature": 38.5,
  "fat_percentage": 3.8,
  "protein_percentage": 3.2,
  "concentrate_intake": 3.5,
  "milking_month": 6,
  "milking_day_of_week": 1
}
```
- **응답 주요 필드**: predicted_milk_yield, confidence, input_features, model_version, prediction_time

---

### 2. 개별 젖소 유방염 예측
- **POST** `/ai/mastitis/predict`
- **요청 예시**
```json
{
  "prediction_id": "uuid-123",
  "cow_id": "cow_123",
  "milk_yield": 25.5,
  "conductivity": 7.5,
  "fat_percentage": 3.8,
  "protein_percentage": 3.2,
  "lactation_number": 2
}
```
- **응답 주요 필드**: prediction_class, prediction_class_label, confidence, input_features, model_version

#### 2. 다중 젖소 일괄 예측
```http
POST /ai/milk-yield/batch-predict
```

**요청 본문:**
```json
{
  "predictions": [
    {
      "cow_id": "cow_123",
      "milking_frequency": 2,
      "conductivity": 7.5,
      "temperature": 38.5,
      "fat_percentage": 3.8,
      "protein_percentage": 3.2,
      "concentrate_intake": 3.5,
      "milking_month": 6,
      "milking_day_of_week": 1
    },
    {
      "cow_id": "cow_456",
      "milking_frequency": 3,
      "conductivity": 8.0,
      "temperature": 38.2,
      "fat_percentage": 4.0,
      "protein_percentage": 3.5,
      "concentrate_intake": 4.0,
      "milking_month": 6,
      "milking_day_of_week": 2
    }
  ]
}
```

#### 3. 모델 상태 확인
```http
GET /ai/model-health
```

**응답:**
```json
{
  "status": "healthy",
  "message": "AI 예측 서비스가 정상 작동 중입니다",
  "timestamp": "2024-01-15T10:30:00",
  "response_time_ms": 12.5,
  "checks": {
    "model_file_exists": true,
    "scaler_file_exists": true,
    "model_load_success": true,
    "prediction_test_success": true,
    "cache_loaded": true
  },
  "model_info": {
    "version": "v1.0.0",
    "cached": true,
    "available": true
  }
}
```

### 🔧 시스템 API

#### 1. 서버 정보
```http
GET /
```

#### 2. 헬스체크
```http
GET /health
```

#### 3. API 문서
```http
GET /docs
```

## 🔒 보안 설정

### CORS 설정
현재 다음 도메인에서만 접근 가능:
- `https://api.blackcowsdairy.com` (메인 EC2 서버)
- `http://api.blackcowsdairy.com` (HTTP 버전)
- `http://localhost:8000` (로컬 테스트용)
- `http://127.0.0.1:8000` (로컬 테스트용)

## 📊 모델 정보

- **모델 타입**: Random Forest
- **입력 특성**: 8개 (착유횟수, 전도율, 온도, 유지방비율, 유단백비율, 농후사료섭취량, 착유기측정월, 착유기측정요일)
- **출력**: 예상 착유량 (리터)
- **확신도**: Random Forest 트리들의 예측 분산을 기반으로 계산

## 🛠️ 개발 환경

- **Python**: 3.11 ( 3.11 버전으로 개발되었습니다.)
- **FastAPI**: 웹 프레임워크
- **scikit-learn**: 머신러닝 모델
- **joblib**: 모델 직렬화
- **uvicorn**: ASGI 서버

## 📝 사용 예시

### Python 클라이언트 예시
```python
import requests

# 개별 예측
response = requests.post(
    "http://localhost:8000/ai/milk-yield/predict",
    json={
        "cow_id": "cow_123",
        "milking_frequency": 2,
        "conductivity": 7.5,
        "temperature": 38.5,
        "fat_percentage": 3.8,
        "protein_percentage": 3.2,
        "concentrate_intake": 3.5,
        "milking_month": 6,
        "milking_day_of_week": 1
    }
)

print(response.json())
```

### cURL 예시
```bash
curl -X POST "http://localhost:8000/ai/milk-yield/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "cow_id": "cow_123",
       "milking_frequency": 2,
       "conductivity": 7.5,
       "temperature": 38.5,
       "fat_percentage": 3.8,
       "protein_percentage": 3.2,
       "concentrate_intake": 3.5,
       "milking_month": 6,
       "milking_day_of_week": 1
     }'
```

## 🔧 문제 해결

### 모델 로드 실패
- `models/` 디렉토리에 모델 파일이 있는지 확인
- 파일 권한 확인
- 메모리 부족 여부 확인

### CORS 오류
- 요청 도메인이 허용된 목록에 있는지 확인
- HTTPS/HTTP 프로토콜 일치 여부 확인

## 📞 지원

문제가 발생하면 다음을 확인해주세요:
1. `/health` 엔드포인트로 서버 상태 확인
2. `/ai/model-health` 엔드포인트로 모델 상태 확인
3. 서버 로그 확인

---

**참고**: 이 서버는 AWS EC2 메인 서버(`api.blackcowsdairy.com`)에서만 접근하도록 설계되었습니다.
