# routers/ai_prediction.py

from fastapi import APIRouter, HTTPException, status
from schemas.ai_prediction import (
    MilkYieldPredictionRequest,
    MastitisPredictionRequest,
    PredictionBatchRequest,
    MastitisBatchRequest,
    SomaticCellCountPredictionRequest,
    SomaticCellCountBatchRequest
)
from services.ai_prediction_service import AIPredictionService


router = APIRouter(prefix="/ai", tags=["AI 예측"])

# 1. 착유량 예측
@router.post(
    "/milk-yield/predict",
    summary="착유량 예측",
    description="""
    젖소의 8개 특성(feature)을 바탕으로 착유량을 예측합니다.

    **필요한 입력:**
    - 착유횟수 (milking_frequency) [필수]
    - 전도율 (conductivity) [필수]
    - 온도 (temperature) [필수]
    - 유지방비율 (fat_percentage) [필수]
    - 유단백비율 (protein_percentage) [필수]
    - 농후사료섭취량 (concentrate_intake) [필수]
    - 착유기측정월 (milking_month) [필수]
    - 착유기측정요일 (milking_day_of_week) [필수]
    - 젖소 ID (cow_id) [선택]
    - 예측 기준일 (prediction_date) [선택]
    - 예측 관련 메모 (notes) [선택]

    **결과:**
    - 예상 착유량 (리터)
    """
)
async def predict_milk_yield(
    prediction_request: MilkYieldPredictionRequest
):
    """개별 젖소의 착유량을 예측합니다."""
    return await AIPredictionService.predict_milk_yield(prediction_request)

# 2. 유방염 예측
@router.post("/mastitis/predict",
             summary="유방염 예측",
             description="""
             젖소의 5개 특성(feature)을 바탕으로 유방염 위험도를 예측합니다.

             **필요한 입력:**
             - 착유량 (milk_yield) [필수]
             - 전도율 (conductivity) [필수]
             - 유지방비율 (fat_percentage) [필수]
             - 유단백비율 (protein_percentage) [필수]
             - 산차수 (lactation_number) [필수]
             - 젖소 ID (cow_id) [선택]
             - 예측 기준일 (prediction_date) [선택]
             - 예측 관련 메모 (notes) [선택]

             **결과:**
             - 예측 클래스 (prediction_class): 0=정상, 1=주의, 2=염증 가능성
             - 신뢰도 점수 (confidence)
             - 입력값 echo (input_features)
             - 예측 시간, 모델 버전 등 메타데이터
             """)
async def predict_mastitis(
    prediction_request: MastitisPredictionRequest
):
    """개별 젖소 유방염 예측"""
    return await AIPredictionService.predict_mastitis(prediction_request)

# 3. 체세포수 기반 유방염 예측
@router.post(
    "/mastitis/predict-by-scc",
    summary="체세포수 기반 유방염 예측",
    description="""
    체세포수를 바탕으로 유방염 위험도를 예측합니다.
    
    **분류 기준:**
    - 정상: ≤ 100개/ml (등급 0)
    - 주의: 101-300개/ml (등급 1)
    - 염증 가능성 + 유방염 의심: > 300개/ml (등급 2)
    
    **필요한 입력:**
    - 체세포수 (somatic_cell_count) [필수] - 개/ml 단위
    - 젖소 ID (cow_id) [선택]
    - 측정일 (measurement_date) [선택]
    - 메모 (notes) [선택]
    
    **결과:**
    - 예측 등급 (0: 정상, 1: 주의, 2: 염증 가능성)
    - 등급별 설명 및 권장사항
    - 분류 기준 정보
    """
)
async def predict_mastitis_by_scc(
    prediction_request: SomaticCellCountPredictionRequest
):
    """체세포수 기반 개별 젖소 유방염 예측"""
    return await AIPredictionService.predict_mastitis_by_scc(prediction_request)

# 4. 다중 착유량 예측
@router.post(
    "/milk-yield/batch-predict",
    summary="다중 젖소 착유량 예측",
    description="""
    여러 젖소의 예측 요청을 한번에 처리하여 결과를 반환합니다.
    
    **예측 항목:**
    - 리스트 형태의 `MilkYieldPredictionRequest` 입력
    """
)
async def predict_milk_yield_batch(
    batch_request: PredictionBatchRequest
):
    """여러 젖소의 착유량을 일괄 예측합니다."""
    return await AIPredictionService.predict_milk_yield_batch(batch_request)

# 5. 다중 유방염 예측
@router.post("/mastitis/batch-predict",
             summary="다중 젖소 유방염 예측",
             description="여러 젖소의 유방염 위험도를 한번에 예측합니다.")
async def predict_mastitis_batch(
    batch_request: MastitisBatchRequest
):
    """다중 젖소 유방염 일괄 예측"""
    return await AIPredictionService.predict_mastitis_batch(batch_request)

# 6. 다중 체세포수 기반 유방염 예측
@router.post(
    "/mastitis/batch-predict-by-scc", 
    summary="다중 젖소 체세포수 기반 유방염 예측",
    description="""
    여러 젖소의 체세포수를 바탕으로 유방염 위험도를 일괄 예측합니다.
    
    **배치 처리 제한:**
    - 최대 1000개까지 한 번에 처리 가능
    - 개별 실패 항목도 결과에 포함되어 반환
    
    **예측 항목:**
    - 리스트 형태의 `SomaticCellCountPredictionRequest` 입력
    """
)
async def predict_mastitis_scc_batch(
    batch_request: SomaticCellCountBatchRequest
):
    """다중 젖소 체세포수 기반 유방염 일괄 예측"""
    return await AIPredictionService.predict_mastitis_scc_batch(batch_request)

# 기타 엔드포인트 (순서 뒤로)
@router.get(
    "/model-health",
    summary="모델 상태 확인",
    description="""
    모델 및 스케일러 파일의 존재 여부, 예측 가능 여부 등을 확인합니다.
    
    반환값:
    - 상태(healthy, degraded, unavailable)
    - 모델 캐시 여부
    - 테스트 예측 성공 여부
    """
)
async def check_model_health():
    """모델 파일 존재, 로드 여부, 테스트 예측 가능 여부를 점검합니다."""
    return await AIPredictionService.check_model_health()

@router.get(
    "/mastitis/scc-classification-info",
    summary="체세포수 분류 기준 정보",
    description="""
    체세포수 기반 유방염 분류 기준과 각 등급별 설명을 제공합니다.
    
    **제공 정보:**
    - 각 등급별 체세포수 범위
    - 등급별 설명 및 권장사항
    - 분류 기준의 배경 정보
    - 주의사항 및 참고자료
    """
)
async def get_scc_classification_info():
    """체세포수 분류 기준 및 설명 정보 제공"""
    return await AIPredictionService.get_scc_classification_info()

# 테스트용 엔드포인트 (개발/디버깅용)
from datetime import datetime
@router.post(
    "/mastitis/test-scc-prediction",
    summary="체세포수 예측 테스트",
    description="샘플 데이터로 체세포수 기반 예측을 테스트합니다 (개발용)"
)
async def test_scc_prediction():
    """체세포수 예측 기능 테스트"""
    try:
        # 샘플 테스트 데이터
        test_cases = [
            {"scc": 50, "expected": "정상"},
            {"scc": 150, "expected": "주의"}, 
            {"scc": 400, "expected": "염증 가능성"}
        ]
        
        results = []
        for case in test_cases:
            class TestRequest:
                def __init__(self, scc):
                    self.cow_id = f"test_cow_{scc}"
                    self.somatic_cell_count = scc
                    self.measurement_date = None
                    self.notes = f"테스트 케이스 - {case['expected']}"
            
            test_request = TestRequest(case["scc"])
            result = await AIPredictionService.predict_mastitis_by_scc(test_request)
            
            results.append({
                "input_scc": case["scc"],
                "expected_label": case["expected"],
                "predicted_label": result["prediction_class_label"],
                "predicted_class": result["prediction_class"],
                "test_passed": result["prediction_class_label"] == case["expected"],
                "processing_time_ms": result["processing_time_ms"]
            })
        
        return {
            "test_status": "completed",
            "total_tests": len(test_cases),
            "passed_tests": sum(1 for r in results if r["test_passed"]),
            "failed_tests": sum(1 for r in results if not r["test_passed"]),
            "test_results": results,
            "test_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e),
            "test_timestamp": datetime.now().isoformat()
        }