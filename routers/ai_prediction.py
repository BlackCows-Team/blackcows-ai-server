# routers/ai_prediction.py

from fastapi import APIRouter, HTTPException, status
from schemas.ai_prediction import (
    MilkYieldPredictionRequest,
    PredictionBatchRequest
)
from services.ai_prediction_service import AIPredictionService

router = APIRouter(prefix="/ai", tags=["AI 예측"])

@router.post(
    "/milk-yield/predict",
    summary="착유량 예측",
    description="""
    젖소의 8개 특성(feature)을 바탕으로 착유량을 예측합니다.

    **필요한 입력:**
    - 착유횟수, 전도율, 온도, 유지방비율, 유단백비율
    - 농후사료섭취량, 착유기측정월, 착유기측정요일

    **결과:**
    - 예상 착유량 (리터)
    """
)
async def predict_milk_yield(
    prediction_request: MilkYieldPredictionRequest
):
    """개별 젖소의 착유량을 예측합니다."""
    return await AIPredictionService.predict_milk_yield(prediction_request)


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