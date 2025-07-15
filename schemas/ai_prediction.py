# schemas/ai_prediction.py

from pydantic import BaseModel, validator, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# 예측 요청 스키마
class MilkYieldPredictionRequest(BaseModel):
    cow_id: Optional[str] = Field(None, description="젖소 ID (선택사항)")
    
    # 필수 예측 변수들 (모델 학습에 사용된 8개 특성)
    milking_frequency: int = Field(..., description="착유횟수", ge=1)
    conductivity: float = Field(..., description="전도율", ge=0)
    temperature: float = Field(..., description="온도 (°C)", ge=-10, le=80)
    fat_percentage: float = Field(..., description="유지방비율 (%)", ge=0, le=100)
    protein_percentage: float = Field(..., description="유단백비율 (%)", ge=0, le=100)
    concentrate_intake: float = Field(..., description="농후사료섭취량 (kg)", ge=0)
    milking_month: int = Field(..., description="착유기측정월 (1-12)", ge=1, le=12)
    milking_day_of_week: int = Field(..., description="착유기측정요일 (0:월요일~6:일요일)", ge=0, le=6)
    
    # 추가 메타데이터
    prediction_date: Optional[str] = Field(None, description="예측 기준일 (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, description="예측 관련 메모")
    
    @validator('prediction_date')
    def validate_prediction_date(cls, v):
        if v is not None and len(v.strip()) > 0:
            try:
                datetime.strptime(v.strip(), '%Y-%m-%d')
                return v.strip()
            except ValueError:
                raise ValueError('예측 기준일은 YYYY-MM-DD 형식으로 입력해주세요')
        return v

# 배치 예측 요청 스키마
class PredictionBatchRequest(BaseModel):
    predictions: List[MilkYieldPredictionRequest] = Field(..., description="예측 요청 목록")
    batch_name: Optional[str] = Field(None, description="배치 이름")

# 에러 응답 스키마
class PredictionErrorResponse(BaseModel):
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    error_details: Optional[Dict[str, Any]] = Field(None, description="에러 상세 정보")
    suggestion: Optional[str] = Field(None, description="해결 방법 제안")

# 유방염 예측 요청 스키마
class MastitisPredictionRequest(BaseModel):
    cow_id: Optional[str] = Field(None, description="젖소 ID (선택사항)")
    milk_yield: float = Field(..., description="착유량 (리터)", ge=0)
    conductivity: float = Field(..., description="전도율", ge=0)
    fat_percentage: float = Field(..., description="유지방비율", ge=0, le=100)
    protein_percentage: float = Field(..., description="유단백비율", ge=0, le=100)
    lactation_number: int = Field(..., description="산차수", ge=1)
    prediction_date: Optional[str] = Field(None, description="예측 기준일 (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, description="예측 관련 메모")
    @validator('prediction_date')
    def validate_prediction_date(cls, v):
        if v is not None and len(v.strip()) > 0:
            try:
                datetime.strptime(v.strip(), '%Y-%m-%d')
                return v.strip()
            except ValueError:
                raise ValueError('예측 기준일은 YYYY-MM-DD 형식으로 입력해주세요')
        return v

# 유방염 배치 예측 요청 스키마
class MastitisBatchRequest(BaseModel):
    predictions: List[MastitisPredictionRequest] = Field(..., description="유방염 예측 요청 목록")
    batch_name: Optional[str] = Field(None, description="배치 이름")
    

# 체세포수 기반 유방염 예측 요청 스키마
class SomaticCellCountPredictionRequest(BaseModel):
    cow_id: Optional[str] = Field(None, description="젖소 ID (선택사항)")
    somatic_cell_count: float = Field(..., description="체세포수 (개/ml)", ge=0)
    measurement_date: Optional[str] = Field(None, description="측정일 (YYYY-MM-DD)")
    notes: Optional[str] = Field(None, description="예측 관련 메모")
    
    @validator('measurement_date')
    def validate_measurement_date(cls, v):
        if v is not None and len(v.strip()) > 0:
            try:
                datetime.strptime(v.strip(), '%Y-%m-%d')
                return v.strip()
            except ValueError:
                raise ValueError('측정일은 YYYY-MM-DD 형식으로 입력해주세요')
        return v
    
    @validator('somatic_cell_count')
    def validate_scc(cls, v):
        if v < 0:
            raise ValueError('체세포수는 0 이상의 값이어야 합니다')
        if v > 10000:  # 일반적으로 매우 높은 값에 대한 경고
            logger.warning(f"매우 높은 체세포수 값: {v}개/ml")
        return v

# 체세포수 기반 배치 예측 요청 스키마
class SomaticCellCountBatchRequest(BaseModel):
    predictions: List[SomaticCellCountPredictionRequest] = Field(..., description="체세포수 예측 요청 목록")
    batch_name: Optional[str] = Field(None, description="배치 이름")
    
    @validator('predictions')
    def validate_predictions_count(cls, v):
        if len(v) == 0:
            raise ValueError('최소 1개 이상의 예측 요청이 필요합니다')
        if len(v) > 1000:  # 배치 처리 한계 설정
            raise ValueError('한 번에 처리할 수 있는 최대 요청 개수는 1000개입니다')
        return v
