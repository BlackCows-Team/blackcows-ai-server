# services/ai_prediction_service.py

import joblib
import numpy as np
import uuid
import time
from datetime import datetime
from typing import Dict, Any
from fastapi import HTTPException, status
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AIPredictionService:
    
    MODEL_VERSION = "v2.0.0"
    
    # 로컬 모델 경로
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MILK_YIELD_MODEL_PATH = MODELS_DIR / "milk_yield_rf_v2.pkl"
    MILK_YIELD_SCALER_PATH = MODELS_DIR / "milk_yield_scaler_v2.pkl"
    MASTITIS_MODEL_PATH = MODELS_DIR / "mastitis_rf_v1.pkl"
    MASTITIS_SCALER_PATH = MODELS_DIR / "mastitis_scaler_v1.pkl"
    
    # 캐시된 모델 (메모리에 한 번만 로드)
    _milk_yield_model = None
    _milk_yield_scaler = None
    _milk_yield_cache_loaded = False
    _mastitis_model = None
    _mastitis_scaler = None
    _mastitis_cache_loaded = False
    
    @classmethod
    def _load_models(cls):
        """로컬에서 모델 로드 (실패해도 서버는 계속 실행)"""
        if cls._milk_yield_cache_loaded:
            return cls._milk_yield_model, cls._milk_yield_scaler
        
        try:
            logger.info("모델 로드 시작...")
            
            if not cls.MILK_YIELD_MODEL_PATH.exists():
                logger.error(f"모델 파일 없음: {cls.MILK_YIELD_MODEL_PATH}")
                cls._milk_yield_cache_loaded = True
                return None, None
            if not cls.MILK_YIELD_SCALER_PATH.exists():
                logger.error(f"스케일러 파일 없음: {cls.MILK_YIELD_SCALER_PATH}")
                cls._milk_yield_cache_loaded = True
                return None, None
            
            start_time = time.time()
            cls._milk_yield_scaler = joblib.load(cls.MILK_YIELD_SCALER_PATH)
            cls._milk_yield_model = joblib.load(cls.MILK_YIELD_MODEL_PATH)
            cls._milk_yield_cache_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"모델 로드 완료: {load_time:.2f}초")
            
            return cls._milk_yield_model, cls._milk_yield_scaler
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            cls._milk_yield_cache_loaded = True
            return None, None

    @classmethod
    def _load_mastitis_models(cls):
        if cls._mastitis_cache_loaded:
            return cls._mastitis_model, cls._mastitis_scaler
        try:
            cls._mastitis_scaler = joblib.load(cls.MASTITIS_SCALER_PATH)
            cls._mastitis_model = joblib.load(cls.MASTITIS_MODEL_PATH)
            cls._mastitis_cache_loaded = True
            return cls._mastitis_model, cls._mastitis_scaler
        except Exception as e:
            logger.error(f"유방염 모델 로드 실패: {e}")
            cls._mastitis_cache_loaded = True
            return None, None
    
    @classmethod
    def _calculate_confidence(cls, model, scaled_features) -> float:
        """Random Forest 모델의 확신도 계산"""
        try:
            # 각 트리의 예측값 가져오기
            tree_predictions = []
            for estimator in model.estimators_:
                pred = estimator.predict(scaled_features)[0]
                tree_predictions.append(pred)
            
            # 예측값들의 표준편차를 이용한 확신도 계산
            import numpy as np
            std_dev = np.std(tree_predictions)
            mean_pred = np.mean(tree_predictions)
            
            # 변동계수(CV)를 이용한 확신도 (낮을수록 확신도 높음)
            if mean_pred > 0:
                cv = std_dev / mean_pred
                # 확신도를 0~100% 스케일로 변환
                confidence = max(0, min(100, (1 - min(cv, 1)) * 100))
            else:
                confidence = 50.0  # 기본값
            
            return round(confidence, 1)
            
        except Exception as e:
            logger.warning(f"확신도 계산 실패: {e}")
            return 75.0  # 기본 확신도

    @classmethod
    def _calculate_mastitis_confidence(cls, model, scaled_features) -> float:
        """유방염 분류(RandomForestClassifier)에서 확신도: 예측 클래스의 확률(트리 투표 비율) 사용"""
        try:
            # predict_proba는 각 클래스별 확률을 반환
            proba = model.predict_proba(scaled_features)[0]
            confidence = max(proba) * 100  # 가장 높은 클래스 확률을 confidence로
            return round(confidence, 1)
        except Exception as e:
            logger.warning(f"유방염 확신도 계산 실패: {e}")
            return 75.0  # 기본 확신도
    @classmethod
    def _prepare_features(cls, request) -> np.ndarray:
        """8개 특성을 numpy 배열로 변환"""
        features = np.array([
            request.milking_frequency,
            request.conductivity,
            request.temperature,
            request.fat_percentage,
            request.protein_percentage,
            request.concentrate_intake,
            request.milking_month,
            request.milking_day_of_week
        ]).reshape(1, -1)
        
        return features

    @classmethod
    def _prepare_mastitis_features(cls, request) -> np.ndarray:
        features = np.array([
            request.milk_yield,
            request.conductivity,
            request.fat_percentage,
            request.protein_percentage,
            request.lactation_number
        ]).reshape(1, -1)
        return features
    
    @classmethod
    async def predict_milk_yield(cls, request) -> Dict[str, Any]:
        """착유량 예측"""
        start_time = time.time()
        
        try:
            # 모델 로드
            model, scaler = cls._load_models()
            
            # 모델이 로드되지 않은 경우
            if model is None or scaler is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI 모델을 사용할 수 없습니다"
                )
            
            # 특성 준비 및 예측
            features = cls._prepare_features(request)
            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]
            
            # 확신도 계산
            confidence = cls._calculate_confidence(model, scaled_features)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 응답 생성
            response = {
                "prediction_id": str(uuid.uuid4()),
                "cow_id": getattr(request, 'cow_id', None),
                "predicted_milk_yield": round(float(prediction), 2),
                "confidence": confidence,  # 확신도 추가
                "input_features": {
                    "착유횟수": request.milking_frequency,
                    "전도율": request.conductivity,
                    "온도": request.temperature,
                    "유지방비율": request.fat_percentage,
                    "유단백비율": request.protein_percentage,
                    "농후사료섭취량": request.concentrate_intake,
                    "착유기측정월": request.milking_month,
                    "착유기측정요일": request.milking_day_of_week
                },
                "model_version": cls.MODEL_VERSION,
                "prediction_time": datetime.now().isoformat(),
                "processing_time_ms": round(processing_time, 2)
            }
            
            logger.info(f"예측 완료: {prediction:.2f}L ({processing_time:.1f}ms)")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="예측 처리 중 오류가 발생했습니다"
            )

    @classmethod
    async def predict_mastitis(cls, request) -> Dict[str, Any]:
        """유방염 예측"""
        start_time = time.time()
        
        try:
            # 모델 로드
            model, scaler = cls._load_mastitis_models()
            
            # 모델이 로드되지 않은 경우
            if model is None or scaler is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="유방염 AI 모델을 사용할 수 없습니다"
                )
            
            # 특성 준비 및 예측
            features = cls._prepare_mastitis_features(request)
            scaled_features = scaler.transform(features)
            pred_class = int(model.predict(scaled_features)[0])
            
            # 확신도 계산 (분류 전용)
            confidence = cls._calculate_mastitis_confidence(model, scaled_features)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 응답 생성
            LABELS = ["정상", "주의", "염증 가능성 + 유방염 의심"]
            response = {
                "prediction_id": str(uuid.uuid4()),
                "cow_id": getattr(request, 'cow_id', None),
                "prediction_class": pred_class,
                "prediction_class_label": LABELS[pred_class],
                "confidence": confidence,
                "input_features": {
                    "착유량": request.milk_yield,
                    "전도율": request.conductivity,
                    "유지방비율": request.fat_percentage,
                    "유단백비율": request.protein_percentage,
                    "산차수": request.lactation_number
                },
                "model_version": "mastitis_rf_v1",
                "prediction_time": datetime.now().isoformat(),
                "processing_time_ms": round(processing_time, 2)
            }
            
            logger.info(f"유방염 예측 완료: {LABELS[pred_class]} ({processing_time:.1f}ms)")
            return response
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"유방염 예측 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="유방염 예측 처리 중 오류가 발생했습니다"
            )

    @classmethod
    def _categorize_scc(cls, scc_value: float) -> Dict[str, Any]:
        """
        체세포수를 기준으로 유방염 등급을 분류합니다.
        
        Args:
            scc_value: 체세포수 (개/ml)
            
        Returns:
            Dict: 분류 결과 (등급, 라벨, 설명)
        """
        if scc_value <= 100:
            return {
                "class": 0,
                "label": "정상",
                "description": "체세포수가 정상 범위입니다. 건강한 상태로 판단됩니다.",
            }
        elif scc_value <= 300:
            return {
                "class": 1,
                "label": "주의",
                "description": "체세포수가 약간 증가한 상태입니다. 주의 깊은 관찰이 필요합니다.",
            }
        else:  # scc_value > 300
            return {
                "class": 2,
                "label": "염증 가능성 + 유방염 의심",
                "description": "체세포수가 높은 상태로 유방염이 의심됩니다.",
            }
    
    @classmethod
    async def predict_mastitis_by_scc(cls, request) -> Dict[str, Any]:
        """
        체세포수 기반 유방염 예측
        
        Args:
            request: 체세포수 예측 요청 객체
            
        Returns:
            Dict: 예측 결과
        """
        start_time = time.time()
        
        try:
            # 체세포수 값 검증
            if request.somatic_cell_count < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="체세포수는 0 이상의 값이어야 합니다"
                )
            
            # 체세포수 기반 분류
            classification_result = cls._categorize_scc(request.somatic_cell_count)
            
            processing_time = (time.time() - start_time) * 1000
            
            # 신뢰도 계산 (규칙 기반이므로 높은 신뢰도)
            confidence = 95.0  # 체세포수 기반 분류는 확립된 기준이므로 높은 신뢰도
            
            # 응답 생성
            response = {
                "prediction_id": str(uuid.uuid4()),
                "cow_id": getattr(request, 'cow_id', None),
                "prediction_method": "somatic_cell_count",
                "prediction_class": classification_result["class"],
                "prediction_class_label": classification_result["label"],
                "confidence": confidence,
                "description": classification_result["description"],
                "recommendation": classification_result["recommendation"],
                "input_features": {
                    "체세포수": request.somatic_cell_count,
                    "단위": "개/ml"
                },
                "classification_criteria": {
                    "정상": "≤ 100개/ml",
                    "주의": "101-300개/ml", 
                    "염증_가능성": "> 300개/ml"
                },
                "prediction_time": datetime.now().isoformat(),
                "processing_time_ms": round(processing_time, 2)
            }
            
            logger.info(f"체세포수 기반 유방염 예측 완료: {request.somatic_cell_count}개/ml -> {classification_result['label']} ({processing_time:.1f}ms)")
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"체세포수 기반 예측 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="체세포수 기반 예측 처리 중 오류가 발생했습니다"
            )
            
    @classmethod
    async def predict_milk_yield_batch(cls, batch_request) -> Dict[str, Any]:
        """다중 젖소 착유량 일괄 예측"""
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            predictions = []
            successful = 0
            failed = 0
            
            for request in batch_request.predictions:
                try:
                    prediction = await cls.predict_milk_yield(request)
                    predictions.append(prediction)
                    successful += 1
                except Exception as e:
                    logger.error(f"배치 예측 개별 실패: {str(e)}")
                    failed += 1
            
            total_time = time.time() - start_time
            
            return {
                "batch_id": batch_id,
                "total_predictions": len(batch_request.predictions),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "predictions": predictions,
                "batch_created_at": datetime.now().isoformat(),
                "total_processing_time_ms": round(total_time * 1000, 2)
            }
                
        except Exception as e:
            logger.error(f"배치 예측 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="배치 예측 처리 중 오류가 발생했습니다"
            )

    @classmethod
    async def predict_mastitis_batch(cls, batch_request) -> Dict[str, Any]:
        """
        다중 젖소 유방염 일괄 예측
        Args:
            batch_request: MastitisBatchRequest 객체
        Returns:
            dict: 배치 예측 결과 (성공/실패/예측 리스트/처리시간 등)
        """
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        try:
            predictions = []
            successful = 0
            failed = 0
            for request in batch_request.predictions:
                try:
                    prediction = await cls.predict_mastitis(request)
                    predictions.append(prediction)
                    successful += 1
                except Exception as e:
                    logger.error(f"배치 유방염 예측 실패: {str(e)}")
                    failed += 1
            total_time = time.time() - start_time
            return {
                "batch_id": batch_id,
                "total_predictions": len(batch_request.predictions),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "predictions": predictions,
                "batch_created_at": datetime.now().isoformat(),
                "total_processing_time_ms": round(total_time * 1000, 2)
            }
        except Exception as e:
            logger.error(f"배치 유방염 예측 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="배치 유방염 예측 처리 중 오류가 발생했습니다"
            )
    
    
    @classmethod
    async def predict_mastitis_scc_batch(cls, batch_request) -> Dict[str, Any]:
        """
        다중 젖소 체세포수 기반 유방염 일괄 예측
        
        Args:
            batch_request: 배치 예측 요청 객체
            
        Returns:
            Dict: 배치 예측 결과
        """
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            predictions = []
            successful = 0
            failed = 0
            
            for request in batch_request.predictions:
                try:
                    prediction = await cls.predict_mastitis_by_scc(request)
                    predictions.append(prediction)
                    successful += 1
                except Exception as e:
                    logger.error(f"배치 체세포수 예측 개별 실패: {str(e)}")
                    failed += 1
                    # 실패한 항목도 결과에 포함 (에러 정보와 함께)
                    error_result = {
                        "prediction_id": str(uuid.uuid4()),
                        "cow_id": getattr(request, 'cow_id', None),
                        "error": True,
                        "error_message": str(e),
                        "input_features": {
                            "체세포수": getattr(request, 'somatic_cell_count', None)
                        }
                    }
                    predictions.append(error_result)
            
            total_time = time.time() - start_time
            
            return {
                "batch_id": batch_id,
                "prediction_method": "somatic_cell_count_batch",
                "total_predictions": len(batch_request.predictions),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "predictions": predictions,
                "batch_created_at": datetime.now().isoformat(),
                "total_processing_time_ms": round(total_time * 1000, 2),
                "average_processing_time_ms": round((total_time * 1000) / len(batch_request.predictions), 2) if batch_request.predictions else 0
            }
                
        except Exception as e:
            logger.error(f"배치 체세포수 예측 실패: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="배치 체세포수 예측 처리 중 오류가 발생했습니다"
            )
   
    @classmethod
    async def get_scc_classification_info(cls) -> Dict[str, Any]:
        """
        체세포수 분류 기준 정보 제공
        
        Returns:
            Dict: 분류 기준 및 설명
        """
        return {
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
    
    @classmethod
    async def check_model_health(cls) -> Dict[str, Any]:
        """모델 상태 확인"""
        try:
            start_time = time.time()
            
            # 파일 존재 확인
            model_exists = cls.MILK_YIELD_MODEL_PATH.exists()
            scaler_exists = cls.MILK_YIELD_SCALER_PATH.exists()
            
            # 모델 로드 및 테스트
            model_load_success = False
            prediction_test_success = False
            
            try:
                model, scaler = cls._load_models()
                if model is not None and scaler is not None:
                    model_load_success = True
                
                    # 테스트 예측
                    test_features = np.array([[2, 7.5, 38.5, 3.8, 3.2, 3.5, 6, 1]])
                    test_scaled = scaler.transform(test_features)
                    test_prediction = model.predict(test_scaled)[0]
                    prediction_test_success = True
                
            except Exception as e:
                logger.error(f"모델 테스트 실패: {e}")
            
            response_time = (time.time() - start_time) * 1000
            
            # 상태 결정
            if model_exists and scaler_exists and model_load_success and prediction_test_success:
                status = "healthy"
                message = "AI 예측 서비스가 정상 작동 중입니다"
            elif model_exists and scaler_exists:
                status = "degraded" 
                message = "모델 파일은 존재하지만 로드 또는 테스트에 실패했습니다"
            else:
                status = "unavailable"
                message = "AI 예측 서비스를 사용할 수 없습니다"
            
            return {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": round(response_time, 2),
                "checks": {
                    "model_file_exists": model_exists,
                    "scaler_file_exists": scaler_exists,
                    "model_load_success": model_load_success,
                    "prediction_test_success": prediction_test_success,
                    "cache_loaded": cls._milk_yield_cache_loaded
                },
                "model_info": {
                    "version": cls.MODEL_VERSION,
                    "cached": cls._milk_yield_cache_loaded,
                    "available": model_load_success
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": "모델 상태 확인 중 오류가 발생했습니다",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    
    @classmethod
    async def test_prediction_with_sample(cls) -> Dict[str, Any]:
        """샘플 테스트"""
        try:
            # 샘플 데이터 클래스
            class SampleRequest:
                def __init__(self):
                    self.cow_id = "test_cow"
                    self.milking_frequency = 2
                    self.conductivity = 7.7
                    self.temperature = 38.5
                    self.fat_percentage = 3.8
                    self.protein_percentage = 3.2
                    self.concentrate_intake = 3.5
                    self.milking_month = 6
                    self.milking_day_of_week = 1
            
            sample_request = SampleRequest()
            
            result = await cls.predict_milk_yield(sample_request)
            
            return {
                "test_status": "success",
                "sample_input": {
                    "착유횟수": 2,
                    "전도율": 7.7,
                    "온도": 38.5,
                    "유지방비율": 3.8,
                    "유단백비율": 3.2,
                    "농후사료섭취량": 3.5,
                    "착유기측정월": 6,
                    "착유기측정요일": 1
                },
                "predicted_milk_yield": result["predicted_milk_yield"],
                "confidence": result["confidence"],
                "processing_time_ms": result["processing_time_ms"],
                "test_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "test_status": "failed",
                "error": str(e),
                "test_timestamp": datetime.now().isoformat()
            }

# 애플리케이션 시작시 모델 로드
def initialize_models():
    """서버 시작시 모델 초기화"""
    try:
        AIPredictionService._load_models()
        logger.info("AI 서비스 초기화 완료")
        return True
    except Exception as e:
        logger.warning(f"AI 서비스 초기화 실패: {e}")
        logger.info("다른 기능들은 정상 작동합니다")
        return False