# main.py
"""
BlackCows AI 예측 서버 - 메인 애플리케이션

이 서버는 젖소 착유량 예측 AI 모델을 제공하는 경량화된 FastAPI 서버입니다.
- 오픈 액세스 AI 모델 추론 서버
- 인증/DB 없는 순수 모델 추론 서버
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

# 프로젝트 내부 모듈
from routers import ai_prediction
from services.ai_prediction_service import initialize_models

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="BlackCows AI 예측 서버",
    version="1.0.0",
    description="""
    젖소 착유량 예측을 위한 AI 모델 서버
    
    **주요 기능:**
    - 개별 젖소 착유량 예측
    - 다중 젖소 배치 예측
    - 모델 상태 확인 및 헬스체크
    """,
)

# CORS 미들웨어 설정 - 모든 origin 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# AI 예측 라우터 등록
app.include_router(ai_prediction.router, tags=["AI 예측"])

# 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작시 AI 모델 초기화"""
    logger.info("🚀 BlackCows AI 서버 시작 중...")
    
    # 모델 초기화
    model_loaded = initialize_models()
    
    if model_loaded:
        logger.info("✅ AI 모델 초기화 완료")
    else:
        logger.warning("⚠️ AI 모델 초기화 실패 - 기본 기능만 제공")
    
    logger.info("🎉 서버 준비 완료")

# 서버 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료시 정리 작업"""
    logger.info("🛑 BlackCows AI 서버 종료 중...")

# 루트 엔드포인트
@app.get("/", tags=["시스템"])
async def root():
    """서버 기본 정보 반환"""
    return {
        "service": "알아서 뭐하게",
        "version": "1.0.0",
        "status": "running"
    }
    
# 헬스체크 엔드포인트
@app.get("/health", tags=["시스템"])
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "service": "BlackCows AI Server",
        "version": "1.0.0"
    }

# 전역 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리기"""
    logger.error(f"예상치 못한 오류: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "내부 서버 오류",
            "message": "예상치 못한 오류가 발생했습니다",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # 개발 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )