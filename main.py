# main.py
"""
BlackCows AI 예측 서버 - 메인 애플리케이션

이 서버는 젖소 착유량 예측 AI 모델을 제공하는 경량화된 FastAPI 서버입니다.
- 오직 AWS EC2 메인 서버에서만 접근 가능
- Cloudflare Tunnel을 통해 ai.blackcowsdairy.com 도메인으로 노출
- 인증/DB 없는 순수 모델 추론 서버
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status, Header
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

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
    
    **보안:**
    - AWS EC2 메인 서버 전용
    - Cloudflare Tunnel 통해서만 접근
    """,
    docs_url=None,    # 기본 Swagger UI 비활성화
    redoc_url=None    # 기본 ReDoc 비활성화
)

security = HTTPBasic()

# ai 서버 접근 비밀번호
TEAM_PASSWORD = "blackcows_bms!"

def verify_team_access(request: Request):
    host = request.headers.get("host", "")
    user_agent = request.headers.get("user-agent", "")
    
    # 디버깅용 로그 추가
    logger.info(f"🔍 Request Debug:")
    logger.info(f"  Host: {host}")
    logger.info(f"  User-Agent: {user_agent}")
    logger.info(f"  All Headers: {dict(request.headers)}")
    
    # 1. EC2 서버에서 오는 요청 감지
    if ("api.blackcowsdairy.com" in host or           # 호스트 헤더
        "python-requests" in user_agent.lower() or    # requests 라이브러리
        "curl" in user_agent.lower()):                # curl 명령어
        return "EC2_SERVER"  # 인증 면제
    
    # 2. 브라우저에서 ai.blackcowsdairy.com 접근 - 인증 필요
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="꺼져.",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    # Basic Auth 검증
    import base64
    try:
        encoded_credentials = auth_header.split(" ")[1]
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
        username, password = decoded_credentials.split(":", 1)
        
        if password != TEAM_PASSWORD:
            raise HTTPException(status_code=401, detail="꺼져라. FUCK YOU.")
            
    except Exception:
        raise HTTPException(status_code=401, detail="FUCK YOU")
    
    return "BROWSER_USER"

# CORS 미들웨어 설정 - EC2 서버만 허용
EC2_ALLOWED_ORIGINS = [
    "https://api.blackcowsdairy.com",  # 메인 EC2 서버
    "http://api.blackcowsdairy.com",   # HTTP도 허용 (필요시)
    "http://localhost:8000",           # 로컬 테스트용
    "http://127.0.0.1:8000",            # 로컬 테스트용
    "https://ai.blackcowsdairy.com",   # 터널 도메인 (인증으로 보호됨)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=EC2_ALLOWED_ORIGINS,
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
async def root(user_type: str = Depends(verify_team_access)):
    """서버 기본 정보 반환"""
    return {
        "service": "알아서 뭐하게",
        "version": "1.0.0",
        "status": "running",
        "accessed_by": user_type
    }
    
# 헬스체크 엔드포인트
@app.get("/health", tags=["시스템"])
async def health_check(user_type: str = Depends(verify_team_access)):
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "service": "BlackCows AI Server 하이요",
        "version": "1.0.0",
        "accessed_by": user_type
    }

@app.get("/docs")
async def custom_docs(user_type: str = Depends(verify_team_access)):
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="BlackCows API Docs",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png"
    )

@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi(user_type: str = Depends(verify_team_access)):
    return get_openapi(
        title="BlackCows AI Server",
        version="1.0.0",
        routes=app.routes,
    )

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