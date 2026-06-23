FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# 데이터, 아웃풋 폴더 생성
RUN mkdir -p data output mlruns

# 포트 설정 (Streamlit)
EXPOSE 8501

# 실행 명령
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
