FROM python:3.12-slim

WORKDIR /app

COPY . /app/

EXPOSE 8501

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]


