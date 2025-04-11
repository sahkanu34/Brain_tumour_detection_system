FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./

# Install system dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app/app.py"]
