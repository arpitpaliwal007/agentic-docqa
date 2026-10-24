FROM python:3.12-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY app ./app
COPY run.py .
EXPOSE 8000
CMD ["python", "run.py"]
