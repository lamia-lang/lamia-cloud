FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY project/ ./project/

CMD ["sh", "-c", "cd /app/project && lamia ${LAMIA_SCRIPT} ${LAMIA_EXTRA_ARGS:-}"]
