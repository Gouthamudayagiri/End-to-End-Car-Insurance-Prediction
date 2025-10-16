FROM python:3.9-slim

# WORKDIR /app

# # Copy requirements first for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy source code
# COPY src/ ./src/
# COPY app.py .
# COPY config/ ./config/
# COPY static/ ./static/
# COPY templates/ ./templates/

# # Install package in development mode
# RUN pip install -e .

# EXPOSE 8080

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]