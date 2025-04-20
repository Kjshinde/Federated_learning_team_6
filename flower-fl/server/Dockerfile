# Optional Dockerfile for server
# Optional: containerize your FL server
FROM python:3.8-slim

WORKDIR /app

# Install Flower
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

CMD ["python", "server.py"]
