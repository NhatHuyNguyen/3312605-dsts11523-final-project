FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code (ignore models, and outputs via .dockerignore)
COPY src/ ./src/
COPY models/ ./models/

# Default command (can be overridden by docker run)
# CMD ["python", "src/train.py"]