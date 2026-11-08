FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code (ignore models, and outputs via .dockerignore)
COPY . .

# Expose port 8501 for the Streamlit application
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]