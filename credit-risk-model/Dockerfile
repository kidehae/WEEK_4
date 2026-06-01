# Use an explicit, lightweight Python base runtime environment
FROM python:3.12-slim

# Set the active working directory inside the container space
WORKDIR /app

# Copy requirement files first to maximize Docker layer cache efficiency
COPY requirements.txt .

# Install all locked backend dependency wheels smoothly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining codebase modules into the container
COPY . .

# Expose port 8000 for network routing traffic
EXPOSE 8000

# Command to spin up the web service on container initialization
# Replace your old CMD line with this exact syntax:
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]