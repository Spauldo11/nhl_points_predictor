FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /code

# Install dependencies first for layer caching
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all repository files into container
COPY . /code

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start command running server.py
CMD ["python", "server.py"]
