FROM registry.access.redhat.com/ubi9/python-312:9.5

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY main.py .
COPY recommender/ recommender/
COPY utils.py .

ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-u", "main.py"] 
