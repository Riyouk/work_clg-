FROM python:3.11-slim

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train_model.py .

# Run the script when the container starts
CMD ["python", "train_model.py"]