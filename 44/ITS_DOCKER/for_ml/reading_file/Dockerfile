FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy CSV and script
COPY breast-cancer.csv .
COPY train_from_csv.py .

# Run training script
CMD ["python", "train_from_csv.py"]
