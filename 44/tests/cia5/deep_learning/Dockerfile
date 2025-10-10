FROM python:3.10-slim

WORKDIR /app

COPY mnist_model.py .

RUN pip install --no-cache-dir tensorflow==2.16.1 numpy pandas matplotlib seaborn scikit-learn

CMD ["python", "mnist_model.py"]