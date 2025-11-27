FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
COPY practise4.py . 
COPY regression_house_prices_anomalous.csv .
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python","practise4.py" ]