FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY WA_Fn-UseC_-HR-Employee-Attrition.csv .
COPY main.py .
CMD [ "python" , "main.py" ]