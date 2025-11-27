FROM python:3.10

WORKDIR /app

COPY requirements1.txt .
RUN pip install --no-cache-dir -r requirements1.txt

COPY . .

RUN python - <<EOF
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
EOF

CMD ["python", "main.py"]