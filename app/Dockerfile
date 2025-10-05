# Imaginea de bază
FROM python:3.10-slim

WORKDIR /app

# Copiem doar fișierele pentru instalarea dependențelor
COPY requirements.txt .

# Instalăm dependențele
RUN pip install --no-cache-dir -r requirements.txt

# Copiem restul aplicației
COPY . .

EXPOSE 5000

CMD ["python", "main.py"]
