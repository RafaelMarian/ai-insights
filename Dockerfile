# Imaginea de bază - Python 3.10
FROM python:3.10-slim

# Setăm directorul de lucru
WORKDIR /app

# Copiem codul aplicației
COPY app/ app/
COPY model/ model/

# Instalăm dependențele
RUN pip install --no-cache-dir -r app/requirements.txt

# Expunem portul Flask
EXPOSE 5000

# Comanda de pornire
CMD ["python", "app/main.py"]
