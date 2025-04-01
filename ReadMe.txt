Uruchomienie lokalnie

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt

gunicorn --bind 0.0.0.0:5000 lab3_2:app



Uruchomienie Docker

docker build -t flask-ml-app .

docker run -p 5000:5000 flask-ml-app


Uruchomienie Docker-compose

docker-compose up --build



Wymagania

Python 3.9+

Redis

Wolne porty 5000 i 6379