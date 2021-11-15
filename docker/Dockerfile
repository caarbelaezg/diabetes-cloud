# syntax=docker/dockerfile:1

# Base OS
FROM python:3.8-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 3000
# docker run -p 8000:5000 iris-model
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]