FROM agrigorev/zoomcamp-model:3.8.12-slim

RUN pip install pipenv
RUN pip install flask gunicorn q

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]

EXPOSE 8006

# docker build -t zoomcamp-test .
# docker run -it --rm -p 8006:8006 zoomcamp-test
# gunicorn --bind 0.0.0.0:8006 predict:app

ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:8006", "predict:app"]
