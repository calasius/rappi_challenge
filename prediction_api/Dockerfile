from python:3.7

RUN mkdir /usr/src/app/
RUN mkdir /usr/models
COPY . /usr/src/app/
WORKDIR /usr/src/app/
EXPOSE 5000
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/usr/src/app/trainer/"
CMD ["python", "prediction_api.py"]