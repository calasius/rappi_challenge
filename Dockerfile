from python:3.7

RUN mkdir /usr/src/app/
RUN mkdir /usr/models
COPY ./prediction_api/ /usr/src/app/
COPY ./trainer/ /usr/scr/app/
WORKDIR /usr/src/app/
EXPOSE 5000
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/usr/src/app/trainer/"
CMD ["python", "prediction_api.py"]