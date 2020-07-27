from python:3.7
WORKDIR /trainer
#VOLUME ["/trainer"]
RUN apt-get update && apt-get install -y python3-tk
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY . ./

ENTRYPOINT ["python", "./trainer.py"]