FROM python:3.8.5
COPY . /app
WORKDIR /app
RUN pip install -r requiements.txt
ENTRYPOINT ["python"]
CMD ["ESRB_Rating_Prediction.py"]