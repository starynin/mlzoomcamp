FROM tensorflow/serving:2.7.0

COPY clothing-model /models/weather-model/1
ENV MODEL_NAME="weather-model"