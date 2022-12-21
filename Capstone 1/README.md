I am interested in Predictive Maintenance. In this project I've made a Deep Learning model pretending predictive maintenance based on Thermal Images from cameras around the equipment. ‘Pretending” because I didn’t find enough thermal images. So, I took a dataset of weather images with clouds, rain, sunshine, sunrise pictures (4 categories) to fit the model, because they are the most similar to thermal images amid open datasets I could find.
- Here is a link to the dataset: https://data.mendeley.com/datasets/4drtyfjtfy/1
- Note. Output of the model has 5 categories, actually, because the cloud platform made hidden folder called “.ipynb_checkpoints” I didn’t find out how to delete it.

- Building the model:
1.	To train the model I used Convolution layers from Keras model Xception and created new Dense layer base on the dataset.
2.	Then I trained the model with different Learning rates and chose the best one.
3.	Then I used Checkpointing to save the model with the best accuracy on validation dataset.  
4.	I tried to add more layers but didn’t get improvement of the model.
5.	After that I should’ve train the model with bigger image size (it can improve accuracy) but didn’t do that, because lack of free time of the cloud service. 
6.	Tested model on the test dataset.

- Deployment of the model:
1.	Save the Keras model into special format tensorflow “SavedModel” – folder “weather-model”
2.	Ran the model "weather-model" with the prebuilt docker image tensorflow/serving:2.7.0. Running it on Windows you should use the command: docker run -it --rm -p 8500:8500 -v "{FULL PATH}\weather-model:/models/weather-model/1" -e MODEL_NAME="weather-model" tensorflow/serving:2.7.0. Linux: docker run -it --rm -p 8500:8500 -v $(pwd)/clothing-model:/models/clothing-model/1 -e MODEL_NAME="clothing-model" tensorflow/serving:2.7.0
3.	Converte notebook “tf-serving-connect” into a python script “gateway.py” to build flask application
4.	To put everything in pipenv run the command to install libraries: pipenv install grpcio==1.42.0 flask gunicorn keras-image-helper and pipenv install tensorflow-protobuf==2.7.0 protobuf==3.19
5.	Create docker image by the name “image-model.dockerfile”.
6.	To build the image we also need to specify the dockerfile name along with the tag: docker build -t weather-model: w_xception_v1_02 -f image-model.dockerfile
7.	Run the image with: docker run -it --rm -p 8500:8500 weather-model: w_xception_v1_02
8.	For gateway service create docker file image-gateway.dockerfile
9.	Build image: docker build -t weather-model-gateway:001 -f image-gateway.dockerfile . 
10.	Run image: docker run -it --rm -p 9696:9696 weather-gateway:001
11.	To connect the two containers and work simultaneously we need docker compose. It requires YAML file docker-compose.yaml
