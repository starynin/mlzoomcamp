I am interested in Predictive Maintenance. I think it is an important problem. 
And ability to solve such problems can save a lot of money for businesses around the World and also significantly decrease a number of accidents.
Machine Learning can be used in the solutions to solving such problems, because usually there are a lot of data captured from many sensors. 
In this project I used open dataset from "University of California". 
Here is a link to download it: https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
There is the same file in the "data" folder.
Steps made to train and deploy the ML model:
1. EDA
2. Train, validation, test split
3. Training and evaluation of Logistic Regression model
4. Training and evaluation of Desision Tree model
5. Training and evaluation of Random Forest model
6. Training and evaluation of XGBoost model
7. The best model was chosen based on AUC
8. This model was tested on the test dataset
9. Made file train.py
10. Made file predict.py
11. Made file project1_model_FR.bin
12. Made web-service using flask
13. Made virtual enviroment using pipenv
14. Made Docker image with Dockerfile
15. Run Docker Container

To run the web-service in the Docker container you have to start Docker.
After coppping Dockerfile you have to build a Docker image by executing command:
"docker build -t predictive ."
Then you have to run command to run it:
"docker run -it -p 9696:9696 predictive:latest"
After that you can execute command:
"python train.py" to train the model
and use file "Test Web-Service.ipynb" to predict the failure of a particular machine (tool) by providing it's features in a json format:

test_json = {"type": 'L',
 "air_temperature": 222.7,
 "process_temperature": 391.1,
 "rotational_speed": 16030,
 "torque": 380.5,
 "tool_wear": 1700}
