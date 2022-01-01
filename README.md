<<<<<<< HEAD
# Predicting the Number of High PM2.5 Levels for Cities in China

### Project Background: PM2.5 stands for particular matter; tiny particles in the air usually with diameters around 2.5 micrometers or smaller. They can affect the respiratory and cardiovascular systems, and is associated with multiple diseases such as lung cancer and chronic bronchitis. It also can worsen the symptoms of patients with respiratory allergies. Sources of PM2.5 are: vehicles exhausts, burning fuels operations, power plants, and natural causes such as forest fires and volcanos. 

### Project Goal: Five datasets for five cities in China are given: Beijing, Chengdu, Guangzhou, Shanghai, Shenyang. In each dataset there is the hourly PM levels reported by $3$ or $4$ stations, and other meteorological measurements such as temperature, pressure, etc. The goal is to predict the number of days that are high for PM2.5 levels averaged over the course of each calendar month for each city from the datasets that we currently have.

### Methods: Dataset contains five datasets for five cities in China. Outliers were removed. Feature engineeing was performed by averaging the hours and different stations to a daily value. Train set was chosen from $2010$ till $2014$, and test set is $2015$. A PM level of 55 or larger is considered large in this analysis. The label was created by counting the number of days that have a PM level equal or greater than 55 for the next 30 days. The prediction is generated from multiple features at one timestamp.

### Results: Catboost model has the best results overall in terms of RMSE and MAE metrics. City, temperature, dew point, month, and PM_avg were the most important features for Catboost model to make its predictions.

### How to Use:
Build a docker image from Dockerfile with the command: "docker build -t [image_name] .". Then create a container from the docker image created from the previous step by running the command: "docker run -it --name [container_name] [image_name]". Finally, open jupyter notebook in a browser and bypass token authentication by running the command: "jupyter notebook --NotebookApp.token='' --NotebookApp.password='' --no-browser --allow-root".

To filter the data and check eda, run eda.ipynb file. To train or load models, run models.ipynb file.


For bugs and questions, contact: saadi.cv4 at gmail.com
=======
# PM_Levels_Predictions
Predicting the Number of High PM2.5 Levels for Cities in China
>>>>>>> d4f79e228fdc5a8cdb62b8de95c22d308da0bcf1
