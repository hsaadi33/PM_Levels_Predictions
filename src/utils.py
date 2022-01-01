import os
import pandas as pd
import numpy as np
from numpy.random import seed
import pickle
import dill
from itertools import product
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from catboost import CatBoostRegressor


def averaging(city_dict_avg):
    """ Average over 24 hours, then average over stations, then create a column with a datetime object 
    Args:
        city_dict_avg: A dictionary that contains a dataframe of the dataset for each city. (dict)
    Returns:
        city_dict_avg: A dictionary that contains a dataframe of the dataset for each city after averaging. (dict)
    """
    # Average over 24 hours
    for city,df in city_dict_avg.items():
        city_dict_avg[city] = city_dict_avg[city].groupby(['year','month','day']).mean()
    
    # Average PM stations
    for city,df in city_dict_avg.items():
        stations_cols = [col for col in df if col.startswith('PM')]
        city_dict_avg[city]['PM_avg'] = city_dict_avg[city][stations_cols].mean(axis=1,skipna=True)
        # Remove the row number, stations and hours columns since we averaged over stations and hours
        city_dict_avg[city] = city_dict_avg[city].drop(stations_cols + ['No','hour'], axis=1)
        city_dict_avg[city] = city_dict_avg[city].reset_index()
        
    # Create a column with a date object consisting of year, month, and day
    for city, df in city_dict_avg.items():
        city_dict_avg[city]['date'] = pd.to_datetime(city_dict_avg[city][['year', 'month', 'day']])
        
    return city_dict_avg

def plot_city_features_histograms(city_dict, city_dict_no_na, relevant_features):
    """ Plot histograms for each city for all relevant features 
    Args:
        city_dict: A dictionary that contains a dataframe of the original dataset for each city. (dict)
        city_dict_no_na: A dictionary that contains a dataframe of the filtered dataset for each city. (dict)
        relevant_features: A list of features to be plotted. (str list)
    """
    for city,df in city_dict.items():
        fig, ax = plt.subplots(2,12,figsize=(25, 10))
        ax = ax.flatten()
        count_col1 = 0
        count_col2 = 0
        for feature_index, feature in enumerate(relevant_features):
            if feature_index > 3:
                ax[feature_index] = plt.subplot2grid((2, 12), (1, 4*count_col1),colspan=4)
                count_col1 += 1
            else:
                ax[feature_index] = plt.subplot2grid((2, 12), (0, 3*count_col2),colspan=3)
                count_col2 += 1

            ax[feature_index].hist(df[feature], bins=15, alpha=0.5, label=feature)
            ax[feature_index].hist(city_dict_no_na[city][feature], bins=15, alpha=0.5, label=feature+' after filtering')
            ax[feature_index].set_xlabel(feature)
            ax[feature_index].legend()

        plt.suptitle(city)
        plt.tight_layout()
        plt.show()

def onehot_encode(df_onehot,col_name):
    """ Onehot encodes a column in for a dataframe
    Args:
        df_onehot:  A dataframe with a column to be onehot encoded. (dataframe)
        col_name: Column name to be onehot encoded. (str)
    Returns:
        df_onehot:  A dataframe after onehot encoding a column. (dataframe)
    """
    one_hot = pd.get_dummies(df_onehot[col_name])
    df_onehot = df_onehot.join(one_hot)
        
    return df_onehot

def display_results(scores,model):
    """ Prints Relevant metrics for a model 
    Args:
        scores: Scores of model.
        model: Name of the model. (str)
    """
    print('Cross Validation MSE for ' + model + '\n')
    print('Scores: ', scores)
    print('Average MSE: ', scores.mean())
    print('Average RMSE: ', np.sqrt(scores.mean()))
        

def ridge(ridge_params,directory_inter,ridge_model_filename,ridge_scores_filename,X_train,y_train):
    """ Ridge regression model with hyperparameter tuning and saving the model and scores
    Args:
        ridge_params: Ridge regression hyperparameters. (dict)
        directory_inter: Directory to save the model. (str)
        ridge_model_filename: Name of the filename for ridge regression model to be saved. (str)
        ridge_scores_filename: Name of the filename for ridge regression scores to be saved. (str)
        X_train: A dataframe containing features to train on. (dataframe)
        y_test: A dataframe containing the label to train on. (dataframe)
    Returns:
        ridge_reg: Ridge regression model after training to the best hyperparameters.
        -ridge_reg_scores: Scores of ridge regression.
    """
    
    # If the model ans scores are already calculated, then just load them
    if os.path.exists(os.path.join(directory_inter,ridge_model_filename)) and os.path.exists(os.path.join(directory_inter,ridge_scores_filename)):      
        ridge_reg = pd.read_pickle(os.path.join(directory_inter,ridge_model_filename))
        ridge_reg_scores = pd.read_pickle(os.path.join(directory_inter,ridge_scores_filename))       
        return ridge_reg, -ridge_reg_scores
    
    ridge_reg = Ridge()
    grid_search = GridSearchCV(estimator = ridge_reg,
                               param_grid=ridge_params,
                               scoring = 'neg_mean_squared_error',
                               cv=5,
                               n_jobs=-1,
                               pre_dispatch = '2*n_jobs',
                               return_train_score = True) 
    
    grid_search.fit(X_train,y_train)
    ridge_reg = Ridge(**grid_search.best_params_)
    ridge_reg_scores = cross_val_score(ridge_reg, X_train,y_train, scoring="neg_mean_squared_error", cv=5)
    ridge_reg.fit(X_train,y_train)
    
    # Save (pickle) the ridge regression model and scores
    pickle.dump(ridge_reg, open(os.path.join(directory_inter,ridge_model_filename), 'wb'))   
    pickle.dump(ridge_reg_scores, open(os.path.join(directory_inter,ridge_scores_filename), 'wb'))   
    
    return ridge_reg, -ridge_reg_scores


def neural(nn_params,directory_inter,nn_scores_filename,nn_model_filename,X_train,y_train):
    """ Neural network regression model with hyperparameter tuning and saving the model and scores
    Args:
        nn_params: Neural network regression hyperparameters. (dict)
        directory_inter: Directory to save the model. (str)
        nn_model_filename: Name of the filename for neural network regression model to be saved. (str)
        nn_scores_filename: Name of the filename for neural network regression scores to be saved. (str)
        X_train: A dataframe containing features to train on. (dataframe)
        y_test: A dataframe containing the label to train on. (dataframe)
    Returns:
        nn_reg: Neural network regression model after training to the best hyperparameters.
        -nn_reg_scores: Scores of neural network regression.
    """
    
    def nn_train(hidden_sizes,dropout_rate,batch,random_state):
        seed(random_state)
        tensorflow.random.set_seed(random_state)
        nn_reg = Sequential()
        nn_reg.add(Dense(hidden_sizes[0], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
        nn_reg.add(Dropout(dropout_rate))
        nn_reg.add(Dense(hidden_sizes[1], activation='relu'))
        nn_reg.add(Dropout(dropout_rate))
        nn_reg.add(Dense(1, activation='linear'))
        nn_reg.summary()
        nn_reg.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

        return nn_reg
    
    # If the model ans scores are already calculated, then just load them
    if os.path.exists(os.path.join(directory_inter,nn_model_filename)) and os.path.exists(os.path.join(directory_inter,nn_scores_filename)):
        nn_reg = KerasRegressor(build_fn=nn_train)
        nn_reg.model = load_model(os.path.join(directory_inter,nn_model_filename))
        nn_reg_scores = pd.read_pickle(os.path.join(directory_inter,nn_scores_filename))
        return nn_reg, -nn_reg_scores
    
    nn_reg = KerasRegressor(build_fn=nn_train)
    grid_search = GridSearchCV(estimator = nn_reg,
                               param_grid=nn_params,
                               scoring = 'neg_mean_squared_error',
                               cv=5,
                               n_jobs=-1,
                               pre_dispatch = '2*n_jobs',
                               return_train_score = True) 
    
    grid_search.fit(X_train,y_train)
    nn_reg = KerasRegressor(build_fn= lambda: nn_train(**grid_search.best_params_))
    nn_reg_scores = cross_val_score(nn_reg, X_train,y_train, scoring="neg_mean_squared_error", cv=5)
    nn_reg.fit(X_train,y_train)
    
    # Save the neural model and scores
    nn_reg.model.save(os.path.join(directory_inter,nn_model_filename))
    pickle.dump(nn_reg_scores, open(os.path.join(directory_inter,nn_scores_filename), 'wb')) 
    
    return nn_reg, -nn_reg_scores

def catboost(catboost_params, cat_features,num_features,directory_inter,catboost_model_filename,catboost_scores_filename,X_train,y_train):
    """ CatBoost regression model with hyperparameter tuning and saving the model and scores
    Args:
        catboost_params: CatBoost regression hyperparameters. (dict)
        cat_features: Categorical features in catboost regression model. (str list)
        num_features: Numerical features in catboost regression model. (str list)
        directory_inter: Directory to save the model. (str)
        catboost_model_filename: Name of the filename for the catboost regression model to be saved. (str)
        catboost_scores_filename: Name of the filename for catboost regression scores to be saved. (str)
        X_train: A dataframe containing features to train on. (dataframe)
        y_test: A dataframe containing the label to train on. (dataframe)
    Returns:
        catboost_reg: CatBoost regression model after training to the best hyperparameters.
        -catboost_reg_scores: Scores of catboost regression.
    """
    
    # If the model ans scores are already calculated, then just load them
    if os.path.exists(os.path.join(directory_inter,catboost_model_filename)) and os.path.exists(os.path.join(directory_inter,catboost_scores_filename)):      
        catboost_reg = pd.read_pickle(os.path.join(directory_inter,catboost_model_filename))
        catboost_reg_scores = pd.read_pickle(os.path.join(directory_inter,catboost_scores_filename))       
        return catboost_reg, -catboost_reg_scores
    
    catboost_reg = CatBoostRegressor(loss_function='RMSE', cat_features=cat_features)
    grid_search = GridSearchCV(estimator = catboost_reg,
                               param_grid=catboost_params,
                               scoring = 'neg_mean_squared_error',
                               cv=5,
                               n_jobs=-1,
                               pre_dispatch = '2*n_jobs',
                               return_train_score = True) 
    
    grid_search.fit(X_train,y_train)
    catboost_reg = CatBoostRegressor(loss_function='RMSE', cat_features=cat_features,**grid_search.best_params_)
    catboost_reg_scores = cross_val_score(catboost_reg, X_train,y_train, scoring="neg_mean_squared_error", cv=5)
    catboost_reg.fit(X_train,y_train)
    
    # Save (pickle) the catboost regression model and scores
    pickle.dump(catboost_reg, open(os.path.join(directory_inter,catboost_model_filename), 'wb'))   
    pickle.dump(catboost_reg_scores, open(os.path.join(directory_inter,catboost_scores_filename), 'wb'))   
    
    return catboost_reg, -catboost_reg_scores

def predict_evaluate(model, test_data, test_data_labels, model_name, display=True):
    """ Print mse and rmse on test data, and returns both predictions and errors
    Args:
        model: A model to generate predictions.
        test_data: A dataframe containing test data features. (dataframe)
        test_data_labels: A dataframe containing test data labels. (dataframe)
        model_name: Name of the model. (str)
    Returns:
        predictions: Predictions of generated by model given.
        results: A list containing errors of mse and rmse.
    """
    predictions = model.predict(test_data)
    mse = mean_squared_error(test_data_labels, predictions)
    rmse = np.sqrt(mse)
    if display:
        print('Test data evaulation for ' + model_name)
        print('\nMSE: ', mse, '\nRMSE: ', rmse)
        print('---------------------------------\n')
    results = [mse, rmse]
    return predictions, results
    
def plot_test_and_predicted(results,model_name,cities):
    """Plot results for each city for a given model
    Args:
        results: A dataframe containing features, predictions, and label for a given model. (dataframe)
        model_name: Name of the model. (str)
        cities: A list of cities in consideration. (str list)
    """
    
    fig, axs = plt.subplots(ncols=5, nrows = 1, figsize=(20, 3))
    axs = axs.flatten()
    
    for ind , city in enumerate(cities):
        if model_name == "Catboost":
            label = results[results['city'] == city]['count_greater_than_PM_high']
            predictions = results[results['city'] == city]['predictions']
        else:
            label = results[results[city] == 1]['count_greater_than_PM_high']
            predictions = results[results[city] == 1]['predictions']
        
        x = [i for i in range(len(label))]
        axs[ind].plot(x,label,'o' ,color="red", label="test")
        axs[ind].plot(x, predictions,'x', color="blue", label="predicted,")
        axs[ind].title.set_text(city)
        axs[ind].legend()
        
    plt.suptitle(model_name)
    plt.tight_layout()
    
def plot_result(title, results, metrics_index):
    """ Bar plots of mse and rmse for all models 
    Args:
        title: Plot title. (str)
        results: Mse and rmse errors for each model. (list)
        metrics_index: 0 for mae, and 1 for rmse. (int)
    """
    plt.style.use('ggplot')
    x = ['Ridge Regression', "Neural Network","Catboost"]
    vals = []
    for model in results:
        vals.append(model[metrics_index])

    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, vals, color=['red', 'green', 'blue'])
    plt.xlabel("Models")
    plt.ylabel("Value")
    plt.title(title)
    plt.xticks(x_pos, x)
    plt.show()
    
def rmse_mae_monthly(monthly_test_data_labels,monthly_predictions,model_name):
    """ Calculates mse and rmse for monthly data 
    Args:
        monthly_test_data_labels: A dataframe containing test data for each month. (dataframe)
        monthly_predictions: A dataframe containing predictions for each month. (dataframe)
        model_name: Name of the model. (str)
    Returns:
        results: A list containing errors of mse and rmse.
    """
    mse = mean_squared_error(monthly_test_data_labels, monthly_predictions)
    rmse = np.sqrt(mse)
    if display:
        print('Test data evaulation for ' + model_name)
        print('\nMSE: ', mse, '\nRMSE: ', rmse)
        print('---------------------------------\n')
    results = [mse, rmse]
    return results