from SALES_PREDICTION_PROJECT.src.data_ingest import IngestData
from SALES_PREDICTION_PROJECT.src.model_build import Build_Model
from SALES_PREDICTION_PROJECT.src.evaluation import Model_Score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import logging

if __name__=="__main__":

    #log file location and format specification
    logging.basicConfig(filename='app.log', level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

    
    # ingested data path is specified in str format
    data = IngestData(path='./SALES_PREDICTION_PROJECT/data/Processed_Advertising_DataSet.csv')

    #getting the data for training and testing purpose
    X_train,X_test,Y_train,Y_test = data.get_data()

    try:
        # building linear regression model

        lr_model = Build_Model(LinearRegression())
        lr_model.fit_model(X_train,Y_train)
        lr_model_y_pred = lr_model.model_predict(X_test)

        # Evaluating the model with r2_score

        lr_model_r2_score = Model_Score(r2_score,Y_test,lr_model_y_pred)
        print(lr_model_r2_score.get_score())

        # mean_square_error

        lr_model_MSE = Model_Score(mean_squared_error,Y_test,lr_model_y_pred)
        print(lr_model_MSE.get_score())

        logging.info("Model Building successful")
        
    except Exception as e:
        
        logging.error(f"Model Building Failed : {e}")
    



    