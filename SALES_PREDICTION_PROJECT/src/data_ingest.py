import logging
import pandas as pd
from sklearn.model_selection import train_test_split



class IngestData:
    def __init__(self,path : str):
        """
        need an argument path of type str
        """
        self.path = path

    def get_data(self) -> pd.DataFrame:
        """
        Read data from path and return data frame
        """
        try:
            logging.info('sucessfully ingested the data')
            """
            reading data from the path (which is already a processed data)
            """
            data = pd.read_csv(self.path)

            # seperating the features and target
            X,Y = data.drop(columns=['sales']),data['sales']

            # split the data for training and prediction
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.2)
            return X_train,X_test,Y_train,Y_test
            
        except Exception as e:
            logging.error(f'Unable get data file {e}')
        
