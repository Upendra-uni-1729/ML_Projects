
#This is the template class for any ml model building
class Build_Model():

    def __init__(self,model):
        """
        instance with X and Y with none values
        """
        self.model = model
        self.X=None
        self.Y=None

    
    def fit_model(self,X,Y):
        """
        model creation with attributes as training set
        returns trained model
        """
        self.X,self.Y = X,Y
        self.model.fit(self.X,self.Y)
    
    def model_predict(self,X):
        """
        model predict with attributes as test set
        returnas the test set
        """
        self.X=X
        return self.model.predict(self.X)
    