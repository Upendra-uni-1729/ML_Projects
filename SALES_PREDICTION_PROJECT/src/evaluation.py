

# template to get different model scores
class Model_Score():

    def __init__(self,Score_type,Y_true,Y_pred):
        """
        Taking attributes the predicted target and actual target
        """
        self.score_type = Score_type
        self.y_true = Y_true
        self.y_pred = Y_pred

    def get_score(self):

        """
        returns the score
        """
        return f'{str(self.score_type.__name__)} : {self.score_type(self.y_true,self.y_pred)}'
        
        