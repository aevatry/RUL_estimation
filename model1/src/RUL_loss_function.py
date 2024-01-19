import torch

class RUL_loss(torch.nn.Module):

    '''
    Based on paper: "Estimation of Remaining Useful Life Based on Switching Kalman Filter Neural Network Ensemble"
    And stack stats exchange post :https://stats.stackexchange.com/questions/554282/how-to-train-a-neural-network-to-minimize-two-loss-functions
    '''

    def __init__(self, theta):
        super().__init__()
        self.theta = theta
    
    def forward(self, predicted, labels):
        
        diff = predicted - labels
        loss_scoring = self.scoring(diff)
        loss_RMSE = self.RMSE(diff)
        loss = self.theta*loss_scoring + (1-self.theta)*loss_RMSE # see https://stats.stackexchange.com/questions/554282 
        return loss


    # Need the operation on tensors to be done using the torch library, not numpy
    def RMSE(self, diff):
        loss_RMSE = torch.sqrt(torch.mean(diff**2))
        return loss_RMSE
    
    def scoring(self, diff):
        loss_scoring = sum(
            torch.exp(-diff[i]/13) -1 for i in range(len(diff))
            if diff[i]<0 
        ) + sum(
            torch.exp(diff[i]/10) -1 for i in range(len(diff))
            if diff[i]>=0 
        )
        return loss_scoring