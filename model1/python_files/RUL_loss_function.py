import torch

class RUL_loss(torch.nn.Module):

    '''
    Based on paper: "Estimation of Remaining Useful Life Based on Switching Kalman Filter Neural Network Ensemble"
    And stack stats exchange post :https://stats.stackexchange.com/questions/554282/how-to-train-a-neural-network-to-minimize-two-loss-functions
    '''

    def __innit__(self):
        super().__innit__()
    
    def forward(self, predicted, true, theta):
        
        diff = predicted - true
        loss_scoring = self.scoring(diff)
        loss_RMSE = self.RMSE(diff)
        loss = theta*loss_scoring + (1-theta)*loss_RMSE # see https://stats.stackexchange.com/questions/554282 
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