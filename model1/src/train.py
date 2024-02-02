import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter

from src.dataset import RUL_Dataset
from src.model import CNN1D_RUL
from src.RUL_loss_function import RUL_loss
from src.config import Config


def get_device():

    # chooses between nvidia gpu, apple silicon gpu or cpu
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'

    return device


def train_1_epoch(model, training_loader, loss_func, optimizer):
    
    tot_loss = 0.
    dev = next(model.parameters()).device

    for data in training_loader:

        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs[0].to(dev)
        labels = labels[0].to(dev)
        

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients

        loss = loss_func(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        tot_loss += loss.item()

    return tot_loss / len(training_loader)



def train_net(config, device):

    writer_path = ''.join(['runs/',config._name_config])
    writer = SummaryWriter(writer_path)

    model_path = ''.join(['saved_models/',config._name_config, '/epoch_', str(config.epoch_best_model)])

    # load the parameters of the model and put it on the gpu if available
    model = load_pmodel(config, device)
    model = model.to(device)

    print('Model bins: ',model.pyramid_pool_bins, '\nAnd model mode: ', model._pooling_mode)

    # load training and testing loaders
    training_dataset = RUL_Dataset(train_dir=config._train_dir, permutations=config.train_permutations )
    testing_dataset = RUL_Dataset(train_dir=config._eval_dir, permutations=config.eval_permutations )

    training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False)
    eval_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    # Training components needed
    optimizer = torch.optim.Adam(model.parameters(), lr=config._learning_rate)
    loss_func = RUL_loss(theta=config._theta)

    last_epoch = config.last_epoch

    
    for epoch in range(last_epoch, last_epoch + 200):

        print (f"EPOCH: {epoch+1} starting \n ...")
        model.train(True)

        avg_train_loss = train_1_epoch(model,training_loader=training_loader, loss_func=loss_func, optimizer=optimizer)
        
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        running_eval_loss = 0.0
        # Disable gradient computation and reduce memory consumption with torch.no_grad()
        with torch.no_grad():
            for eval_data in eval_loader:

                # Get data from evaluation dataset
                vinputs, vlabels = eval_data
                vinputs = vinputs[0].to(device)
                vlabels = vlabels[0].to(device)

                # Model estimation
                voutputs = model(vinputs)

                print(f"Eval outputs: {voutputs}\n Eval labels: {vlabels}\n")
                # Loss computations
                eval_loss = loss_func(voutputs, vlabels)
                running_eval_loss += eval_loss

        avg_eval_loss = running_eval_loss / len(eval_loader)
        print(f"LOSS train: {avg_train_loss} vs eval: {avg_eval_loss} for EPOCH: {epoch+1}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_train_loss, 'Validation' : avg_eval_loss },
                        epoch + 1)
        writer.flush()


        # Update config JSON with new epoch number
        config.last_epoch += 1

        # Track best performance, and save the model's state
        if avg_eval_loss < config.best_eval_loss:

            # Update best loss and model
            config.best_eval_loss = avg_eval_loss.item()
            # Update the config file for new best model
            config.epoch_best_model = config.last_epoch

            new_model_path = ''.join(['saved_models/',config._name_config, '/epoch_', str(config.epoch_best_model)])
            torch.save(model.state_dict(), new_model_path)

        
        # Save configuration file
        config.save()




def load_pmodel(config, device):

    model_path = ''.join(['saved_models/',config._name_config, '/epoch_', str(config.epoch_best_model)])
    model = CNN1D_RUL(config._pyramid_bins, config._pooling_mode)

    # If model doesn't exist at location, initialize it
    if not os.path.isfile(model_path):

        dir_path = ''.join(['saved_models/',config._name_config])
        os.makedirs(dir_path)

        print(f"Model weights for {model._get_name()} starting initialization\n ...")
        model = init_net(model)
        print(f"Model weights for {model._get_name()} finished initialization\n")

        print(f"Model weights saving at {model_path} \n ...")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved at {model_path} \n")
    
    else:
        print(f"Model weights for {model_path} loading \n...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights for {model_path} loaded\n")

    return model


def init_net (model):

    # from :https://pytorch.org/docs/stable/nn.init.html
    torch.no_grad()
    model.apply(init_weights)
    
    return model
    


def init_weights(layer):

    # Initialize weights and biases
    if isinstance(layer, nn.Conv1d):
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('leaky_relu', 0.01))
        layer.bias.data.fill_(0.)

    if isinstance(layer, nn.Linear) and layer.weight.data.shape[0] == 1:
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('sigmoid'))
        layer.bias.data.fill_(0.)

    if isinstance(layer, nn.Linear) and layer.weight.data.shape[0] != 1:
        nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('tanh'))
        layer.bias.data.fill_(0.)


