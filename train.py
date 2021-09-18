# import required libraries 
import torch
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models 
from torchvision import datasets, transforms, models 
import numpy as np
from PIL import Image
import argparse
import os
from collections import OrderedDict


# Main program function define below 
def main():
    # Creates and retrieves command line arguments 
    in_args = get_input_args()
    
    # Create pretrained model and define classifier to train network on dataset 
    model, hyperparameter = classifier(in_args.arch, in_args.hidden_units)
    
    #Load transform dataloader
    trainloader, validationloader, train_datasets = loader(in_args.data_directory)

    # define loss function 
    criterion = nn.NLLLoss()
    
    # Pass the classifier parameters to optimizer Adam
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
    # Move model to GPU if avaliable and use gpu is passed from command line
    if torch.cuda.is_available() and in_args.gpu:
        model.cuda()
    
    epochs = in_args.epochs 
    steps = 0
    print_every = 40 
    train_loss = 0 
    
    # loop through for number of epochs
    for e in range(epochs):
        # model in training mode, dropout is on
        model.train()
        
        #loop through all batches in trainloader
        for inputs, labels in iter(trainloader):
            steps += 1
            
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Move input and labels to GPU if avaialable and use gpu passed from command line
            if torch.cuda.is_available() and in_args.gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            #Make forward and backward pass calculating Loss and weights
            outputs = model.forward(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            
            #add training Loss for each batch 
            train_loss += train_loss.data[0]
            
            #step every 40 batches
            if steps % print_every == 0:
                
                #model in inference mode, dropout is off
                model.eval()
                Validation_Loss = 0
                Validation_accuracy = 0
                
                #loop through all batches in validation loader
                for inputs, labels in iter(validationloader):
                    
                    inputs = Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)
                    
                    #Move input and labels to GPU if avaialable and use gpu passed from command line
                    if torch.cuda.is_available() and in_args.gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    
                    #make forward pass through network and calculate validation loss
                    outputs = model.forward(inputs)
                    Validation_Loss += criterion(outputs, labels).data[0]
                    
                    #output will be exp of logmax
                    validationps = torch.exp(outputs).data
                    
                    #take max probability and compare with label class
                    Validation_Equality = (labels.data == validationps.max(1)[1])
                    
                    #calculate accuracy mean on validation batch 
                    Validation_accuracy += Validation_Equality.type_as(torch.FloatTensor()).mean()
                
                print("\nEpoch: {}/{}..".format(e+1, epochs))
                print("Training Loss: {:.4f}".format(train_loss.data[0]/print_every))
                print("Validation Loss: {:.4f}".format(Validation_Loss/len(validationloader)))
                print("Validation Accuracy: {:.4f}".format(Validation_accuracy/len(validationloader)))
                
                train_loss = 0
                
                #turn on the dropout for training
                model.train()
    
    # Set the model class_to_idx atttribute
    model.class_to_idx = train_datasets.class_to_idx
    
    #define checkpoint with parameters to be saved 
    checkpoint = {'input_size': hyperparameter['input_size'],
                 'output_size': 102,
                 'epochs': epochs,
                 'hidden_units': hyperparameter['hidden_units'],
                 'learning_rate': in_args.learning_rate,
                 'class_to_idx': model.class_to_idx,
                 'state_dict': model.state_dict(),
                 'model_name': in_args.arch}
    
    #If save_directory does not exits then create directory
    if not os.path.exists(in_args.save_dir):
        os.makedirs(in_args.save_dir)
    
    # Save checkpoint to set directory
    torch.save(checkpoint, in_args.save_dir + '/checkpoint.pth')
    
               
def loader(Image_Folder_path):
    '''
        Define your transforms for the training and validation sets
        Load the datasets with ImageFolder
        Using the image datasets and the transforms, define the dataloaders
        Input: Image_Folder_path - path to dataset folder 
        returns trainloader, validationloader and train_datasets 
    '''
    
    data_dir = Image_Folder_path        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'  
    
    # Define transforms for training and Validation datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    # Load the datasets with the ImageFolder
    train_datasets = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_datasets = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)
    
    # Using the image datasets and transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=False)
    
    return trainloader, validationloader, train_datasets
    

def classifier(model_name,hidden_units):
    '''
        Loads the pretained model, defines clasifier and returns model 
        Inputs: Model_Name = Model name passed in command line used as pretrained model 
                hidden_units = Number of hidden units to be used in classifier for pretrained model
        Returns: returns pretrained model with customize classifier 
                 returns hyperparameters dict to save later in checkpoint
    '''
    vgg16 = torchvision.models.vgg16(pretrained=True)
    densenet121 = torchvision.models.densenet121(pretrained=True)
    
    # dict for pretrained model
    models = {'vgg16': vgg16, 'densenet121': densenet121}
    
    # dict to use for in_features
    in_features = {'vgg16': 25088, 'densenet121': 1024}
    
    # default hidden units if not provided from command line
    default_hidden_units = {'vgg16': 4096, 'densenet121': 512}
    
    # If hidden_units is not passed from command line then default based on model chosen
    if hidden_units == 0:
        hidden_units = default_hidden_units[model_name]
    
    # load pretrained chosen model 
    model = models[model_name]
    
    # Freeze all the parameters in the feature network 
    for param in model.parameters():
        param.require_grad = False
    
    #define layers in classifier
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features[model_name],hidden_units)),
                                           ('relu', nn.ReLU()),
                                           ('Dropout', nn.Dropout(0.5)),
                                           ('fc2', nn.Linear(hidden_units,102)),
                                           ('output', nn.LogSoftmax(dim=1))]))
    
    # Set the classifier to model
    model.classifier = classifier
    
    #Add hyperparameters to dict for saving in checkpoint later
    hyperparameter = {'input_size': in_features[model_name],
                     'hidden_units': hidden_units}
    
    return model, hyperparameter
    
    

def get_input_args():
    """ 
    Retrieves and parses the command line arguments created and defined using the argparse module. 
    Returns: parse_args() command line arguments
    """
    
    parser = argparse.ArgumentParser()
    
    # command line parameter for dataset path
    parser.add_argument('data_directory', action='store', type=str, 
                       help='directory for new dataset to train')
    
    # command line parameter for saving checkpoint path 
    parser.add_argument('--save_dir',type=str,default='checkpoint',
                        help='directory where checkpoint is saved')
    
    # command line parameter to input model chosen. Defaults to vgg16. Chosen model can be either vgg16 or densenet121
    parser.add_argument('--arch',type=str,default='vgg16',choices=('vgg16', 'densenet121'),
                        help='chosen model: vgg16 or densenet121')
    
    # command line parameter to input learning rate. Defaults to 0.00001
    parser.add_argument('--learning_rate',type=float,default='0.00001',
                       help='learning rate for model')
    
    # command line parameter to input number of hidden units
    parser.add_argument('--hidden_units',type=int,default=0,
                       help='add number of units in hidden layer to list')
    
    # command line paramter to input epochs. Defaults to 5
    parser.add_argument('--epochs',type=int,default='5',
                       help='number of epochs')
    
    # command line parameter for using gpu
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Set a switch to true')
    
    #return parsed argument collection
    return parser.parse_args()

if __name__ == "__main__":
    main()

    

