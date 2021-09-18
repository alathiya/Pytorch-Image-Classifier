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
from collections import OrderedDict

#Main program function defined below
def main():
    # Read arguments from command line
    in_args = get_input_args()    
    
    # Load model from checkpoint
    model = load_checkpoint(in_args.checkpoint)
    
    # predict topK class probabilities 
    probs, classes = predict(in_args.input_image, model, in_args.top_k, in_args.gpu)
    
    #read file from command line and extract file content into dictionary
    cat_to_name_dict = {}
    cat_to_name = in_args.category_names
    if cat_to_name:
        cat_to_name_data = cat_to_name.read()
        cat_to_name.close()
        arr_classes = cat_to_name_data.split(',')
    
        #loop through class array and strip off any spaces, quotation and brackets to convert into dictionary
        for item in arr_classes:
            cat_to_name_dict[item.split(':')[0].strip().strip('{').strip('"')] = item.split(':')    [1].strip().strip('}').strip('"')
    
        #loop through topK classes to get the list of flower names 
        Flower_Names = []
        for item in classes:
            Flower_Names.append(cat_to_name_dict[item])
        
        # print Flower names and associated probabilites 
        for flower_class, prob in zip(Flower_Names, probs):
            print("Flower Class: {},  Probability: {:.4f}".format(flower_class,prob))
    else:
        #print class and associated probability
        for class_cat, prob in zip(classes, probs):
            print("Class: {},  Probability: {:.4f}".format(class_cat,prob))
            
        
def predict(Image_Path, model,topK,use_gpu):
    '''
        predicts the top K probability for the input image and returns respective class category. 
    '''
    ps_topk = []
    ps_topk_idx = []
    idx_to_class= {}
    ps_topk_class = []
    
    #Load the image into PIL image object 
    image = Image.open(Image_Path)
    
    #Convert PIL image for a pytorch model 
    image_tensor = process_image(image)
    
    #Convert 3D into 4D array to feed into model. 
    image_4d = np.expand_dims(image_tensor, axis = 0)
    image_tensor = torch.from_numpy(image_4d)
    
    image_tensor = Variable(image_tensor).float()
    
    if torch.cuda.is_available() and use_gpu:
        image_tensor = image_tensor.cuda()
        model = model.cuda()
        
    #make forward pass through model
    outputs = model.forward(image_tensor)
    
    #convert logsoftmax output into exponential 
    ps = torch.exp(outputs).data
    
    #move tensor to cpu 
    ps = ps.cpu()
    
    #retrieve top k probability and indexes from outputs
    ps_topk = ps.topk(topK)[0].numpy()[0]
    ps_topk_idx = ps.topk(topK)[1].numpy()[0]
    
    #loop through class_to_idx dict to reverse key, values in idx_to_class dict
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key
        
    # loop through index to retrieve class from idx_to_class dict 
    for item in ps_topk_idx:
        ps_topk_class.append(idx_to_class[item])
    
    return ps_topk, ps_topk_class

def process_image(image):
    '''
        Scales, crops and normalizes a PIL image for a PyTorch model, 
        returns a Numpy array
    '''
    basewidth = None
    baseheight = None 
    
    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio 
    if image.size[0] < image.size[1]:
        basewidth = 256 
    else:
        baseheight = 256 
    
    if basewidth:
        wpercent = (basewidth/float(image.size[0]))
        hsize = int(float(image.size[1])*float(wpercent))
        image = image.resize((basewidth,hsize), Image.ANTIALIAS)
    
    if baseheight:
        hpercent = (baseheight/float(image.size[1])) 
        wsize = int(float(image.size[0])*float(hpercent))
        image = image.resize((wsize,baseheight), Image.ANTIALIAS)
    
    # crop image to center at 224, 224
    image = image.crop((16,16,240,240))
    
    #convert the image into numpy array 
    img = np.array(image)
    
    #Normalize image substracting mean and dividing by standard deviation 
    img = (img - np.mean(img))/np.std(img)
    
    #return transpose of image with color channel at first dim
    return img.transpose(2,0,1)
    

def load_checkpoint(filepath):
    '''
        Load a checkpoint from filepath and rebuild model.  
    '''
    #Load checkpoint dictionary 
    checkpoint = torch.load(filepath)
    
    #Load pretrained model based on model_name saved in checkpoint
    if checkpoint['model_name'] == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    else:
        model = torchvision.models.densenet121(pretrained=True)
     
    # define classifier from checkpoint parameters
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_units'])),
                                           ('relu', nn.ReLU()),
                                           ('Dropout', nn.Dropout(0.5)),
                                           ('fc2', nn.Linear(checkpoint['hidden_units'],checkpoint['output_size'])),
                                           ('output', nn.LogSoftmax(dim=1))]))
    
    # Set the classifier to model 
    model.classifier = classifier 
    
    #load state_dict into model 
    model.load_state_dict(checkpoint['state_dict'])
    
    #load class_to_idx attribute
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    

def get_input_args():
    """ 
    Retrieves and parses the command line arguments created and defined using the argparse module. 
    Returns: parse_args() command line arguments
    """
    
    parser = argparse.ArgumentParser()
    
    # Input Command line parameter. Reads input image path.  
    parser.add_argument('input_image', action='store', type=str, 
                       help='input image path')
    
    # checkpoint command line parameter. Reads checkpoint filepath. 
    parser.add_argument('checkpoint', action='store', type=str, 
                       help='checkpoint path')
    
    # top_k command line parameter. defaults to top 5. 
    parser.add_argument('--top_k',type=int,default=5,
                       help='returns top k probability')
    
    # command line parameter for loading json file cat_to_name
    parser.add_argument('--category_names',metavar='in-file',type=argparse.FileType('rt'))
    
    # command line parameter for using gpu
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Set a switch to true')
    
    #return parsed argument collection
    return parser.parse_args()
    
if __name__ == "__main__":
    main()