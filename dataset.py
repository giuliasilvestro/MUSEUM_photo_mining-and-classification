import os
import glob
from skimage import io as sio
from matplotlib import pyplot as plt
import numpy as np
from copy import copy

class Dataset:
    
    def __init__(self,path_to_dataset):
        self.path_to_dataset = path_to_dataset
        classes=sorted(os.listdir(path_to_dataset))
        self.paths = dict()
        for cl in classes: 
            current_paths=sorted(glob.glob(os.path.join(path_to_dataset,cl,"*.jpg")))
            self.paths[cl]=current_paths
            
    def getImagePath(self,cl,idx):
        return self.paths[cl][idx]
    
    def getClasses(self):
        return sorted(self.paths.keys())
        
    def showImage(self,class_name,image_number):
        im = sio.imread(self.getImagePath(class_name,image_number)) # Read the image from the file.
        plt.figure() # Create a new figure.
        plt.imshow(im) # Show the figure.
        plt.show()
    
    def getNumberOfClasses(self):
        return len(self.paths)
    
    def getClassLength(self,cl):
        return len(self.paths[cl])
    
    def getLength(self):
        return sum([len(x) for x in self.paths.values()])
    
    def restrictToClasses(self,classes):
        new_paths = {cl:self.paths[cl] for cl in classes}
        self.paths=new_paths
    
    def splitTrainingTest(self,percent_train):
        training_paths=dict()                               # Initialize the dictionaries that will contain...
        test_paths=dict()                                   # ...the paths of the training and test set.
        for cl in self.getClasses():                        # For each class... 
            paths=self.paths[cl]                            # ... obtain a list of the paths relative to the current class.  
            split_idx=int(len(paths)*percent_train)         # Index at which the array is split.
            training_paths[cl]=paths[0:split_idx]           # Save the first "split_idx" images in the training set...
            test_paths[cl]=paths[split_idx::]               # ...and the rest in the test set.
        training_dataset = copy(self)                       # To create the training dataset, copy the current instance of this dataset.
        training_dataset.paths=training_paths               # Update list of paths.
        test_dataset = copy(self)                           # Do the same for the test set.
        test_dataset.paths=test_paths
        return training_dataset,test_dataset
