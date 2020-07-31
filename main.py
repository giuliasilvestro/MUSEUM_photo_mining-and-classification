from skimage import io as sio
from skimage.feature import daisy
from dataset import Dataset
from time import time
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.preprocessing import Normalizer
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN 
from sklearn.naive_bayes import MultinomialNB as NB 


def extract_features(dataset):
    nimgs = dataset.getLength()                      # Obtain the total number of images in the dataset
    features = list()                                # Create an empty list of features 
    ni=0                                             # Number of images analyzed
    total_time=0
    for cl in dataset.getClasses():
        paths=dataset.paths[cl]
        for impath in paths:
            t1=time()                                # Current timestamp 
            im=sio.imread(impath,as_gray=True)       # Read images in gray-scale 
            feats=daisy(im,step=10)                  # Extract feature 
            feats=feats.reshape((-1,200))          
            features.append(feats)                   # Add features to the list 
            t2=time()                                # Current timestamp 
            t3=t2-t1                                 # Elapsed time
            total_time+=t3
            # Print advancement message with estimated remaining time 
            ni+=1                                    # Update the number of images analyzed 
            if nimgs-ni==5:
                print ("...")
            if nimgs-ni<5:
                print ("Image {0}/{1} [{2:0.2f}/{3:0.2f} sec]".format(ni,nimgs,t3,t3*(nimgs-ni)))
    print ("Stacking all features...")
    t1=time()
    stacked = np.vstack(features)                    # Stack together the features extracted from all the images 
    t2=time()
    total_time+=t2-t2
    print ("Total time: {0:0.2f} sec".format(total_time))
    return stacked

def extract_and_describe(img,kmeans):
    features=daisy(rgb2gray(img),step=10).reshape((-1,200))            # Extract features from one image           
    assignments=kmeans.predict(features)                               # Assign the local features to the vocabulary words
    histogram,_=np.histogram(assignments,bins=500,range=(0,499))       # Compute the histogram
    return histogram

def display_image_and_representation(X,y,paths,classes,i):
	im=sio.imread(paths[i])
	plt.figure(figsize=(12,4))
	plt.suptitle("Class: {0} - Image: {1}".format(classes[y[i]],i))
	plt.subplot(1,2,1)
	plt.imshow(im)
	plt.subplot(1,2,2)
	plt.plot(X[i])
	plt.show()

def show_image_and_representation(img,image_representation):
    plt.figure(figsize=(13,4))
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.plot(image_representation)
    plt.show()
	
def compare_representations(r1,r2):
	plt.figure(figsize=(12,4))
	plt.subplot(1,2,1)
	plt.plot(r1)
	plt.subplot(1,2,2)
	plt.plot(r2)
	plt.show()
	
def describe_dataset(dataset,kmeans):    
    y=list()                                                            # Initialize the list of labels  
    X=list()                                                            # Initialize the list of observations 
    paths=list()                                                        # Initialize the list of paths         
    classes=dataset.getClasses()        
    total_number_of_images=dataset.getLength()    
    ni=0    
    t1=time()    
    for cl in classes:                                                  # For each class        
        for path in dataset.paths[cl]:                                  # For each path 
            img=sio.imread(path,as_gray=True)                           # Read the image             
            feat=extract_and_describe(img,kmeans)                       # Extract features            
            X.append(feat)                                              # Load features in X 
            y.append(classes.index(cl))                                 # Insert the intex of the current class in Y             
            paths.append(path)                                          # Insert the image path of the current image in the list 
            ni+=1                                                      
            print("Processing Image {0}/{1}".format(ni,total_number_of_images))  
    X=np.array(X)                                                       # Convert X and y from list to numpy arrays  
    y=np.array(y)   
    t2=time()    
    print ("Elapsed time {0:0.2f}".format(t2-t1))    
    return X,y,paths


## EXTRACTION OF FEATURES FROM THE TRAINING SET AND CONSTRUCTION OF THE VOCABULARIES ##
dataset_path="dataset/dataset"
dataset=Dataset(dataset_path)
classes=["artifacts","paintings","sculptures"]
print("Total images: %d" % dataset.getLength())
training_set, test_set = dataset.splitTrainingTest(0.7)                 # 70% training, 30% test
print("Total images in training set: %d" % training_set.getLength())
print("Total images in test set: %d" % test_set.getLength())

# Extract the features from all images in the dataset
training_local_features = extract_features(training_set)

# To build our dictionary of visual words, we will use the K-Means clustering algorithm (scikit-learn), choosing 500 clusters (visual words).
kmeans = KMeans(500, random_state=25)  
kmeans.fit(training_local_features) 
kmeans.cluster_centers_.shape

# Reference the classes with numeric indeces rather than strings:
classes_idx=range(len(classes))

# Extract features from the training set:
X_training,y_training,paths_training=describe_dataset(training_set,kmeans) 
X_test,y_test,paths_test=describe_dataset(test_set,kmeans)

#TF-IDF Normalization:
presence=(X_training>0).astype(int)
df=presence.sum(axis=0)
n=len(X_training)
idf=np.log(float(n)/(1+df))
X_training_tfidf=X_training*idf
X_test_tfidf=X_training*idf
norm=Normalizer(norm='l2')
X_training_tfidf_12=norm.transform(X_training_tfidf)
X_test_tfidf_12=norm.transform(X_test_tfidf)


## TRAINING THE MODELS ##

# 1NN
nn5 = KNN(1) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 
M = confusion_matrix(y_test,predicted_labels)
print ("1-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

# 3NN
nn5 = KNN(3) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 
M = confusion_matrix(y_test,predicted_labels)
print ("3-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

# 5NN
nn5 = KNN(5) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 
M = confusion_matrix(y_test,predicted_labels)
print ("5-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

# 7NN
nn5 = KNN(7) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 
M = confusion_matrix(y_test,predicted_labels)
print ("7-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

# 9NN
nn5 = KNN(9) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 
M = confusion_matrix(y_test,predicted_labels)
print ("9-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

# NAIVE BAYES
nb=NB()                                       # Train the model
nb.fit(X_training, y_training)
predicted_labels=nb.predict(X_test)           # Evaluate the performance 
print("NAIVE BAYES: Accuracy: %0.2f, Confusion Matrix:/n"% accuracy_score(y_test,predicted_labels))
print(confusion_matrix(y_test,predicted_labels))

# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(X_training)
X_training_pca=pca.transform(X_training)
X_test_pca=pca.transform(X_test)
lr=LogisticRegression()                        #  One-vs-rest method as default 
lr.fit(X_training_pca,y_training)
p=lr.predict(X_test_pca)
print ("LOGISTIC REGRESSION: Accuracy: %0.2f, Confusion matrix:\n" % accuracy_score(y_test,p))
print(confusion_matrix(y_test,p))
