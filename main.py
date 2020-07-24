from skimage import io as sio
from skimage.feature import daisy
from dataset import Dataset
from time import time
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.preprocessing import Normalizer
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray


def extract_features(dataset):
    #ottieni il numero totale di immagini nel dataset
    nimgs = dataset.getLength()
    #crea una lista vuota di features
    features = list()
    ni=0 #numero di immagini analizzate finora
    total_time=0
    for cl in dataset.getClasses():
        paths=dataset.paths[cl]
        for impath in paths:
            t1=time() #timestamp attuale
            im=sio.imread(impath,as_gray=True) #carica immagine in scala di grigi
            feats=daisy(im,step=10) #estrai features
            feats=feats.reshape((-1,200)) #reshape dell'array'
            features.append(feats) #aggiungi features alla lista
            t2=time() #timestamp attuale
            t3=t2-t1 #tempo trascorso
            total_time+=t3
            #Stampa un messaggio di avanzamento, con la stima del tempo rimanente
            ni+=1 #aggiorna il numero di immagini analizzate finora
            if nimgs-ni==5:
                print ("...")
            if nimgs-ni<5:
                print ("Image {0}/{1} [{2:0.2f}/{3:0.2f} sec]".format(ni,nimgs,t3,t3*(nimgs-ni)))
    print ("Stacking all features...")
    t1=time()
    stacked = np.vstack(features) #metti insieme le feature estratte da tutte le immagini
    t2=time()
    total_time+=t2-t2
    print ("Total time: {0:0.2f} sec".format(total_time))
    return stacked

def extract_and_describe(img,kmeans):
    #estrai le feature da una immagine
    features=daisy(rgb2gray(img),step=10).reshape((-1,200))
    #assegna le feature locali alle parole del vocabolario
    assignments=kmeans.predict(features)
    #calcola l'istogramma
    histogram,_=np.histogram(assignments,bins=500,range=(0,499))
    #restituisci l'istogramma normalizzato
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
    y=list() #inizializziamo la lista delle etichette    
    X=list() #inizializziamo la lista delle osservazioni    
    paths=list() #inizializziamo la lista dei path        
    classes=dataset.getClasses()        
    total_number_of_images=dataset.getLength()    
    ni=0    
    t1=time()    
    for cl in classes: #per ogni classe        
        for path in dataset.paths[cl]: #per ogni path relativo alla classe corrente            
            img=sio.imread(path,as_gray=True) #leggi immagine            
            feat=extract_and_describe(img,kmeans) #estrai features            
            X.append(feat) #inserisci feature in X            
            y.append(classes.index(cl)) #inserisci l'indice della classe corrente in y            
            paths.append(path) #inserisci il path dell'immagine corrente alla lista            
            ni+=1            #rimuovere il commento di seguito per mostrare dei messaggi durante l'esecuzione    
            #Adesso X e y sono due liste. Per comodità, convertiamoli in array di numpy:   
    X=np.array(X)    
    y=np.array(y)   
    t2=time()    
    print ("Elapsed time {0:0.2f}".format(t2-t1))    
    return X,y,paths

print("DATASET COMPLETO - step=10")
dataset=Dataset('dataset')
classes=["edifici","quadri","sculture"]

print(dataset.getLength())
#dividiamo in test set e training set
training_set, test_set = dataset.splitTrainingTest(0.7) #70% training, 30% test
print(training_set.getLength())
print(test_set.getLength())



#------------Estrazione delle feature dal training set e costruzione del vocabolari


#estraiamo tutte le features dalle immagini del dataset
training_local_features = extract_features(training_set)

#Per costruire il nostro dizionario di parole visuali, utilizzeremo l'algoritmo di clustering K-Means incluso 
#nella libreria scikit-learn scegliendo come numero di cluster (parole visuali) 500. In particolare,
#utilizzeremo MiniBatchKMeans una implementazione dell'algoritmo ottimizzata per lavorare su un grande numero di campioni.

#inizializziamo l'oggetto "KMeans" impostando il numero di centroidi 
kmeans = KMeans(500, random_state=25) #avviamo il kmeans sulle feature estratte  
kmeans.fit(training_local_features) 
#i centroidi dei cluster ottenuti dall'algoritmo k-means sono conservati all'interno di k-means cluster
kmeans.cluster_centers_.shape

#codifichiamo le classi piuttosto che con delle stringhe, con degli indici numerici:
classes_idx=range(len(classes))

#estraggo le features dal training set:
X_training,y_training,paths_training=describe_dataset(training_set,kmeans) 
X_test,y_test,paths_test=describe_dataset(test_set,kmeans)


#NORMALIZZAZIONE TF-IDF
presence=(X_training>0).astype(int)
df=presence.sum(axis=0)
n=len(X_training)
idf=np.log(float(n)/(1+df))
X_training_tfidf=X_training*idf
X_test_tfidf=X_training*idf
norm=Normalizer(norm='l2')
X_training_tfidf_12=norm.transform(X_training_tfidf)
X_test_tfidf_12=norm.transform(X_test_tfidf)



#--------------------------------------------------------------------------------------KNN
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN 
#----------------------------------------------------------------------------1NN
nn5 = KNN(1) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("1-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)


#----------------------------------------------------------------------------3NN
nn5 = KNN(3) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("3-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

#----------------------------------------------------------------------------5NN
nn5 = KNN(5) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("5-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)


#----------------------------------------------------------------------------7NN
nn5 = KNN(7) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("7-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

#----------------------------------------------------------------------------9NN
nn5 = KNN(9) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("9-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

#----------------------------------------------------------------------------11NN
nn5 = KNN(11) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("11-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)


#----------------------------------------------------------------------------13NN
nn5 = KNN(13) 
nn5.fit(X_training,y_training) 
predicted_labels=nn5.predict(X_test) 
a = accuracy_score(y_test,predicted_labels) 

M = confusion_matrix(y_test,predicted_labels)
 
print ("13-NN, accuracy: %0.2f, Confusion Matrix:\n" %a) 
print (M)

#---------------------------------------------------------------------------NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB as NB
nb=NB()

#alleno il modello
nb.fit(X_training, y_training)
#valutiamo la performance
predicted_labels=nb.predict(X_test)
print("NAIVE BAYES: Accuracy: %0.2f, Confusion Matrix:/n"% accuracy_score(y_test,predicted_labels))
print(confusion_matrix(y_test,predicted_labels))

#---------------------------------------------------------------------------LOGISTIC

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
pca=PCA()
pca.fit(X_training)
X_training_pca=pca.transform(X_training)
X_test_pca=pca.transform(X_test)

lr=LogisticRegression() #viene usato il metodo one vs rest di default

lr.fit(X_training_pca,y_training)

p=lr.predict(X_test_pca)
print ("LOGISTIC REGRESSION: Accuracy: %0.2f, Confusion matrix:\n" % accuracy_score(y_test,p))
print(confusion_matrix(y_test,p))
