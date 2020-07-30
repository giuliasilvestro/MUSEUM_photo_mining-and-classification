# MUSEUM - Photo Mining and Classification Methods
The aim of the project is to create a classifier that classifies, with the best possible performance, images of objects commonly found in museums, divided into 3 categories: **artifacts**, **sculptures**, **paintings**.

## DATA
The dataset consists of 110 images belonging to the 'paintings' category, 89 to the 'artifacts' category and 121 to the 'sculptures' category. The images that make up the dataset were obtained using the Flickr API: social media that allows members to share personal photographs. (code: download_images.py).

The dataset was then necessarily divided into two parts: "training set", containing 70% of the images of each category and "test set" which contains 30%. The training set is used by classifier models in the training phase, the test set, instead, in the performance evaluation phase.

## REPRESENTATION OF DATA
The data representation model used is the Bag of Visual Words (BOVW): construction of a "dictionary of visual words" useful for describing images. This dictionary is created by extracting local features from the images of the training set using Daisy Computation, which returns grids of descriptors extracted from the images in the form of three-dimensional tensors.
The “step” parameter of the daisy function indicates the distance between the descriptor sampling points; therefore determines the number of features extracted.
For the purposes of the project both classifiers will be trained using the representation of BOVW data with feature extraction using the parameter of the daisy function "step = 4" (more features will be extracted.
A clustering algorithm (K-Means) is then used on the extracted features to obtain a predetermined number of "visual words".
Once a "visual words" histogram has been created, which contains the number of occurrences of each visual word within each image, the TF-IDF (Term Frequency Inverse Document Frequency) normalization method is used. This process is important because the scale of the values ​​of each representation depends on the number of local features extracted from the image. The TF-IDF method gives a value of different importance to each word, eliminating too frequent words from the vocabulary that do not allow us to discriminate between images.

## ALGORITHMS
Three different classifier models were used:
1. **KNN**: Algorithm whereby each sample of the test set is represented in the same space as the features of the training set; the class to be assigned is decided considering the class to which the k elements of the closest training set belong, considering the one that appears several times in the set of k elements.
7 models with different K values ​​were trained: k [1,9] with odd k.
2. **Multinomial logistic regression**: Built using the one-vs-all paradigm, according to which, in this case, 3 logistic classifiers are trained and each of whom discriminates a given class from all the others.
3. **Naive Bayes classifier**: It uses Bayes' theorem, and is based on the fact that all the features are not related to each other (Naive Bayes Assumption). It calculates the probability of belonging to a class and the final decision is identified in the class that obtains the highest probability value.

## RESULTS

    **1-NN**, accuracy: **0.68**, Confusion Matrix:
    [[20  3  4]
     [ 4 24  5]
     [11  4 22]]

    **3-NN**, accuracy: **0.70**, Confusion Matrix:
    [[22  0  5]
     [ 8 23  2]
     [11  3 23]]

    **5-NN**, accuracy: **0.67**, Confusion Matrix:

    [[23  0  4]
     [ 8 21  4]
     [11  5 21]]

    **7-NN**, accuracy: **0.65**, Confusion Matrix:
    [[21  1  5]
     [10 20  3]
     [12  3 22]]
    **9-NN**, accuracy: **0.68**, Confusion Matrix:
    [[21  2  4]
     [ 8 23  2]
     [11  4 22]]

    **NAIVE BAYES**: Accuracy: **0.71**, Confusion Matrix:/n
    [[18  4  5]
     [ 2 22  9]
     [ 2  6 29]]

    LOGISTIC REGRESSION: Accuracy: 0.74, Confusion matrix:
    [[19  1  7]
     [ 3 24  6]
     [ 5  3 29]]
