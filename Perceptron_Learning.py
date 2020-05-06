import os
import math
import string
from nltk import *
import numpy as np


class Perceptron:
    #Initializations :
    def __init__(self, learning_rate, _iterations):
        
        self._iterations = _iterations
        self.learning_rate = learning_rate
        self.weights = []
        self.bias = None
        
        
    #Training phase :
    def fit(self, X_train, Y_train, nfeatures):
        self.bias = 0
        n = len(X_train)
        m = nfeatures
        #Initalizing the weights to zero
        self.weights = np.zeros(m)
        
        #To loop over the desired number of iterations
        for _ in range(self._iterations):
            for idx in range(n):
                # Dot product of W'(transpose) * X_train
                temp = sum (X_train[idx][k] * self.weights[k] for k in X_train[idx] )
                #Getting the predictions
                _predicted = int(temp > 0)
                # Update Rule:
                for j in X_train[idx]:
                    #delta(w) = lr * (trues - predicted) * X_train
                    update = self.learning_rate * (Y_train[idx] - _predicted)
                    # updating the weights and bias
                    self.weights[j] += update  * X_train[idx][j]
                    self.bias += update

    #Testing Phase :
    def predict(self, X_test):
        res = []
        #looping over the entire X_test
        for i in range(len(X_test)):
            # Dot product of W'(transpose) * X_test
            temp = sum (X_test[i][k] * self.weights[k] for k in X_test[i])
            res = res + [int(temp > 0)]
            #returning the result
        return res

    def accuracy(self,predicted, actual):
        #Gettting the number of correctly classified 
        result = float(sum( int(predicted[i] == actual[i]) for i in range(len(predicted)))) / float(len(predicted))
        #returning the result
        return result
    
# Helper Function to list the contents of a directory
def _directory_reading(directory):
    directory_entries = os.listdir(directory)
    return directory_entries

#helper function to develop a dictionary of each individual file
def dictionary_file(all_words,dictionary):
    feature = {}
    for w in dictionary:
        feature[all_words.index(w)+1] = dictionary[w]
    feature[0] =1
    return feature

#Function for building the dictionary/vocabulary
def _building_dictionary(the_path,with_stopwords):
    if with_stopwords.lower() =="yes":
        features = []
        for filename in _directory_reading(the_path):
            words = read_dataset(the_path + filename)
            the_dict = {}
            for w in words:
                if w in stopwords:
                    #Checking if it is a stopword
                    continue
                else:
                    #If word is absent
                    if the_dict.get(w, 0) == 0:
                        the_dict[w] = 1
                    #If the word is already present
                    else:
                        the_dict[w] += 1
            
            temp = dictionary_file(all_words,the_dict)
            features.append(temp)
        return features
    elif with_stopwords.lower() =="no":
        features = []
        for filename in _directory_reading(the_path):
            words = read_dataset(the_path + filename)
            the_dict = {}
            for w in words:
                #If word is absent
                if the_dict.get(w, 0) == 0:
                    the_dict[w] = 1
                #If the word is already present
                else:
                    the_dict[w] += 1
            
            temp = dictionary_file(all_words,the_dict)
            features.append(temp)
        return features
    else:
        print("\n Enter either 'yes' or 'no'.\n")
        

#To remove duplicates from a list
def Remove(duplicate): 
	final_list = [] 
	for num in duplicate: 
		if num not in final_list: 
			final_list.append(num) 
	return final_list 

# to get the total number of features i.e. the number of total unique words
def extract_words(path,kind):
    if kind.lower() == "yes":
        all_words = []
        #for path in target_dir:
        for filename in _directory_reading(path):
            words = read_dataset(path + filename)
            for w in words:
                if not w in stopwords:
                    all_words.append(w)
                    
        all_words = Remove(all_words)
        all_words = sorted(all_words)  
        return all_words 
    elif kind.lower() == "no":
        all_words = []
        #for path in target_dir:
        for filename in _directory_reading(path):
            words = read_dataset(path + filename)
            for w in words:
                all_words.append(w)
        #duplicates removal            
        all_words = Remove(all_words)
        #sorting the words alphabetically
        all_words = sorted(all_words)  
        return all_words
    else:
        print("\n Enter either 'yes' or 'no'.\n")
        
              
  
#stemming the word before putting it into the dictionary
def _stemmer(word):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(word)

# Function to load the dataset
def read_dataset(file_path):
    words = []
    f_file = open(file_path, 'r')
    for line in f_file:
        for word in line.split():
            if sum(1 for chr in word 
                    if chr.islower())>1:
                new = word.strip(string.punctuation)
                new = new.lower()
                new = _stemmer(new)
                words.append(new)
    f_file.close()
    return words
    
    
#Merging any two lists
def merge(first,second):
    temp = []
    for i in first:
        temp.append(i)
    for j in second:
        temp.append(j)
    return temp


# True labels development
def create_trues(ham,spam,temp):
    ham = np.zeros(len(ham)) 
    spam = np.ones(len(spam))
    temp = merge(list(ham),list(spam))
    return temp




stopwords = []  
f = open("stopwords.txt", 'r')
for line in f:
    for word in line.split():
        stopwords.append(word)

#Loading the Datasets
train_ham_path,train_spam_path = os.sys.argv[1],os.sys.argv[2]
test_ham_path,test_spam_path = os.sys.argv[3],os.sys.argv[4]


Number_of_iterations = [15,30,45,60,70]
Differing_learning_rates = [0.0001,0.001,0.01,0.1]

_list = []
for i in Number_of_iterations:
    for j in Differing_learning_rates:
        _list.append((i,j))



sample = open("Perc_Accuracy.txt","w")
def write_to_file(text,sample):
    sample.write(text)

####################################################################

write_to_file("\n Removing Stopwords: \n ",sample)
print("\n Removing Stopwords: \n ")

all_words = []
for path in train_ham_path,train_spam_path,test_ham_path,test_spam_path:
    all_words += extract_words(path,"yes")
length_all = len(all_words) +1

#Building Dictionaries by removing stopwords
train_ham,train_spam  = _building_dictionary(train_ham_path,"yes"),_building_dictionary(train_spam_path,"yes")
test_ham,test_spam   = _building_dictionary(test_ham_path,"yes"),_building_dictionary(test_spam_path,"yes")


X_train = merge(train_ham , train_spam)
X_test  = merge(test_ham , test_spam)


Y_train,Y_test = [],[]
Y_train = create_trues(train_ham,train_spam,Y_train)
Y_test = create_trues(test_ham,test_spam,Y_test)


local_accuracy = []
for _iterations,learning_rate in _list:
    perceptron = Perceptron(learning_rate, _iterations)
    perceptron.fit(X_train, Y_train, length_all)
    pred = perceptron.predict(X_test)
    acc = perceptron.accuracy(pred, Y_test)
    local_accuracy.append(acc)
    print("n_iter = "+ str(_iterations) +"\tlearning_rate =  "+ str(learning_rate) + "\taccuracy = {:.5f}".format( acc ))
    write_to_file("n_iter = "+ str(_iterations) +"\tlearning_rate =  "+ str(learning_rate) + "\taccuracy = {:.5f}".format( acc ),sample)
    write_to_file("\n",sample)

avg = sum(local_accuracy)/len(local_accuracy)
print("\n Average accuracy :")
write_to_file("\n Average accuracy :" + str(avg)+"\n",sample)
print(avg)


####################################################################
print("\n Keeping Stopwords: \n ")
write_to_file("\n Keeping Stopwords: \n ",sample)


all_words = []
for path in train_ham_path,train_spam_path,test_ham_path,test_spam_path:
    all_words += extract_words(path,"no")
length_all = len(all_words) +1

#Building Dictionaries by keeping stopwords
train_ham,train_spam  = _building_dictionary(train_ham_path,"no"),_building_dictionary(train_spam_path,"no")
test_ham,test_spam   = _building_dictionary(test_ham_path,"no"),_building_dictionary(test_spam_path,"no")

#Merging the lists
X_train = merge(train_ham,train_spam)
X_test  = merge(test_ham,test_spam)

Y_train,Y_test = [],[]
Y_train = create_trues(train_ham,train_spam,Y_train)
Y_test = create_trues(test_ham,test_spam,Y_test)

local_accuracy = []
for _iterations,learning_rate in _list:
    perceptron = Perceptron(learning_rate, _iterations)
    perceptron.fit(X_train, Y_train, length_all)
    pred = perceptron.predict(X_test)
    acc = perceptron.accuracy(pred, Y_test)
    local_accuracy.append(acc)
    print("n_iter = "+ str(_iterations) +"\tlearning_rate =  "+ str(learning_rate) + "\taccuracy = {:.5f}".format( acc ))
    write_to_file("n_iter = "+ str(_iterations) +"\tlearning_rate =  "+ str(learning_rate) + "\taccuracy = {:.5f}".format( acc ),sample)
    write_to_file("\n",sample)

avg = sum(local_accuracy)/len(local_accuracy)
print("\n Average accuracy :")
write_to_file("\n Average accuracy :" + str(avg)+"\n",sample)
print(avg)

sample.close()