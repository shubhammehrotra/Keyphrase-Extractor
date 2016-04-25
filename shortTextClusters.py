from __future__ import division
from sklearn.cluster import KMeans
from numbers import Number
import sys, codecs, numpy
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.cluster import AgglomerativeClustering
import csv

#sentence clustering using glove word embeddings and k-means clustering
# sentence vectors are obtained simply by adding word vectors
class autovivify_list(dict):
        '''Pickleable class to replicate the functionality of collections.defaultdict'''
        def __missing__(self, key):
                value = self[key] = []
                return value

        def __add__(self, x):
                '''Override addition for numeric types when self is empty'''
                if not self and isinstance(x, Number):
                        return x
                raise ValueError

        def __sub__(self, x):
                '''Also provide subtraction method'''
                if not self and isinstance(x, Number):
                        return -1 * x
                raise ValueError

def build_word_vector_matrix(vector_file, n_words):
        '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
        numpy_arrays = []
        labels_array = []
        with codecs.open(vector_file, 'r', 'utf-8') as f:
                for c, r in enumerate(f):
                        sr = r.split()
                        labels_array.append(sr[0])
                        numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )

                        if c == n_words:
                                return numpy.array( numpy_arrays ), labels
                                _array

        return numpy.array(numpy_arrays), labels_array

def find_word_clusters(labels_array, cluster_labels):
        '''Read the labels array and clusters label and return the set of words in each cluster'''
        cluster_to_words = autovivify_list()
        for c, i in enumerate(cluster_labels):
                cluster_to_words[ i ].append( labels_array[c] )
        return cluster_to_words

def find_sentence_clusters(sentence_label_array, cluster_labels):
        cluster_to_sentences = autovivify_list()
        for c, i in enumerate(cluster_labels):
                cluster_to_sentences[i].append(sentence_label_array[c])
        return cluster_to_sentences

#builds sentence vectors by summing up all the word vector for a sentence
def build_sentence_vector(sentence_file_path, vector_file, n_words):

        numpy.arrays, labels_array =  build_word_vector_matrix(vector_file, n_words)
        filereader = open(sentence_file_path,"r", encoding = 'utf-8', errors = 'ignore' )
        sentence_label_array = []
        vectorSum = numpy.empty([300,],dtype = float)
        sentence_arrays = []
        n = 0
        for sentence in filereader.readlines():
                tokens = word_tokenize(sentence, language = 'english')
                n = n+1
                print(n)
                sentence_label_array.append(sentence)
                for token in tokens:
                    token = token.lower()
                    if(token in labels_array):
                            index = labels_array.index(token)
                            vectorSum = numpy.add(vectorSum, numpy.arrays[index])
                sentence_arrays.append(vectorSum)


        return numpy.array(sentence_arrays), sentence_label_array

#vectors of only content words
def build_topic_sentence_vector(sentence_file_path, vector_file, n_words):

        numpy.arrays, labels_array =  build_word_vector_matrix(vector_file, n_words)
        filereader = open(sentence_file_path,"r", encoding = 'utf-8', errors = 'ignore')
        sentence_label_array = []
        vectorSum = numpy.empty([300,],dtype = float) #change the dimensions here in case you change the glove vector size
        sentence_arrays = []
        n = 0
        for sentence in filereader.readlines():
                tokens = word_tokenize(sentence, language = 'english')
                taggedSent = pos_tag(tokens)
                n = n+1
                print(n)
                sentence_label_array.append(sentence)
                for word, tag in taggedSent:
                    if(tag not in ['NN','NNP','NNPS','NNS','VB','VBG','VBN','VBP','VBZ']):
                        token = word.lower()
                        if(token in labels_array):
                            index = labels_array.index(token)
                            vectorSum = numpy.add(vectorSum, numpy.arrays[index])
                sentence_arrays.append(vectorSum)


        return numpy.array(sentence_arrays), sentence_label_array



if __name__ == "__main__":
        print ('Running');
        fileName = ["Tweets"]
        input_vector_file = "C:/Users/shubham_15294/Downloads/glove.6B.300d.txt" # sys.arg[1] The Glove file to analyze (e.g. glove.6B.300d.txt)
        #sentence_file_path = "dataset/abortionSplit.csv"
        for file in fileName:
            sentence_file_path = "dataset/" + file + ".csv"
            #out_directory  = "output/hotelBerlin191437"
            #out_directory  = "output/debateAbortion"
            out_directory  = "output/"
            n_words           =   6000000000   # int(sys.argv[2]) The number of lines to read from the input file
            reduction_factor  =   0.000000002        #float(sys.argv[3]) The desired amount of dimension reduction
            clusters_to_make  = int( n_words * reduction_factor ) # The number of clusters to make
            df, sentence_label_array  = build_sentence_vector(sentence_file_path,input_vector_file, n_words)
    
    
            kmeans_model      = KMeans(init='k-means++', n_clusters=clusters_to_make, n_init=10)
            kmeans_model.fit(df)
    
            cluster_labels    = kmeans_model.labels_
            cluster_inertia   = kmeans_model.inertia_
            cluster_to_sentences  = find_sentence_clusters(sentence_label_array, cluster_labels)
    
    
    
            cluster_count = 1
            for c in cluster_to_sentences:
                    #file_next = open(out_directory + "/cluster" + str(cluster_count) + ".txt", "w+")
                                   
                    print(cluster_to_sentences[c])
                    buffer = ""
                    for i in cluster_to_sentences[c]:
                            buffer = buffer + i
                    #csvFile = csv.writer(open(out_directory + file + "cluster" + str(cluster_count) + ".csv",'w'))        
                    f = open(out_directory + file + "/ Topic " + str(cluster_count) + ".txt",'w')        
                    f.write(buffer)
                    #file_next.write(buffer)
                    #csvFile.writerow(buffer)
                    print("\n")
                    cluster_count = cluster_count+1
