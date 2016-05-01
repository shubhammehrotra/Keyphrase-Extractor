from __future__ import division
from sklearn.cluster import KMeans
from numbers import Number
import sys, codecs, numpy
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.cluster import AgglomerativeClustering
import csv
import os

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
                #print(n)
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
                #print(n)
                sentence_label_array.append(sentence)
                for word, tag in taggedSent:
                    if(tag not in ['NN','NNP','NNPS','NNS','VB','VBG','VBN','VBP','VBZ']):
                        token = word.lower()
                        if(token in labels_array):
                            index = labels_array.index(token)
                            vectorSum = numpy.add(vectorSum, numpy.arrays[index])
                sentence_arrays.append(vectorSum)


        return numpy.array(sentence_arrays), sentence_label_array

def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):
    import itertools, nltk, string
    
    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda word__pos__chunk: word__pos__chunk[2] != 'O') if key]
    #print(cand)
    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]
    #print(candidates)
    return candidates

def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    #import gensim, nltk
    
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
        #print (boc_texts)
    # make gensim dictionary and corpus
    dictionary = gensim.corpora.Dictionary(boc_texts)
    corpus = [dictionary.doc2bow(boc_text) for boc_text in boc_texts]
    # transform corpus with tf*idf model
    tfidf = gensim.models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    return corpus_tfidf, dictionary

def score_keyphrases_by_textrank(text, n_keywords=0.05):
    from itertools import takewhile, tee
    import networkx, nltk
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        a = ''
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            #print(kp_words)
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    #print (kp_words)
    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)

def extract_candidate_features(candidates, doc_text, doc_excerpt, doc_title):
    import collections, math, nltk, re
    
    candidate_scores = collections.OrderedDict()
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(doc_text)
                                          for word in nltk.word_tokenize(sent))
    
    for candidate in candidates:
        
        pattern = re.compile(r'\b'+re.escape(candidate)+r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # frequency-based
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(doc_text))
        # count could be 0 for multiple reasons; shit happens in a simplified example
        if not cand_doc_count:
            print ('**WARNING:', candidate, 'not found!')
            continue
    
        # statistical
        candidate_words = candidate.split()
        max_word_length = max(len(w) for w in candidate_words)
        term_length = len(candidate_words)
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        # positional
        # found in title, key excerpt
        in_title = 1 if pattern.search(doc_title) else 0
        in_excerpt = 1 if pattern.search(doc_excerpt) else 0
        # first/last position, difference between them (spread)
        doc_text_length = float(len(doc_text))
        first_match = pattern.search(doc_text)
        abs_first_occurrence = first_match.start() / doc_text_length
        if cand_doc_count == 1:
            spread = 0.0
            abs_last_occurrence = abs_first_occurrence
        else:
            for last_match in pattern.finditer(doc_text):
                pass
            abs_last_occurrence = last_match.start() / doc_text_length
            spread = abs_last_occurrence - abs_first_occurrence

        candidate_scores[candidate] = {'term_count': cand_doc_count,
                                       'term_length': term_length, 'max_word_length': max_word_length,
                                       'spread': spread, 'lexical_cohesion': lexical_cohesion,
                                       'in_excerpt': in_excerpt, 'in_title': in_title,
                                       'abs_first_occurrence': abs_first_occurrence,
                                       'abs_last_occurrence': abs_last_occurrence}

    return candidate_scores

def inputFormat(file, inputFilePath):
    import csv
    #fileName = "C:/Users/shubham_15294/Downloads/reviews_ElectronicsSplitTagged.csv"
    fileRead = open(inputFilePath + file + ".csv","r")
    inputSentence = ''
    with fileRead as tsvfile:
        tsvreader = csv.reader(tsvfile)
        id = 0
        for lines in tsvreader:
            id = id + 1
            #tag = int(lines[2])
            #if  (tag == 1):
            text = lines[1]
            inputSentence = inputSentence + ' ' + text
            #output = score_keyphrases_by_textrank(inputSentence)
            #fileWrite(output,fileName, filePath)
    return inputSentence
        
def fileWrite(output, file, outputFilePath):
    import csv, os
    try:
        os.remove(outputFilePath + file + ".csv")
    except OSError:
        pass 
    if not os.path.exists(outputFilePath):
            os.makedirs(outputFilePath)
    csvFile = csv.writer(open(outputFilePath + file + ".csv", "w",newline=''))
    for out in output:
        csvFile.writerow([out[0]])
            
def sentenceOrganize(fileName):
    import os
    import csv
    for v in fileName:
        fileDirectory = 'output/' + v + '/'
        inputFilePath = "Suggestions Classified/"
        fileName = v
        for fn in os.listdir(fileDirectory):
            f = open(fileDirectory + fn,'r')
            count = 0
            s = fn.strip(".txt")
            if not os.path.exists('output/Sentences/' + v + '/'):
                os.makedirs('output/Sentences/' + v + '/')
            newFile = open('output/Sentences/' + v + '/' + s + ' Sentence.txt', 'w')
            for text in f:
                count = count + 1
                fileRead = open(inputFilePath + fileName + ".csv","r")
                if (count != -1):
                    cnt = 0
                    with fileRead as tsvfile:
                        tsvreader = csv.reader(tsvfile)
                        for lines in tsvreader:
                            
                            tag = lines[2]
                            if (tag == "1"):                            
                                sentence = lines[1]
                                a = str(text).strip()
                                if a in sentence:
                                    cnt = cnt + 1
                                    newFile.write(sentence + '\n' + '\n')
        print(v + ' Sentence Organized!')                            



#inputFormat(fileName,filePath)


if __name__ == "__main__":
        print ('Running');
        fileName = ["Baby", "Tweets", "Electronics", "Healthcare"]
        inputFilePath = "Suggestions Classified/"
        outputFilePath = "dataset/"
        input_vector_file = "C:/Users/shubham_15294/Downloads/glove.6B.300d.txt" # sys.arg[1] The Glove file to analyze (e.g. glove.6B.300d.txt)
        for file in fileName:
            inputSentence = inputFormat(file, inputFilePath)
            output = score_keyphrases_by_textrank(inputSentence)
            fileWrite(output, file, outputFilePath)
            print(file + ' Done Keyphrase!')    
        print ('Keyphrase Extraction Done!')
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
                                   
                    #print(cluster_to_sentences[c])
                    buffer = ""
                    for i in cluster_to_sentences[c]:
                            buffer = buffer + i
                    if not os.path.exists(out_directory + file + "/"):
                        os.makedirs(out_directory + file + "/")
                    #csvFile = csv.writer(open(out_directory + file + "cluster" + str(cluster_count) + ".csv",'w'))        
                    f = open(out_directory + file + "/ Topic " + str(cluster_count) + ".txt",'w')        
                    f.write(buffer)
                    #file_next.write(buffer)
                    #csvFile.writerow(buffer)
                    #print("\n")
                    cluster_count = cluster_count+1
            print (file + ' Done Clustering!')
        print ('Clustering Done!')
        sentenceOrganize(fileName)
        print('Sentences Organized!')