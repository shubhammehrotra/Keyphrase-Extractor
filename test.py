def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS','VB','VBD','VBG','VBN','VBP','VBZ'])):
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

inputFilePath = "Suggestions Classified/"
outputFilePath = "dataset/"
fileName = ["Tweets"]
#inputFormat(fileName,filePath)
for file in fileName:
    inputSentence = inputFormat(file, inputFilePath)
    #print(inputSentence)
    print('Hello')
    output = score_keyphrases_by_textrank(inputSentence)
    #output = extract_candidate_chunks(inputSentence)
    #print(output)
    fileWrite(output, file, outputFilePath)
    #for key in output:
    #    print (key[0])
    print(file + ' Done')
