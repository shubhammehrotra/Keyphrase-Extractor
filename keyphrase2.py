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

    return [cand for cand in candidates
            if cand not in stop_words and not all(char in punct for char in cand)]


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

    return candidates

def score_keyphrases_by_tfidf(texts, candidates='chunks'):
    import gensim, nltk
    
    # extract candidates from each text in texts, either chunks or words
    if candidates == 'chunks':
        boc_texts = [extract_candidate_chunks(text) for text in texts]
    elif candidates == 'words':
        boc_texts = [extract_candidate_words(text) for text in texts]
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
    candidates = extract_candidate_chunks(text)
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
    outputFilePath = "C:/Python34/Keyphrase/SentenceWiseKeyPhrase/"
    #fileName = "C:/Users/shubham_15294/Downloads/reviews_ElectronicsSplitTagged.csv"
    fileRead = open(inputFilePath + file + ".csv","r")
    inputSentence = ''
    with fileRead as tsvfile:
        tsvreader = csv.reader(tsvfile)
        id = 0
        for lines in tsvreader:
            id = id + 1
            tag = int(lines[2])
            #if  (tag == 1):
            text = lines[1]
            #inputSentence = inputSentence + ' ' + text
            output = score_keyphrases_by_textrank(text)
            fileWrite(text, output, file, outputFilePath)
    #return inputSentence
        
def fileWrite(text, output, file, outputFilePath):
    import csv, os
    csvFile = csv.writer(open(outputFilePath + file + "Output.csv", "a",newline=''))
    csvFile.writerow([text, output])
            


#output1 = extract_candidate_chunks("Despite wide applicability and much research, keyphrase extraction suffers from poor performance relative to many other core NLP tasks, partly because there’s no objectively “correct” set of keyphrases for a given document. While human-labeled keyphrases are generally considered to be the gold standard, humans disagree about what that standard is! As a general rule of thumb, keyphrases should be relevant to one or more of a document’s major topics, and the set of keyphrases describing a document should provide good coverage of all major topics. (They should also be understandable and grammatical, of course.) The fundamental difficulty lies in determining which keyphrases are the most relevant and provide the best coverage. As described in Automatic Keyphrase Extraction: A Survey of the State of the Art, several factors contribute to this difficulty, including document length, structural inconsistency, changes in topic, and (a lack of) correlations between topics.") 
#print(output1)
#output2 = extract_candidate_words("Despite wide applicability and much research, keyphrase extraction suffers from poor performance relative to many other core NLP tasks, partly because there’s no objectively “correct” set of keyphrases for a given document. While human-labeled keyphrases are generally considered to be the gold standard, humans disagree about what that standard is! As a general rule of thumb, keyphrases should be relevant to one or more of a document’s major topics, and the set of keyphrases describing a document should provide good coverage of all major topics. (They should also be understandable and grammatical, of course.) The fundamental difficulty lies in determining which keyphrases are the most relevant and provide the best coverage. As described in Automatic Keyphrase Extraction: A Survey of the State of the Art, several factors contribute to this difficulty, including document length, structural inconsistency, changes in topic, and (a lack of) correlations between topics.")
#print('Output for method 2 '+ output2)

#output3 = score_keyphrases_by_tfidf("Despite wide applicability and much research, keyphrase extraction suffers from poor performance relative to many other core NLP tasks, partly because there’s no objectively “correct” set of keyphrases for a given document. While human-labeled keyphrases are generally considered to be the gold standard, humans disagree about what that standard is! As a general rule of thumb, keyphrases should be relevant to one or more of a document’s major topics, and the set of keyphrases describing a document should provide good coverage of all major topics. (They should also be understandable and grammatical, of course.) The fundamental difficulty lies in determining which keyphrases are the most relevant and provide the best coverage. As described in Automatic Keyphrase Extraction: A Survey of the State of the Art, several factors contribute to this difficulty, including document length, structural inconsistency, changes in topic, and (a lack of) correlations between topics.")
#print('Output for method 3 '+ output3)
inputFilePath = "C:/Users/shubham_15294/Downloads/"
outputFilePath = "C:/Python34/Keyphrase/"
fileName = ["reviews_Home_and_KitchenTagged","reviews_BabySplitTagged","reviews_ElectronicsSplitTagged"]
#inputFormat(fileName,filePath)
for file in fileName:
    inputFormat(file, inputFilePath)
    #output = score_keyphrases_by_textrank(inputSentence)
    #output = extract_candidate_chunks(inputSentence)
    #fileWrite(output, file, outputFilePath)
    #for key in output:
    #    print (key[0])
    print(file + ' Done')
    
print ('Done')
