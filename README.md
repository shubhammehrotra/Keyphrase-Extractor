Keyphrase Extraction techniques adopted from : http://bdewilde.github.io/blog/2014/09/23/intro-to-automatic-keyphrase-extraction/

Run allTogether.py

1. Keyphrases are extracted using the TextRank algorithm, 
2. Extracted keyphrases are grouped using k means clustering using GloVe word embeddings.
3. All sentences containing the clustered keyphrases are grouped together according to their cluster and displayed on the UseCase section.

Input files are in Suggestions Classified, new files can be added and their name appended in the list fileName, provided they follow the same input format as shown in the given files.

Output files are stored in Output/Sentences/fileName

Change path of GloVe file on line 343, and change dimensions on line 67 and line 91  according to the dimesion of the GloVe file.
