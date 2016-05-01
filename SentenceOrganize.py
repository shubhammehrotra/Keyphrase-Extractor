import os
import csv
fileName = ["Baby", "Tweets", "Electronics", "Healthcare"]
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