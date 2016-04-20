import os
import csv
variable = ['Baby', 'Electronics']
for v in variable:
    fileDirectory = 'C:/Python34/Keyphrase/output/' + v + '/'
    inputFilePath = "C:/Users/shubham_15294/Downloads/"
    fileName = "reviews_" + v + "SplitTagged"
    for fn in os.listdir(fileDirectory):
        f = open(fileDirectory + fn,'r')
        count = 0
        newFile = open('C:/Python34/Keyphrase/output/Sentences/' + v + '/' + fn + 'check.txt', 'w')
        for text in f:
            count = count + 1
            fileRead = open(inputFilePath + fileName + ".csv","r")
            if (count < 10):
                with fileRead as tsvfile:
                    tsvreader = csv.reader(tsvfile)
                    for lines in tsvreader:
                        tag = int(lines[2])
                        if (tag == 1):
                            sentence = lines[1]
                            a = str(text).strip()
                            if a in sentence:
                                print(fn + ' ' + a + '     ' + sentence)
                                newFile.write(sentence + '\n')
print('Done')
