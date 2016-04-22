import os
import csv
variable = ['Electronics', 'Baby']
for v in variable:
    fileDirectory = 'C:/Python34/Keyphrase/output/' + v + '/'
    inputFilePath = "C:/Users/shubham_15294/Downloads/"
    fileName = v
    for fn in os.listdir(fileDirectory):
        f = open(fileDirectory + fn,'r')
       
        count = 0
        s = fn.strip(".txt")
        newFile = open('C:/Python34/Keyphrase/output/Sentences/' + v + '/' + s + ' Sentence.txt', 'w')
        for text in f:
            count = count + 1
            fileRead = open(inputFilePath + fileName + ".csv","r")
            if (count != -1):
                cnt = 0
                with fileRead as tsvfile:
                    tsvreader = csv.reader(tsvfile)
                    for lines in tsvreader:
                        
                        tag = int(lines[2])
                        if (tag == 1):
                            
                            sentence = lines[1]
                            a = str(text).strip()
                            if a in sentence:
                                cnt = cnt + 1
                                newFile.write(sentence + '\n' + '\n')
    print(v + ' Done !')
print('Done')
