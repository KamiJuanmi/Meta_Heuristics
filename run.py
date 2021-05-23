import os
import glob

#Small script to run all the data sets and move the results to a file

data_sets = glob.glob("data/*txt") #update data route in case its moved
for i in data_sets:
        #update final file name 
        os.system("echo -------------------------------Run------------------------------- >> results/ExamRun2.txt")
        os.system("python main.py " + i + " 2 >> results/ExamRun2.txt")

        

