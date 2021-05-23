from pdp_utils import load_problem
import sys

#pdp_utils has to be in the same directory as these files

from algorithms import *

def run_Alg(opt, prob, raw):
        if(opt < 1):
                print('I am sorry, local search does not work right now')
                #local_Search(prob)
        elif(opt < 2):
                print('I am sorry, simulated annealing does not work right now')
                #simulated_Annealing(prob)
        else:
                adaptive_large_neighborhood(prob, raw)


#To run it -> python main.py data_sheet [0 (L.S) / 1 (S.A) / 2 (A.N.L.S)] (Sth-> raw)
#Mode 0 and 1 are not able to be used after implementing mode 2
prob = load_problem(sys.argv[1])
raw = False

#Raw mode is not in use now, it was used to make some excell, it doesn't work now
if(len(sys.argv) > 3):
        raw=True
        f = open("results/FinalRun2.txt", "a")
        f.write('-------------------------------ANLS-------------------------------')
        f.write('\nData set for the solution -> ' + sys.argv[1])
        f.close()        
else:
        print('Data set for the solution -> ' + sys.argv[1])

run_Alg(int(sys.argv[2]), prob, raw)