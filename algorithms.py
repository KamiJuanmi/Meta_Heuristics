from pdp_utils import *
from time import time
import random
import math

PARAM_TEMP = 0.2
NUMBER_EXE = 5

from operators import *
from utilities import *

'''
        They don't return anything, they just print all the results
'''

#They don't work because I changed the operators!!! And I don't use now the final local search
def final_local_Search(prob, copy_current_sol):
        

        current_sol = copy_current_sol
        best_sol = current_sol

        best_cost = cost_function(best_sol,prob)

        for i in range(1, 2000):
                probab = random.random()

                new_sol, feasible = apply_local_search(prob, current_sol, probab)
                
                cost_curr_sol = cost_function(new_sol,prob)

                if(feasible and cost_curr_sol < best_cost):
                        best_cost = cost_curr_sol
                        best_sol = new_sol
        return best_sol

def local_Search(prob):
        elapsed_time = 0

        best_Of_The_Best = load_Initial_Solution(prob)

        cost_initial = cost_function(best_Of_The_Best,prob)

        best_cost_OTB = cost_initial

        probability_1 = 1/3
        probability_2 = 1/3

        cost_average = 0

        for i in range (0,10):
                start_time = time()

                current_sol = load_Initial_Solution(prob)
                best_sol = current_sol

                best_cost = cost_function(best_sol,prob)

                for i in range(1, 10000):
                        probab = random.random()

                        if probab < probability_1:
                                current_sol = k_exchange(prob, current_sol, 2)
                        else: 
                                if (probab < probability_1 + probability_2):
                                        current_sol = k_exchange(prob, current_sol, 3)
                                else:
                                        current_sol = one_reinsert(prob, current_sol)
                        
                        feasibility,c = feasibility_check(current_sol,prob)

                        cost_curr_sol = cost_function(current_sol,prob)

                        if(feasibility and cost_curr_sol < best_cost):
                                best_cost = cost_curr_sol
                                best_sol = current_sol

                elapsed_time += time() - start_time

                cost_average += best_cost

                if(best_cost < best_cost_OTB):
                        best_cost_OTB = best_cost
                        best_Of_The_Best = best_sol

        elapsed_time /= 10
        cost_average /= 10

        print_all_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time)
        
def simulated_Annealing(prob):
        elapsed_time = 0

        best_Of_The_Best = load_Initial_Solution(prob)

        cost_initial = cost_function(best_Of_The_Best,prob)

        best_cost_OTB = cost_initial

        probability_1 = 0.33
        probability_2 = 0.33

        cost_average = 0

        for i in range (0,10):
                start_time = time()
                        
                best_sol = load_Initial_Solution(prob)

                best_cost = cost_function(best_sol,prob)

                cost_inc = best_cost

                incumbent = best_sol 

                temperature = 100
                alfa = 0.98

                for j in range(1,10000):
                        rand = random.random()
                        
                        if rand < probability_1:
                                new_solution = one_reinsert_best_car(prob, incumbent) 
                        else:
                                if(rand < probability_1 + probability_2):
                                        new_solution = get_car_full(prob, incumbent)
                                else:
                                        new_solution = k_exchange(prob, incumbent, 2)
                                              
                        cost_new = cost_function(new_solution, prob)

                        delta_energy = cost_new - cost_inc
                        
                        feasibility,c = feasibility_check(new_solution, prob)

                        if(feasibility):
                                if(delta_energy < 0):
                                        incumbent = new_solution
                                        cost_inc = cost_new
                                        
                                        if(cost_inc < best_cost):
                                                best_cost = cost_inc
                                                best_sol = incumbent
                                else:
                                        rand2 = random.random()
                                        prob_to_worst = math.exp(-delta_energy/temperature)
                                        if(rand2 < prob_to_worst):
                                                incumbent = new_solution
                                                cost_inc = cost_new
                                temperature *= alfa


                elapsed_time += time() - start_time

                cost_average += best_cost

                if(best_cost < best_cost_OTB):
                        best_cost_OTB = best_cost
                        best_Of_The_Best = best_sol

        
        
        elapsed_time /= 10
        cost_average /= 10

        print_all_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time)


#The one that is in use
def adaptive_large_neighborhood(prob, raw):
        elapsed_time = 0

        NUMBER_SECONDS = limit_seconds(prob)
        NUMBER_ITE = avg_ite(prob)

        best_Of_The_Best = load_Initial_Solution(prob)

        cost_initial = cost_function(best_Of_The_Best,prob)

        best_cost_OTB = cost_initial

        cost_average = 0

        for i in range(0, NUMBER_EXE):
                start_time = time()

                current_sol = load_Initial_Solution(prob)
                best_sol = current_sol

                best_cost = cost_function(best_sol,prob)

                it_since_best = 0
                end_escape = False
                it_best = 0

                mem = create_memory()
                j = 0

                parameters = generate_init_parameters()           
                points = parameters[NUMBER_OPERATORS:2*NUMBER_OPERATORS]
                times_chosen = parameters[2*NUMBER_OPERATORS:]
                parameters = parameters[:NUMBER_OPERATORS]

                while(time() % start_time < NUMBER_SECONDS):
                        j += 1
                        
                        new_current_sol = current_sol
                        if(it_since_best > 500 and (not end_escape)):
                                
                                new_current_sol = current_sol
                                new_current_sol, aux_feas = apply_escape_alg(prob, new_current_sol)
                                D = PARAM_TEMP * ((NUMBER_ITE - j) / NUMBER_ITE) * best_cost
                                if(aux_feas):
                                        new_cost = cost_function(new_current_sol, prob)

                                        if(new_cost < cost_function(current_sol, prob)):
                                                current_sol = new_current_sol
                                        elif(new_cost < D + best_cost):
                                                current_sol = new_current_sol

                                        if(new_cost < best_cost):
                                                best_sol = new_current_sol
                                                best_cost = new_cost
                                                it_best = j
                                                end_escape = True
                                it_since_best += 1
                                if(it_since_best > 520):
                                        end_escape = True
                                        it_since_best = 0
                                continue
                        psi = 0
                        end_escape = False
                        
                        selected = select_heuristic(parameters)

                        new_current_sol, aux_feas = apply_heuristic(prob, new_current_sol, selected)
                        D = PARAM_TEMP * ((NUMBER_ITE - j) / NUMBER_ITE) * best_cost
                        it_since_best += 1

                        if(aux_feas):
                                new_cost = cost_function(new_current_sol, prob)
                                
                                mem, is_new = check_and_update(mem, new_cost)

                                if(is_new):
                                        psi = 1

                                if(new_cost < cost_function(current_sol, prob)):
                                        psi = 2
                                        current_sol = new_current_sol
                                elif(new_cost < D + best_cost):
                                        current_sol = new_current_sol

                                if(new_cost < best_cost):
                                        psi = 4
                                        best_sol = new_current_sol
                                        best_cost = new_cost
                                        it_best = j
                                        it_since_best = 0


                        points[selected] += psi
                        times_chosen[selected] += 1

                        if(j % 250 == 0):
                                parameters = update_parameters(parameters, points, times_chosen)
                
                new_time = time() - start_time
                elapsed_time += time() - start_time

                cost_average += best_cost

                if(best_cost < best_cost_OTB):
                        best_cost_OTB = best_cost
                        best_Of_The_Best = best_sol
        
        elapsed_time /= NUMBER_EXE
        cost_average /= NUMBER_EXE
        
        if(raw):
                write_all_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time)
                print_raw_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time)
        else:
                print_all_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time)
