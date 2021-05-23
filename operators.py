from pdp_utils import *
import random
import numpy as np

from utilities import *

'''
        All the operators return a new solution after doing their changes and a boolean variable feasibility
        Feasibility is obtained by checking only the cars that have been modified during the use of the operator so there is no need 
        to use the feasibility_check for the complete function. This is a huge improvement in perfomance
'''

def k_exchange(prob, copy_current_sol, k):
        if k > prob.get('n_calls'):
                return 0, False
        
        current_sol = copy_current_sol[:]

        veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_removing_empties(prob, current_sol)

        chosen_calls = []
        chosen_vehs = []

        for i in range(0, k):
                valid_chosen_call = False

                while(not valid_chosen_call):
                        
                        chosen_veh = random.choice(veh)

                        chosen_call = choose_a_random_call(current_sol, veh, n_veh, index_changing_car, chosen_veh)
                        
                        if chosen_call not in chosen_calls:
                                valid_chosen_call = True
                
                remove_one_call(chosen_veh, veh, number_calls_car)

                chosen_calls.append(chosen_call)
                chosen_vehs.append(chosen_veh)

        random.shuffle(chosen_calls)

        previous_index = []
        previous_index.append(current_sol.index(chosen_calls[0]))
        previous_index.append(current_sol.index(chosen_calls[0], previous_index[0] + 1))

        current_sol[previous_index[0]] = chosen_calls[1]
        current_sol[previous_index[1]] = chosen_calls[1]


        for i in range(1, len(chosen_calls)):
                index_replacer = 0 if i == (len(chosen_calls) - 1) else i+1

                to_Replace_index = []

                aux_index = -1

                while(len(to_Replace_index) < 2):

                        new_index = current_sol.index(chosen_calls[i], aux_index + 1)

                        if new_index not in previous_index:
                                to_Replace_index.append(new_index)
                                current_sol[new_index] = chosen_calls[index_replacer]

                        aux_index = new_index
                
                previous_index = to_Replace_index
        for i in range(k):
                feasible, c = car_feasibility_check(current_sol, prob, chosen_vehs[i])
                if(not feasible):
                        break
        return current_sol, feasible


def one_reinsert(prob, copy_current_sol):
        current_sol = copy_current_sol[:]

        veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_removing_empties(prob, current_sol)

        chosen_veh = random.choice(veh)

        chosen_call = choose_a_random_call(current_sol, veh, n_veh, index_changing_car, chosen_veh)

        veh = load_vehicles(prob)

        current_sol.remove(chosen_call)
        current_sol.remove(chosen_call)

        index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)

        current_sol, chosen_veh = insert_call_in_random_veh(chosen_call, current_sol, veh, n_veh, index_changing_car)

        feasible, c = car_feasibility_check(current_sol, prob, chosen_veh)

        return current_sol, feasible


def get_car_full(prob, copy_current_sol):
        '''
                Tries to select a car, then check if there is some call that is able to transport but hasn't been picked and insert it in the car
        '''
        current_sol = copy_current_sol[:]

        veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_not_removing_empties(prob, current_sol)

        chosen_veh = random.choice(veh[0:n_veh-1]) #Is not possible to choose the dummy

        not_picked = check_not_picked(prob, chosen_veh, get_calls_in_car(current_sol, veh, n_veh, index_changing_car, chosen_veh))

        if(len(not_picked) == 0):
                return current_sol, False

        chosen_call = random.choice(not_picked)
        current_sol.remove(chosen_call)
        current_sol.remove(chosen_call)

        index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)

        current_sol = insert_call_in_vehicle(chosen_call, chosen_veh, current_sol, veh, n_veh, index_changing_car)

        feasible, c = car_feasibility_check(current_sol, prob, chosen_veh)

        return current_sol, feasible


def one_reinsert_best_car(prob, copy_current_sol):
        '''
                Tries to insert a random call in the best car (function explained in utilities)
        '''
        while(True):
                current_sol = copy_current_sol[:]

                veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_removing_empties(prob, current_sol)

                chosen_veh = random.choice(veh)

                chosen_call = choose_a_random_call(current_sol, veh, n_veh, index_changing_car, chosen_veh)
                veh = load_vehicles(prob)

                current_sol.remove(chosen_call)
                current_sol.remove(chosen_call)

                index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)

                current_sol, best_car = insert_best_car(prob, chosen_call, current_sol, veh, n_veh, index_changing_car)

                feasibility, c = car_feasibility_check(current_sol, prob, best_car)

                if(feasibility):
                        break

        return current_sol, feasibility


def dummy_back_exchange(prob, copy_current_sol):
        '''
                1. Moves a random call to the dummy
                2. Chooses a random call and apply one_reinsert or one_reinsert_best_car based on the size of the instance
                3. Makes a 2_exchange (totally random too)
        '''
        current_sol = choose_random_move_to_dummy(prob, copy_current_sol)
        
        aux = current_sol
        contador = 0

        k = 50
        bigProblem = False

        if(prob['n_calls'] > 30):        
                bigProblem = True
                if(prob['n_calls'] > 50):
                        k = 20
              
        
        while (True):
                contador +=1
                if(bigProblem):
                        current_sol, feasibility = one_reinsert_best_car(prob, copy_current_sol)
                else:
                        current_sol, feasibility = one_reinsert(prob, current_sol)
                if(contador > k):
                        break
                if(feasibility):
                        break
                else:
                        current_sol = aux

        if(not feasibility):
                return current_sol, feasibility
                
        aux = current_sol
        contador = 0

        while (True):
                contador +=1
                current_sol, feasibility = k_exchange(prob, current_sol, 2)
                if(contador > k):
                        break
                if(feasibility):
                        break
                else:
                        current_sol = aux
        return current_sol, feasibility


def random_removal(prob, copy_current_sol, k):   
        '''
                1. Selects some calls randomly
                2. Reinsert these calls using one_Reinsert or one_reinsert_best_car based on the size of the instance
        '''

        new_sol = copy_current_sol[:]

        chosen_calls = []

        for i in range(k):
                veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_removing_empties(prob, new_sol)

                chosen_veh = random.choice(veh)

                chosen_calls.append(choose_a_random_call(new_sol, veh, n_veh, index_changing_car, chosen_veh))

                new_sol.remove(chosen_calls[i])
                new_sol.remove(chosen_calls[i])

        bigProblem = False

        if(prob['n_calls'] > 30):        
                bigProblem = True

        feasibility = True

        for i in range(k):

                veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_not_removing_empties(prob, new_sol)
        
                if(bigProblem):
                        new_sol, car = insert_best_car(prob, chosen_calls[i], new_sol, veh, n_veh, index_changing_car)
                else:
                        new_sol, car = insert_call_in_random_veh(chosen_calls[i], new_sol, veh, n_veh, index_changing_car)

                feasibility, c = car_feasibility_check(new_sol, prob, car)
                
                if(not feasibility):
                        break

        return new_sol, feasibility


def reinsert_best_position(prob, copy_current_sol, k):
        '''
                1. Selects the calls that have the biggest cost
                2. While there are calls remaining:
                        2.1 Find the best position for all the calls and the cost of this
                        2.2 Choose the less cost among all the calls and insert it in that position
        '''

        current_sol = copy_current_sol[:]
        feasib = True
        chosen_calls = []

        
        for i in range(k):
                veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_removing_empties(prob, current_sol)

                chosen_veh = random.choice(veh)

                chosen_call = choose_call_most_cost_in_car(prob, current_sol, veh, n_veh, index_changing_car, chosen_veh)
                
                current_sol.remove(chosen_call)
                current_sol.remove(chosen_call)
                chosen_calls.append(chosen_call)
        
        for j in range(k):
                if(len(chosen_calls) < 1):
                        break
                costs = []

                data = []

                veh, n_veh, index_changing_car, number_calls_car = create_update_utilities_removing_empties(prob, current_sol)

                for i in chosen_calls:
                        best_car, best_pos1, best_pos2, best_cost = get_best_car_best_position(prob, i, current_sol, veh, n_veh, index_changing_car)
                        data.append([i, best_car, best_pos1, best_pos2])
                        costs.append(best_cost)
                
                costs_array = np.array(costs)

                order = np.argsort(costs_array)

                current_sol = insert_call_in_veh_in_position(data[order[0]][0], data[order[0]][1], current_sol, veh, n_veh, index_changing_car, data[order[0]][2])
                current_sol = insert_call_in_veh_in_position(data[order[0]][0], data[order[0]][1], current_sol, veh, n_veh, index_changing_car, data[order[0]][3])
                
                feasib, c = car_feasibility_check(current_sol, prob, data[order[0]][1])
                if(not feasib):
                        break
                chosen_calls.remove(data[order[0]][0])

        return current_sol, feasib


def apply_heuristic(prob, copy_current_sol, chosen_heuristic):
        '''
                return: solution, feasible
        '''
        how_many = random.randint(1,3)

        if(chosen_heuristic == 0):
                return reinsert_best_position(prob, copy_current_sol, how_many)
        elif(chosen_heuristic == 1):
                return dummy_back_exchange(prob, copy_current_sol)
        elif(chosen_heuristic == 2):
                return random_removal(prob, copy_current_sol, how_many)
        elif(chosen_heuristic == 3):
                return get_car_full(prob, copy_current_sol)
        else:
                return one_reinsert_best_car(prob, copy_current_sol)

def apply_local_search(prob, copy_current_sol, probab):
        '''
                return solution, feasible
                Used to do a final local search after the iterations, now is not in used
        '''
        bigProblem = False
        probability_1 = probability_2 = 0.33

        if(prob['n_calls'] > 50):        
                bigProblem = True

        if probab < probability_1:
                        return k_exchange(prob, copy_current_sol, 2)
        else: 
                if (probab < probability_1 + probability_2):
                        if(bigProblem):
                                return dummy_back_exchange(prob, copy_current_sol)
                        else:
                                return reinsert_best_position(prob, copy_current_sol, 2)
                else:
                        return random_removal(prob, copy_current_sol, 2)

def apply_escape_alg(prob, copy_current_sol):
        '''
                return solution, feasible
                Used to apply an escape algorithm
        '''
        probab = random.random()

        how_many = random.randint(1,3)

        if probab < 0.5:
                return random_removal(prob, copy_current_sol, 3)
        else: 
                return dummy_back_exchange(prob, copy_current_sol)