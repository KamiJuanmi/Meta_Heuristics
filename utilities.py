from pdp_utils import *
import random
import numpy as np

NUMBER_OPERATORS = 5

def load_Initial_Solution(prob):
        '''
                return solution (with everything in the dummy)
        '''
        sol = []

        n_veh =prob.get('n_vehicles')

        for i in range(0,n_veh):
                sol.append(0)

        for i in range(0, prob.get('n_calls')):
                sol.append(i+1)
                sol.append(i+1)

        return sol

def load_vehicles(prob):
        '''
                return veh (set of all the numbers for the vehicles)
        '''
        veh = []

        for i in range(0, prob.get('n_vehicles')+1): #Vehicles 0,1,2,3 <- dummy one
                veh.append(i)
        
        return veh

def obtain_index_change_of_cars(current_sol, n_veh):
        '''
                return index_changing_car (i position where the car starts)
        '''
        index_changing_car = [0]

        for i in range(n_veh-1):
                index_changing_car.append(current_sol.index(0,index_changing_car[i]) + 1)
        
        return index_changing_car

def remove_empty_car(veh, n_call_veh):
        '''
                removes from veh the empty vehicles
        '''
        if 0 in n_call_veh:
                for i in range(len(n_call_veh)):
                        del veh[n_call_veh.index(0)]
                        n_call_veh.remove(0)
                        
                        if 0 not in n_call_veh:
                                break
                        
def remove_one_call(chosen_veh, veh, n_call_veh):
        '''
                removes one call from the vehicle, and if empty removes the car
        '''
        n_call_veh[veh.index(chosen_veh)] -= 2
        remove_empty_car(veh, n_call_veh)

def obtain_number_calls_car(index_changing_car, length_sol):
        '''
                return the number of calls in a car
        '''
        number_calls_car = []

        for i in range(len(index_changing_car)):
                number_calls_car.append(0)

        number_calls_car[len(index_changing_car)-1] = length_sol - index_changing_car[len(index_changing_car)-1]
        
        for i in range(len(index_changing_car)-1,0,-1):
                number_calls_car[i-1] = index_changing_car[i] - index_changing_car[i-1] -1
        
        return number_calls_car

def choose_a_random_call(current_sol, veh, n_veh, index_changing_car, chosen_veh):
        '''
                return chosen_call (random call from a chosen veh)
        '''
        if(chosen_veh < n_veh -1):
                chosen_call = random.choice(current_sol[index_changing_car[chosen_veh]:index_changing_car[chosen_veh+1]-1])
        else:
                chosen_call = random.choice(current_sol[index_changing_car[chosen_veh]:len(current_sol)])

        return chosen_call

def choose_a_random_index(current_sol, veh, n_veh, index_changing_car, chosen_veh):
        '''
                return chosen_index (random index for insertion in chosen_veh)
        '''
        if(chosen_veh < n_veh -1):
                n_index = len(current_sol[index_changing_car[chosen_veh]:index_changing_car[chosen_veh+1]-1])-1
        else:
                n_index = len(current_sol[index_changing_car[chosen_veh]:len(current_sol)])-1

        if(n_index > 1):
                chosen_index = random.randint(0, n_index)
        else:
                chosen_index = 0

        return chosen_index

def insert_call_in_veh_in_position(chosen_call, chosen_veh, current_sol, veh, n_veh, index_changing_car, chosen_index):
        '''
                return solution (with chosen_call inserted at the chosen_index of the chosen_car)
        '''
        if chosen_veh > 0:
                chosen_index += index_changing_car[chosen_veh]

        current_sol.insert(chosen_index, chosen_call)
        
        index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)  

        return current_sol

def insert_call_in_vehicle(chosen_call, chosen_veh, current_sol, veh, n_veh, index_changing_car):
        '''
                return solution (with chosen_call inserted randomly in the chosen_Car)
        '''
        for i in range(2):
                chosen_index = choose_a_random_index(current_sol, veh, n_veh, index_changing_car, chosen_veh)
                current_sol = insert_call_in_veh_in_position(chosen_call, chosen_veh, current_sol, veh, n_veh, index_changing_car, chosen_index)
                        
        return current_sol

def insert_call_in_random_veh(chosen_call, current_sol, veh, n_veh, index_changing_car):
        '''
                return solution (with chosen_call inserted in a random position of a random car)
        '''
        chosen_veh = random.choice(veh)

        return insert_call_in_vehicle(chosen_call, chosen_veh, current_sol, veh, n_veh, index_changing_car), chosen_veh

def obtain_cost_call_in_car(prob, chosen_call, chosen_veh):
        '''
                return cost_car (cost of chosen_veh transporting chosen_Call)
        '''
        if(chosen_veh == prob.get('n_vehicles')):
                return prob['Cargo'][chosen_call-1][3]
        else:
                return prob['PortCost'][chosen_veh, chosen_call-1]

def insert_best_car(prob, chosen_call, current_sol, veh, n_veh, index_changing_car):
        '''
                return solution (where the chosen_call has been inserted in the 'best car', having the less cost by the previous funct)
        '''
        min_cost = 1e9
        best_car = 0
        
        for i in range(n_veh):
                cost_car = obtain_cost_call_in_car(prob, chosen_call, i)
                if(cost_car > 0):
                        if(cost_car < min_cost):
                                min_cost = cost_car
                                best_car = i

        return insert_call_in_vehicle(chosen_call, best_car, current_sol, veh, n_veh, index_changing_car), best_car

def is_car_full_with_call(prob, copy_current_sol, chosen_call, chosen_veh, veh, n_veh, index_changing_car):
        '''
                return True/False (check if you are over the capacity of chosen_veh after inserting chosen_Call)
        '''
        if(chosen_veh < n_veh - 1):
                sol = copy_current_sol[:]

                calls_car = get_calls_in_car_duplicated(copy_current_sol, veh, n_veh, index_changing_car, chosen_veh)

                if(chosen_call in calls_car):
                        return False

                sol = insert_call_in_vehicle(chosen_call, chosen_veh, sol, veh, n_veh, index_changing_car)

                return check_capacity(sol, prob, chosen_veh)
        return False
                   
def get_best_car_best_position(prob, chosen_call, current_sol, veh, n_veh, index_changing_car):
        '''
                return best_car, pos1, pos2, cost
        '''
        costs_car = []
        cars = []
        best_cars = []
        
        for i in range(n_veh):
                cost_car = obtain_cost_call_in_car(prob, chosen_call, i)

                if(not is_car_full_with_call(prob, current_sol, chosen_call, i, veh, n_veh, index_changing_car) and cost_car > 0):
                        costs_car.append(cost_car)
                        cars.append(i)

        order = np.argsort(np.array(costs_car))
        
        if(len(order) < 1):
                return n_veh-1, 0, 1, 0

        costs_car = [(costs_car[order[0]])/float(costs_car[i]) for i in order]

        costs_car = normalise_parameters(costs_car)

        selected = select_heuristic(costs_car)

        best_car = cars[order[selected]]

        if(best_car  == n_veh -1):
                return n_veh-1, 0, 1, 0

        best_pos1, best_pos2, best_cost = find_best_position_in_car(prob, chosen_call, best_car, current_sol, veh, n_veh, index_changing_car)

        if(best_pos1 < 0):
                return n_veh-1, 0, 1, 0


        return best_car, best_pos1, best_pos2, best_cost

def find_best_position_in_car(prob, chosen_call, best_car, current_sol, veh, n_veh, index_changing_car):
        '''
                return pos1, pos2, cost
        '''
        inf = sup = -1
        best_one = 1e9
        copy_current_sol = current_sol[:]
        calls_car = get_calls_in_car_duplicated(copy_current_sol, veh, n_veh, index_changing_car, best_car)
        
        if(len(calls_car) < 1):
                copy_current_sol = insert_call_in_veh_in_position(chosen_call, best_car, copy_current_sol, veh, n_veh, index_changing_car, 0)
                copy_current_sol = insert_call_in_veh_in_position(chosen_call, best_car, copy_current_sol, veh, n_veh, index_changing_car, 1)
                cost_new_sol = calc_cost_car(copy_current_sol, prob, best_car)
                return 0, 1, cost_new_sol



        for i in range (len(calls_car)+1):
                for j in range(i+1, len(calls_car)+2):
                        copy_current_sol = insert_call_in_veh_in_position(chosen_call, best_car, copy_current_sol, veh, n_veh, index_changing_car, i)
                        copy_current_sol = insert_call_in_veh_in_position(chosen_call, best_car, copy_current_sol, veh, n_veh, index_changing_car, j)
                        
                        if(car_feasibility_check(copy_current_sol, prob, best_car)):
                                cost_new_sol = calc_cost_car(copy_current_sol, prob, best_car)
                                if(cost_new_sol < best_one):
                                        best_one = cost_new_sol
                                        inf = i
                                        sup = j
                        else:
                                copy_current_sol.remove(chosen_call)
                                copy_current_sol.remove(chosen_call)
                                break

                        copy_current_sol.remove(chosen_call)
                        copy_current_sol.remove(chosen_call)


        return inf, sup, best_one

def choose_random_move_to_dummy(prob, copy_current_sol):
        '''
                return solution (where a call was randomly chosen and moved to the dummy)
        '''
        current_sol = copy_current_sol[:]

        veh = load_vehicles(prob)

        n_veh = len(veh)

        index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)

        number_calls_car = obtain_number_calls_car(index_changing_car, len(current_sol))

        remove_empty_car(veh, number_calls_car)

        if(len(veh) == 1):
                return current_sol
        
        chosen_veh = random.choice(veh[0:n_veh-1]) #Is not possible to choose the dummy

        chosen_call = choose_a_random_call(current_sol, veh, n_veh, index_changing_car, chosen_veh)

        veh = load_vehicles(prob)

        current_sol.remove(chosen_call)
        current_sol.remove(chosen_call)

        current_sol.append(chosen_call)
        current_sol.append(chosen_call)

        return current_sol

def get_calls_in_car(current_sol, veh, n_veh, index_changing_car, chosen_veh):
        '''
                return calls_in_car (the calls that are inserted in the chosen_veh, not in order nor duplicated)
        '''
        if(chosen_veh < n_veh -1):
                calls_in_car = current_sol[index_changing_car[chosen_veh]:index_changing_car[chosen_veh+1]-1]
        else:
                calls_in_car = current_sol[index_changing_car[chosen_veh]:len(current_sol)]
        
        calls_in_car = list(set(calls_in_car))

        return calls_in_car

def get_calls_in_car_duplicated(current_sol, veh, n_veh, index_changing_car, chosen_veh):
        '''
                return calls_in_car (the calls that are inserted in the chosen_veh, exactly how they are in the solution)
        '''
        if(chosen_veh < n_veh -1):
                calls_in_car = current_sol[index_changing_car[chosen_veh]:index_changing_car[chosen_veh+1]-1]
        else:
                calls_in_car = current_sol[index_changing_car[chosen_veh]:len(current_sol)]

        return calls_in_car

def check_not_picked(prob, chosen_veh, calls_in_car):
        '''
                return not_picked (calls that can be picked by chosen_veh but haven't been picked yet)
        '''
        not_picked = []
        for i in range(prob['n_calls']):
                if(prob['VesselCargo'][chosen_veh][i] == 1):
                        if((i+1) not in calls_in_car):
                                not_picked.append(i+1)
        return not_picked

def choose_call_most_cost_in_car(prob, current_sol, veh, n_veh, index_changing_car, chosen_veh):
        '''
                return call (checks the cost of all the calls in chosen_veh, chooses the most costly but with a probability to don't choose always the same)
        '''
        calls_in_car = get_calls_in_car(current_sol, veh, n_veh, index_changing_car, chosen_veh)

        cost_array = []

        for k in calls_in_car:
                cost_array.append(obtain_cost_call_in_car(prob, k, chosen_veh))
        
        cost_array = np.array(cost_array)

        order = np.argsort(cost_array)

        cost_array = [float(cost_array[i])/(cost_array[order[len(order)-1]]) for i in order]
        '''
                most_cost --> 1, but then they are normalised
        '''

        cost_array = normalise_parameters(cost_array)

        selected = select_heuristic(cost_array)

        return calls_in_car[selected]

def generate_init_parameters():
        '''
                return parameters 
                [0...Number_op) = probabilities
                [Number_op...Number_op*2) = points
                [Number_op*2...End) = times_chosen
        '''
        parameters = [1] * NUMBER_OPERATORS
        parameters += [0] * NUMBER_OPERATORS
        parameters += [0] * NUMBER_OPERATORS
        parameters = normalise_parameters(parameters)
        return parameters

def update_parameters(parameters, points, times_chosen):
        '''
                return parameters (updated with the times_chosen and points, applying a kill barrier to have always a small prob of choosing everyone)
        '''
        lamb = 0.2
        for k in range(len(parameters)):
                parameters[k] = (1-lamb) * parameters[k] + lamb * points[k]/times_chosen[k]  
        
        parameters = normalise_parameters(parameters)
        kill_barrier = False

        order = np.argsort(np.array(parameters))
        

        for i in range(NUMBER_OPERATORS):
                if(parameters[i] < 0.1):
                        kill_barrier = True
                        parameters[i] = 0.15
                        parameters[order[len(order)-1]] -= 0.08
        if(kill_barrier):
                parameters = normalise_parameters(parameters)
        return parameters
        
def normalise_parameters(parameters):
        '''
                return parameters (normalised)
        '''
        norm = [float(i)/sum(parameters) for i in parameters]
        return norm
        
def select_heuristic(parameters):
        '''
                return selected_heuristic (chosen randomly between the probabilities)
        '''
        rand = random.random()
        aux = 0
        selected = 0
        for i in range(len(parameters)):
                aux += parameters[i]
                if(rand < aux):
                        selected = i
                        break
        return selected

def get_calls_not_removed(current_sol, veh, n_veh, index_changing_car):
        '''
                return calls (calls that are being transported right now)
        '''
        calls = []
        for k in veh:
                calls += get_calls_in_car(current_sol, veh, n_veh, index_changing_car, k)
        return calls

def car_feasibility_check(solution, problem, selected_car):
        '''
                return feasibility, c (checks if selected_car is feasible (feasibility bool) and in case is not c is the reason why)
        '''
        num_vehicles = problem['n_vehicles']

        if(selected_car == num_vehicles):
                return True, 'Feasible'

        num_vehicles = problem['n_vehicles']
        Cargo = problem['Cargo']
        TravelTime = problem['TravelTime']
        FirstTravelTime = problem['FirstTravelTime']
        VesselCapacity = problem['VesselCapacity']
        LoadingTime = problem['LoadingTime']
        UnloadingTime = problem['UnloadingTime']
        VesselCargo = problem['VesselCargo']
        solution = np.append(solution, [0])
        ZeroIndex = np.array(np.where(solution == 0)[0], dtype=int)
        feasibility = True
        tempidx = 0
        c = 'Feasible'
        for i in range(num_vehicles):
                currentVPlan = solution[tempidx:ZeroIndex[i]]
                currentVPlan = currentVPlan - 1
                NoDoubleCallOnVehicle = len(currentVPlan)
                tempidx = ZeroIndex[i] + 1
                if(i == selected_car):
                        if NoDoubleCallOnVehicle > 0:

                                if not np.all(VesselCargo[i, currentVPlan]):
                                        feasibility = False
                                        c = 'incompatible vessel and cargo'
                                        break
                                else:
                                        LoadSize = 0
                                        currentTime = 0
                                        sortRout = np.sort(currentVPlan)
                                        I = np.argsort(currentVPlan)
                                        Indx = np.argsort(I)
                                        LoadSize -= Cargo[sortRout, 2]
                                        LoadSize[::2] = Cargo[sortRout[::2], 2]
                                        LoadSize = LoadSize[Indx]
                                        if np.any(VesselCapacity[i] - np.cumsum(LoadSize) < 0):
                                                feasibility = False
                                                c = 'Capacity exceeded'
                                                break
                                        Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
                                        Timewindows[0] = Cargo[sortRout, 6]
                                        Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
                                        Timewindows[1] = Cargo[sortRout, 7]
                                        Timewindows[1, ::2] = Cargo[sortRout[::2], 5]

                                        Timewindows = Timewindows[:, Indx]

                                        PortIndex = Cargo[sortRout, 1].astype(int)
                                        PortIndex[::2] = Cargo[sortRout[::2], 0]
                                        PortIndex = PortIndex[Indx] - 1

                                        LU_Time = UnloadingTime[i, sortRout]
                                        LU_Time[::2] = LoadingTime[i, sortRout[::2]]
                                        LU_Time = LU_Time[Indx]
                                        Diag = TravelTime[i, PortIndex[:-1], PortIndex[1:]]
                                        FirstVisitTime = FirstTravelTime[i, int(Cargo[currentVPlan[0], 0] - 1)]

                                        RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

                                        ArriveTime = np.zeros(NoDoubleCallOnVehicle)
                                        for j in range(NoDoubleCallOnVehicle):
                                                ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
                                                if ArriveTime[j] > Timewindows[1, j]:
                                                        feasibility = False
                                                        c = 'Time window exceeded at call {}'.format(j)
                                                        break
                                                currentTime = ArriveTime[j] + LU_Time[j]

        return feasibility, c

def calc_cost_car(solution, problem, selected_car):
        '''
                return TotalCost (return the cost of the selected_Car) 
        '''
        Solution = solution
        
        num_vehicles = problem['n_vehicles']
        Cargo = problem['Cargo']
        TravelCost = problem['TravelCost']
        FirstTravelCost = problem['FirstTravelCost']
        PortCost = problem['PortCost']


        NotTransportCost = 0
        RouteTravelCost = np.zeros(num_vehicles)
        CostInPorts = np.zeros(num_vehicles)

        Solution = np.append(Solution, [0])
        ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
        tempidx = 0

        for i in range(num_vehicles + 1):
        
                currentVPlan = Solution[tempidx:ZeroIndex[i]]
                currentVPlan = currentVPlan - 1
                NoDoubleCallOnVehicle = len(currentVPlan)
                tempidx = ZeroIndex[i] + 1

                if (i == selected_car):
                        if i == num_vehicles:
                                NotTransportCost = np.sum(Cargo[currentVPlan, 3]) / 2
                        else:
                                if NoDoubleCallOnVehicle > 0:
                                        sortRout = np.sort(currentVPlan)
                                        I = np.argsort(currentVPlan)
                                        Indx = np.argsort(I)

                                        PortIndex = Cargo[sortRout, 1].astype(int)
                                        PortIndex[::2] = Cargo[sortRout[::2], 0]
                                        PortIndex = PortIndex[Indx] - 1

                                        Diag = TravelCost[i, PortIndex[:-1], PortIndex[1:]]

                                        FirstVisitCost = FirstTravelCost[i, int(Cargo[currentVPlan[0], 0] - 1)]
                                        RouteTravelCost[i] = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
                                        CostInPorts[i] = np.sum(PortCost[i, currentVPlan]) / 2

        TotalCost = NotTransportCost + sum(RouteTravelCost) + sum(CostInPorts)
        return TotalCost

def check_capacity(solution, problem, selected_car):
        '''
                return excedeed_capacity (checks if selected_car has excedeed his maximun capacity)
        '''
        num_vehicles = problem['n_vehicles']

        if(selected_car == num_vehicles):
                return False

        num_vehicles = problem['n_vehicles']
        Cargo = problem['Cargo']
        TravelTime = problem['TravelTime']
        FirstTravelTime = problem['FirstTravelTime']
        VesselCapacity = problem['VesselCapacity']
        LoadingTime = problem['LoadingTime']
        UnloadingTime = problem['UnloadingTime']
        VesselCargo = problem['VesselCargo']
        solution = np.append(solution, [0])
        ZeroIndex = np.array(np.where(solution == 0)[0], dtype=int)
        excedeed_capacity = False
        tempidx = 0
        c = 'Feasible'
        for i in range(num_vehicles):
                currentVPlan = solution[tempidx:ZeroIndex[i]]
                currentVPlan = currentVPlan - 1
                NoDoubleCallOnVehicle = len(currentVPlan)
                tempidx = ZeroIndex[i] + 1
                if(i == selected_car):
                        if NoDoubleCallOnVehicle > 0:
                                LoadSize = 0
                                currentTime = 0
                                sortRout = np.sort(currentVPlan)
                                I = np.argsort(currentVPlan)
                                Indx = np.argsort(I)
                                LoadSize -= Cargo[sortRout, 2]
                                LoadSize[::2] = Cargo[sortRout[::2], 2]
                                LoadSize = LoadSize[Indx]
                                if np.any(VesselCapacity[i] - np.cumsum(LoadSize) < 0):
                                        excedeed_capacity = True

        return excedeed_capacity

def create_memory():
        '''
                return an empty dictionary
        '''
        return {}

def check_and_update(mem, new_entry):
        '''
                return mem, is_new
        '''
        copy_mem = mem
        aux = hash(new_entry)
        is_new = False
        if(not(aux in copy_mem)):
                is_new = True
                mem[aux] = 1
        
        return copy_mem, is_new

def create_update_utilities_removing_empties(prob, current_sol):
        '''
                return veh, n_veh, index_changing_car, number_calls_car (removes the empty cars)
        '''
        veh = load_vehicles(prob)

        n_veh = len(veh)

        index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)

        number_calls_car = obtain_number_calls_car(index_changing_car, len(current_sol))

        remove_empty_car(veh, number_calls_car)

        return veh, n_veh, index_changing_car, number_calls_car

def create_update_utilities_not_removing_empties(prob, current_sol):
        '''
                return veh, n_veh, index_changing_car, number_calls_car (not removing the empty cars)
        '''
        veh = load_vehicles(prob)

        n_veh = len(veh)

        index_changing_car = obtain_index_change_of_cars(current_sol, n_veh)

        number_calls_car = obtain_number_calls_car(index_changing_car, len(current_sol))

        return veh, n_veh, index_changing_car, number_calls_car

'''
PRINT AND LIMITS
'''

def print_all_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time):
        print('Best solution of the 10 times -> ' + str(best_Of_The_Best))

        print('Best cost of the 10 times -> ' + str(best_cost_OTB))

        print('Average cost -> ' + str(cost_average))

        print('Cost of the initial solution -> ' + str(cost_initial))

        print('Improvement -> ' + str(float("{0:,.3f}".format(100*(cost_initial-best_cost_OTB)/cost_initial))) + '%')

        print('Average running time -> ' + str(float("{0:,.7f}".format(elapsed_time))))

def print_raw_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time):
        '''
                prints the results in the 'raw' format used to pass it to excell
        '''
        print(str(cost_average))
        print(str(best_cost_OTB))
        print(str(float("{0:,.3f}".format(100*(cost_initial-best_cost_OTB)/cost_initial))) + '%')
        print(str(float("{0:,.7f}".format(elapsed_time))))
        print(str(best_Of_The_Best))

def write_all_results(best_Of_The_Best, best_cost_OTB, cost_average, cost_initial, elapsed_time):
        '''
                when 'raw' this writes the results in a file
        '''
        f = open("results/FinalRun2.txt", "a")
        
        f.write('\nBest solution of the 10 times -> ' + str(best_Of_The_Best))

        f.write('\nBest cost of the 10 times -> ' + str(best_cost_OTB))

        f.write('\nAverage cost -> ' + str(cost_average))

        f.write('\nCost of the initial solution -> ' + str(cost_initial))

        f.write('\nImprovement -> ' + str(float("{0:,.3f}".format(100*(cost_initial-best_cost_OTB)/cost_initial))) + '%')

        f.write('\nAverage running time -> ' + str(float("{0:,.7f}".format(elapsed_time))) + str('\n'))

        f.close()

def limit_seconds(problem):
        '''
                return seconds (limit of time based on the number of calls)
        '''
        n_calls = problem['n_calls']

        if(n_calls < 10):
                seconds = 10
        elif(n_calls < 30):
                seconds = 30
        elif(n_calls < 80):
                seconds = 80
        elif(n_calls < 110):
                seconds = 180
        else:
                seconds = 300

        return seconds

def avg_ite(problem):
        '''
                return avg_ite (used for the cooling factor, was fixed while testing)
        '''
        return 16000