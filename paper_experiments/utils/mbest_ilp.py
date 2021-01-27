from gurobipy import Model, quicksum, LinExpr, GRB
import numpy as np
import copy
import time
from sklearn.utils.linear_assignment_ import linear_assignment
import pickle
import itertools
import pdb
from copy import deepcopy
import math
"""
Fn: ilp_assignment
------------------
Solves ILP problem using gurobi
"""
def ilp_assignment(model):
    
    model.optimize()
    if(model.status == 3):
        return -1
    return

"""
Fn: initialize_model
--------------------
Initializes gurobi ILP model by setting the base objective
"""
# @profile
def initialize_model(cost_matrix, cutoff, model = None):
    #Add dummy detection
    cost_matrix = np.insert(cost_matrix,0, np.ones(cost_matrix.shape[0])*cutoff, axis=1)
    M,N = cost_matrix.shape
    if model is None:
        model = Model()
    else:
        model.remove(model.getVars())
        model.remove(model.getConstrs())
    model.setParam('OutputFlag', False)
    # y = []
    # for i in range(M):
    #     y.append([])
    #     for j in range(N):
    #         y[i].append(m.addVar(vtype=GRB.BINARY, name = 'y_%d%d'%(i,j)))
    y = model.addVars(M,N, vtype=GRB.BINARY, name = 'y')
    model.setObjective(quicksum(quicksum([y[i,j]*cost_matrix[i][j] for j in range(N)]) for i in range(M)), GRB.MINIMIZE)
    # for i in range(M):
    model.addConstrs((quicksum(y[i,j] for j in range(N))==1 for i in range(M)), name='constraint for track')
    # for j in range(1,N):
    model.addConstrs((quicksum(y[i,j] for i in range(M))<=1 for j in range(1, N)), name='constraint for detection')
    y = list(y.values())
    return model, M, N, y

"""
Fn: m_best_sol
--------------
Finds m_best solutions for object/track association givent the
input cost matrix. Solves constrained ILP problems using gurobi solver.
"""
def cache(func):
    cache = {}
    def cached_function(*args):
        cost_matrix = args[0]
        cost_matrix = np.hstack((np.ones((cost_matrix.shape[0], 1))*args[1], cost_matrix))
        if (cost_matrix.shape[0], cost_matrix.shape[1]) in cache:
            solution_list = cache[(cost_matrix.shape[0], cost_matrix.shape[1])]
            solution_vals = np.sum(solution_list*cost_matrix.reshape(1, -1), axis = 1)
            return solution_list, solution_vals
        else: 
            solution_list, solution_vals = func(*args)
            cache[(cost_matrix.shape[0], cost_matrix.shape[1])] = solution_list
            return solution_list, solution_vals
    return cached_function
# @profile
def num_solutions(cost_matrix):
    M,N = cost_matrix.shape
    N += 1
    count = 0
    for i in range(min(M+1, N)):
        count += np.prod(range(M-i+1, M+1))*np.prod(range(N-i, N))//math.factorial(i)
        if count > 2000:
            break
    return int(count)

@cache
def enumerate_solutions(cost_matrix, cutoff, num_solutions):
    # num_solutions = [[2, 3, 4, 5, 6, 7],[3, 7, 13, 21, 31],[4, 13, 34, 73, 136],[5, 21, 73, 209, 501],[6, 31, 136, 501, 1546], [7]]
    cost_matrix = np.hstack((np.ones((cost_matrix.shape[0], 1))*cutoff, cost_matrix))
    M,N = cost_matrix.shape
    solution_list = np.zeros((num_solutions, M, N), dtype = np.int32)
    solution_list[:, :, 0] = 1
    count = 0
    for i in range(min(M+1, N)):
        for chosen in itertools.combinations(range(M), i):
            for perm in itertools.permutations(range(1,N), i):
                if chosen:
                    solution_list[[count]*len(chosen), chosen, perm] = 1
                    solution_list[[count]*len(chosen), chosen, [0]*len(chosen)] = 0
                count += 1
    solution_vals = np.sum(np.sum(solution_list*np.expand_dims(cost_matrix, 0), axis = 1), axis = 1)
    solution_list = np.reshape(solution_list, (num_solutions, -1))
    return solution_list, solution_vals


def new_m_best_sol(cost_matrix, m_sol, cutoff, model = None):
    sols = num_solutions(cost_matrix)
    if sols <= 2000:
        return enumerate_solutions(cost_matrix, cutoff, sols)
    model, M, N, y = initialize_model(cost_matrix, cutoff, model)
    X = np.zeros((m_sol, M*N))
    xv = []
    if (ilp_assignment(model) == -1):
        xv.append(0)
    else:
        x = model.getAttr("X", y)
        X[0] = x
        xv.append(model.objVal)
    if m_sol > 1:
        model.addConstr(LinExpr(x,y) <= M-1, name = 'constraint_0')
        if (ilp_assignment(model) == -1):
            xv.append(0)
        else:
            x = model.getAttr("X", y)
            X[1] = x
            xv.append(model.objVal)
    if m_sol > 2:
        model.remove(model.getConstrByName('constraint_0'))
        second_best_solutions = []
        second_best_solution_vals = []
        partitions = []
        j = np.argmax(np.logical_xor(X[0], X[1]))
        partitions.append([j])
        partitions.append([j])
        model.addConstr(y[j]==X[0][j], name = 'partition_constraint')
        model.addConstr(LinExpr(X[0], y) <= M-1, name = 'non_equality_constraint')
        ilp_assignment(model)
        second_best_solutions.append(model.getAttr("X", y))
        second_best_solution_vals.append(model.objVal)
        model.remove(model.getConstrByName('non_equality_constraint'))
        model.remove(model.getConstrByName('partition_constraint'))
        model.addConstr(y[j]==X[1][j], name = 'partition_constraint')
        model.addConstr(LinExpr(X[1], y) <= M-1, name = 'non_equality_constraint')
        ilp_assignment(model)
        second_best_solution_vals.append(model.objVal)
        second_best_solutions.append(model.getAttr("X", y))
        model.remove(model.getConstrByName('non_equality_constraint'))
        model.remove(model.getConstrByName('partition_constraint'))
        
        for m in range(2, m_sol):
            l_k = np.argmin(second_best_solution_vals)
            X[m] = second_best_solutions[l_k]
            xv.append(second_best_solution_vals[l_k])
            if m==m_sol-1:
                break
            j = np.argmax(np.logical_xor(X[m], X[l_k]))
            parent_partition = partitions[l_k]
            constrs = []
            for idx in parent_partition:
                constrs.append(model.addConstr(y[idx]==X[l_k, idx]))
            model.addConstr(y[j]==X[m][j], name = 'partition_constraint_new')
            model.addConstr(LinExpr(X[m], y) <= M-1, name = 'non_equality_constraint')
            if(ilp_assignment(model) == -1):
                second_best_solutions.append(np.ones((M,N)))
                second_best_solution_vals.append(np.inf)
            else:
                second_best_solutions.append(model.getAttr("X", y))
                second_best_solution_vals.append(model.objVal)
            model.remove(model.getConstrByName('partition_constraint_new'))
            model.remove(model.getConstrByName('non_equality_constraint'))
            model.addConstr(LinExpr(X[l_k], y) <= M-1, name = 'non_equality_constraint')
            model.addConstr(y[j]==X[l_k][j], name = 'partition_constraint_new')
            if(ilp_assignment(model) == -1):
                second_best_solution_vals[l_k] = np.inf
                second_best_solutions[l_k] = np.ones((M,N))
            else:
                second_best_solution_vals[l_k] = model.objVal
                second_best_solutions[l_k] = model.getAttr("X", y)
            model.remove(model.getConstrByName('partition_constraint_new'))
            model.remove(model.getConstrByName('non_equality_constraint'))
            partitions[l_k].append(j)
            partitions.append(copy.deepcopy(partitions[l_k]))
            for constr in constrs:
                model.remove(constr)



    # X = np.asarray(X)
    xv = np.asarray(xv)
    return X, xv
def linear_assignment_wrapper(a):
    return linear_assignment(a)

if __name__=='__main__':
    # a = np.random.randn(100,100)
    # # cProfile.run('m_best_sol(a,1,10)', 'mbest.profile')
    # # cProfile.run('linear_assignment(a)', 'hungarian.profile')
    # total = 0
    # for i in range(10):
    #     start = time.time()
    #     _, sol_cost = m_best_sol(a, 1, 10)
    #     end = time.time()
    #     total+= end-start
    # print("Time for JPDA m=1, is %f"%(total/10))
    # total = 0
    # for i in range(10):
    #     start = time.time()
    #     ass = linear_assignment(a)
    #     end = time.time()
    #     total+= end-start
    # print("Time for Hungarian, is %f"%(total/10))
    
    np.random.seed(14295)
    # Check JPDA matches Hungarian
    # while True:
    #     print('*******')
    #     a = np.random.randn(100,100)
    #     X, _ = new_m_best_sol(a, 1, 10)
    #     X = np.reshape(X[0], (100,101))[:,1:]
    #     ass = linear_assignment(a)
    #     output_hungarian = np.zeros(a.shape)
    #     output_hungarian[ass[:,0], ass[:, 1]] = 1
    #     assert(np.all(output_hungarian==X))
    #
    # Output to file to check

    #  np.random.seed(14295)
    # vals = []
    # a = np.random.randn(5,5)
    a = np.array([[0.1,0.6,0.2,0.3],[0.4,0.1,0.9,0.4],[0.3,0.5,0.1,0.7],[0.8,0.2,0.2,0.1]])
    num_solutions(a)
    # enumerate_solutions(a.shape[0], a.shape[1]+1)
    # ass = linear_assignment_wrapper(a)
    # m = Model()
    sols, vals = new_m_best_sol(a, 100, 10)
    for i, val in enumerate(vals):
        print(np.reshape(sols[i], (4,5)), val)
    # print(np.reshape(sols[1], (4,5)), vals[1])
    # print(np.reshape(sols[2], (4,5)), vals[2])
    # print(np.reshape(sols[3], (4,5)), vals[3])

    # with open('test.pkl', 'wb') as f:
    #     pickle.dump(vals, f)
