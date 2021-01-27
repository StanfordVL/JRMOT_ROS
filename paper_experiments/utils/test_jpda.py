from gurobipy import *
from numpy import *
'''
def mycallback(model, where):
    if where == GRB.callback.MIP:
        print model.cbGet(GRB.callback.MIP_NODCNT)
        print model.cbGet(GRB.callback.MIP_ITRCNT),'HEY MOTHERFUCKER'
    if where == GRB.callback.MIPNODE:
        print model.cbGet(GRB.callback.MIPNODE_OBJBST),'BEST OBJ'
'''


numT = 100
numC = 100

Assignment = random.random((numT,numC))

m=Model("Assignment")

X = []
for t in range(numT):
    X.append([])
    for c in range(numC):
        X[t].append(m.addVar(vtype=GRB.BINARY,name="X%d%d"% (t, c)))
m.update()
m.modelSense = GRB.MAXIMIZE
constraintT = []
constraintC = []
for t in range(numT):
    constraintT.append(m.addConstr(quicksum(X[t][c] for c in range(numC)) == 1 ,'constraintT%d' % t))
    
for c in range(numC):
    constraintT.append(m.addConstr(quicksum(X[t][c] for t in range(numT)) == 1 ,'constraintC%d' % t))

m.setObjective(quicksum(quicksum([X[t][c]*Assignment[t][c]    for c in range(numC)]) for t in range(numT)))
    
m.update()

#m.optimize(mycallback)
m.optimize()


print('runtime is %f'%m.Runtime)

