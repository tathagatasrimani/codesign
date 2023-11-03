
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
model = pyo.ConcreteModel()
model.nVars = pyo.Param(initialize=4)
model.N = pyo.RangeSet(model.nVars)
model.x = pyo.Var(model.N, within=pyo.Binary)
model.obj = pyo.Objective(expr=pyo.summation(model.x))
model.cuts = pyo.ConstraintList()
opt = SolverFactory('glpk')
opt.solve(model) 

# Iterate, adding a cut to exclude the previously found solution
for i in range(5):
   expr = 0
   for j in model.x:
       if pyo.value(model.x[j]) < 0.5:
           expr += model.x[j]
       else:
           expr += (1 - model.x[j])
   model.cuts.add( expr >= 1 )
   results = opt.solve(model)
   print ("\n===== iteration",i)
   model.display()