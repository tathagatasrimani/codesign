import cvxpy as cp


def main():
    d1 = cp.Variable(pos=True)
    d2 = cp.Variable(pos=True)
    d3 = cp.Variable(pos=True)
    d4 = cp.Variable(pos=True)
    e1 = cp.Variable(pos=True)
    e2 = cp.Variable(pos=True)
    e3 = cp.Variable(pos=True)
    e4 = cp.Variable(pos=True)
    p1 = cp.Variable(pos=True)
    p2 = cp.Variable(pos=True)
    p3 = cp.Variable(pos=True)
    p4 = cp.Variable(pos=True)
    d = cp.maximum(d1+cp.maximum(d2*d4, d3), d4 + cp.maximum(d1, d2))
    eact = e1+e2+e3+e4
    ppass = p1+p2+p3+p4
    edp = d*(eact + d*ppass)
    obj = edp
    constr = [d1 >= 1, d2 >= 1, d3 >= 1, d4 >= 1, e1 >= 1, e2 >= 1, e3 >= 1, e4 >= 1, p1 >= 1, p2 >= 1, p3 >= 1, p4 >= 1]
    constr += [cp.maximum(d1/2, 2/d1) <= 2]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(gp=True)
    print(f"d: {d.value}, eact: {eact.value}, ppass: {ppass.value}, edp: {edp.value}, obj: {obj.value}")

if __name__ == "__main__":
    main()