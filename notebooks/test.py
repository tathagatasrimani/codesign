import sympy as sp



test_object = {}

def make_test(arg):
    e = {}
    for obj in test_object:
        e[obj] = test_object[obj]
    return e, arg

d = {
    "test": make_test("bruh")
}



def main():
    test_object["1"] = 1
    test_object["2"] = 2


    print(d["test"])

main()