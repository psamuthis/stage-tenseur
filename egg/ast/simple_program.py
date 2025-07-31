def add(a, b):
    return a + b

def weird_func(a, b):
    return a + b * 2 - (b*a)

def calculate():
    # Redundant constant folding
    a = 1 + 0
    a = a * 2 / 2
    # b = 4
    # c = 5
    # a = weird_func(b, c) * 2  # Can be reduced to 3
    # d = 1 + 2 * 3 / 4 - 5
    #b = 5 * 1      # Mult by 1 is unnecessary
    #c = 10 / 2 * 2 # Can be reduced to 10
    #d = (4 + 6) - (3 + 3)  # Can be reduced to 4
    #e = ((a + b) - a)      # Redundant cancelation: just b

    # Identity operations
    #f = c + 0   # +0 does nothing
    #g = d * 1   # *1 does nothing

    # Slightly deeper redundancy
    #h = ((a * 1 + 0) / 1)  # All identities, just a

    #k = 1
    #for i in range(4):
        #k *= k + 1 - 1

if __name__ == "__main__":
    calculate()
