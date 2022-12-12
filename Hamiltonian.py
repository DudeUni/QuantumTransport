import numpy as np

# store hopping matrix and its direction
class Hopping:
    matrixValue = np.matrix()
    hopVector = np.vector()

# generate the whole system
class system:
    # all the hoppings and number of terms
    Hops = [] 
    termNumber = 0

    def __init__(self) -> None:
        self.termNumber = 0



a = system()
print(a)