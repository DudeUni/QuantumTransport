import numpy as np

# store hopping matrix and its direction
class Hopping:
    matrixValue = []
    hopVector = np.vector()

    def __init__(self,H,dir) -> None:
        self.matrixValue, self.hopVector = H, dir

# generate the whole system
class Hamiltonian:
    # all the hoppings and number of terms
    hops = [] 
    termNumber = 0
    dimension = 2

    def __init__(self, dim=2) -> None:
        self.termNumber = 0
        self.dimension = dim

    def setHopping(self,H,dir):
        # H:hopping matrix, dir: direction in lattice units
        H = Hopping(H,dir)
        self.hops.append(H)
        self.termNumber += 1
    
    def constructHamiltonian():
        # return hopping to next layer matrix H1, H2, H3...
        pass

    def visualizeShape():
        pass
    



a = system()
print(a)