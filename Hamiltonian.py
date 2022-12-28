import numpy as np

# store hopping matrix and its direction
class Hopping:
    matrixValue = np.matrix()
    hopVector = np.vector()

    def __init__(self,H,dir) -> None:
        self.matrixValue, self.hopVector = H, dir

# generate the whole system
class system:
    # all the hoppings and number of terms
    Hops = [] 
    termNumber = 0

    def __init__(self) -> None:
        self.termNumber = 0

    def setHopping(self,H,dir):
        H = Hopping(H,dir)
        self.Hops.append(H)
        self.termNumber += 1
    
    def constructHamiltonian():
        pass

    def visualizeShape():
        pass
    



a = system()
print(a)