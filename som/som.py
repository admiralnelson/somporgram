import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SelfMap(object):
    def __init__(self, h, w, dimension):
        self.height = h
        self.width = w
        self.dimension = dimension
        self.som = np.zeros((h, w, dimension))

        """
        Initial Training Rate: L0
        Time constant: τ
        """
        self.L0 = 0
        self.Σ0 = 0
        self.τ = 0

    def PlotInit(self):
        #plt.ion()
        #fig, ax = plt.subplots()
        pass

    def Draw(self):
        plt.cla()
        x,y = self.som.T
        plt.scatter(x, y)
        plt.scatter(*zip(*self.data))
        plt.pause(0.1)
       


    def Train(self, L0, τ, Σ0, data):
        self.L0 = L0
        self.τ = τ
        self.Σ0 = Σ0
        self.data = data
        self.PlotInit()
        plt.show()

        t = 0
        while (self.Σ(t) >= 1.0):
            data_i = np.random.choice(range(len(data)))

            bmu = self.FindBestMatchingUnit(data[data_i])
            self.UpdateSOM(bmu, data[data_i], t)
            print(self.som)
            self.Draw()
            t += 1
        self.Draw()
        plt.pause(20)

    def QuantisationError(self):
        bmuDistancesList = []
        for vector in self.data:
            bmu = self.FindBestMatchingUnit(vector)
            bmuFeat = self.som[bmu]
            bmuDistancesList.append(np.linalg.norm(vector-bmuFeat))
        return np.array(bmuDistancesList).mean() 
        
    def FindBestMatchingUnit(self, vector):
        """
        BMU = best matching unit
        """
        bmuList = []
        for y in range(self.height):
            for x in range(self.width):
                distance = np.linalg.norm((vector-self.som[y,x]))
                bmuList.append(((y,x),distance))
        bmuList.sort(key=lambda x:  x[1])
        return bmuList[0][0]

    def UpdateBestMatchingUnit(self, bmu, vector, time):
        """
        update bmu
        on position bmu (y,x)
        with vector "vector" 
        at time "time"

        """
        self.bom[bmu] += self.L(time) * (vector-self.som[bmu])

    def UpdateSOM(self, bmu, vector, time):
        """
        update SOM cell
        on position bmu (y,x)
        with vector "vector" 
        at time "time"
        """
        for y in range(self.height):
            for x in range(self.width):
                distance2BMU = np.linalg.norm((np.array(bmu) - np.array((y,x))))
                self.UpdateCell((y,x), distance2BMU, vector, time)

    def UpdateCell(self, cell,distance2BMU, vector, time):
        self.som[cell] += self.N(distance2BMU, time)*self.L(time)*(vector-self.som[cell])

    def N(self, distance2BMU, time):
        currentΣ = self.Σ(time)
        return np.exp(-(distance2BMU**2)/(2*currentΣ**2))

    def L(self, t):
        return self.L0 * np.exp(-t/self.τ)

    def Σ(self, t):
        return self.Σ0*np.exp(-t/self.τ)

    def printArray(self):
        print(self.som)


def OpenFile(path):
    with open(path, "r") as file:
        data = file.read().split("\n")

        if(data.index("")):
            data.pop()

        for i in range(len(data)): 
            line = data[i].split(",")
            line[0] = float(line[0])
            line[1] = float(line[1])
            data[i] = (line[0], line[1])

    return data

print(OpenFile("dataset.csv"))
dat = OpenFile("dataset.csv")
som = SelfMap(20,20,2)
som.Train(0.8, 1e2, 10, dat)

