
import csv
import numpy as np
import matplotlib.pyplot as plt

def LoadDataset():
    data = []
    with open("dataset.csv") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            data.append([float(row[0]),float(row[1]) ])
    return np.array(data)

def ManhattanDistance(x1, y1, x2, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)


class SomClassifier(object):
    def __init__(self, data, epochs, neighbour, col, row, dim, learnRate ):
        self.data = data        
        self.epochs = epochs
        self.neighbour = neighbour
        self.col = col
        self.row = row
        self.learnRate = learnRate
        self.dim = dim
        self.range_max = col + row
        self.som = np.random.uniform(4, 18, (col, row, 2))

        m = np.reshape(self.som, (col * row, 2))
        m = np.transpose(m)
        plt.title("Posisi neuron SoM sekarang, close untuk lanjut")
        plt.plot(m[0], m[1], 'bo')
        plt.scatter(*zip(*self.data))
        plt.show()
    
    def Draw(self, s):
        plt.cla()
        plt.title("Epoch ke " + str(s) + " dari " + self.epochs)
        n = np.reshape(self.som, (self.col * self.row, 2)).T
        plt.plot(n[0], n[1], 'ro')
        plt.scatter(*zip(*self.data))
        plt.pause(0.05)
        

    def FindBMU(self, selectedData):
        bmu = np.array([0, 0])
        minimumDist = float("inf")
        for x in range(self.col):
            for y in range(self.row):
                dist = np.linalg.norm(self.som[x][y] - selectedData)
                if(dist < minimumDist):
                    minimumDist = dist
                    bmu = np.array([x, y])

        return bmu

    def UpdateSOM(self, bmu, selectedData, currRange, currRate):
        for x in range(self.col):
            for y in range(self.row):
                if(ManhattanDistance(bmu[0], bmu[1], x, y) < currRange):
                    self.som[x][y] = self.som[x][y] + currRate * -(self.som[x][y] - selectedData)

    def Classify(self):
        clusters = []
        i = 1
        for data in self.data:
            # Find the closest vector
            bmu = self.FindBMU(data)
            clusters.append([data, bmu[0]+bmu[1]*self.col])
            print("Processing ", i , " dari ", len(self.data))
            i += 1

        cl = np.zeros(self.col*self.row)
        for a in clusters:
            cl[a[1]] += 1
            if(a[1] == 0):
                plt.plot(a[0][0], a[0][1], 'ro')
            elif(a[1] == 1):
                plt.plot(a[0][0], a[0][1], 'go')
            elif(a[1] == 2):
                plt.plot(a[0][0], a[0][1], 'bo')
            elif(a[1] == 3):
                plt.plot(a[0][0], a[0][1], 'co')
            elif(a[1] == 4):
                plt.plot(a[0][0], a[0][1], 'mo')
            elif(a[1] == 5):
                plt.plot(a[0][0], a[0][1], 'yo')
            elif(a[1] == 6):
                plt.plot(a[0][0], a[0][1], 'ko')
            elif(a[1] == 7):
                plt.plot(a[0][0], a[0][1], 'r+')
            elif(a[1] == 8):
                plt.plot(a[0][0], a[0][1], 'g+')
            elif(a[1] == 9):
                plt.plot(a[0][0], a[0][1], 'b+')
            elif(a[1] == 10):
                plt.plot(a[0][0], a[0][1], 'c+')
            elif(a[1] == 11):
                plt.plot(a[0][0], a[0][1], 'm+')
            elif(a[1] == 12):
                plt.plot(a[0][0], a[0][1], 'y+')
            elif(a[1] == 13):
                plt.plot(a[0][0], a[0][1], 'k+')
            elif(a[1] == 14):
                plt.plot(a[0][0], a[0][1], 'gv')
            else:
                plt.plot(a[0][0], a[0][1], 'bv')
            
        plt.show()
        print(cl)        




    def Train(self, showAnimation=False):
        for s in range(self.epochs):
            epochLeft = 1 - (s*1.0/self.epochs)
            currRange = np.round(epochLeft * self.range_max)
            currRate = epochLeft * self.learnRate
            # pick a random data
            pickData =  self.data[np.random.randint(len(self.data))]
            # Find Best Matching Unit
            bmu = self.FindBMU(pickData)
            self.UpdateSOM(bmu, pickData, currRange, currRate)
            print("epoch: ", s, " dari ", self.epochs)
            print(self.som)
            if(showAnimation) : self.Draw(s)
        
        plt.title("Hasil train neuron, click close untuk clasification")    
        n = np.reshape(self.som, (self.col * self.row, 2)).T
        plt.plot(n[0], n[1], 'ro')
        plt.scatter(*zip(*self.data))
        plt.show()
        plt.cla()
        self.Classify()
        

classifier = SomClassifier(LoadDataset(), 10000, 1, 4, 3, 2, 0.3 )
classifier.Train()

