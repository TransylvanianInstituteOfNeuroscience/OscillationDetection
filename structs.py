class Spectrum2D:
    def __init__(self, timeValues, frequencyValues, powerValues):
        self.timeValues = timeValues
        self.frequencyValues = frequencyValues
        self.powerValues = powerValues

    def binToFrequency(self, row_index):
        return self.frequencyValues[row_index]

    def binToTimepoint(self, col_index):
        return self.timeValues[col_index]




class BoundingBox:
    def __init__(self, L, T, R, B):
        self.L = L
        self.T = T
        self.R = R
        self.B = B
