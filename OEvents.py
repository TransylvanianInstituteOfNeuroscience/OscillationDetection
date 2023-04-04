import numpy as np

from common.extrema import find_extrema_2D
from structs import BoundingBox


class Event:
    def __init__(self, peakValue, coordinates, spectrumCoordinates, boundingBox, timeRange, frequencyRange, data=None):
        """

        :param peakValue:
        :param coordinates:
        :param spectrumCoordinates:
        :param boundingBox:
        :param timeRange:
        :param frequencyRange:
        :param data:
        """
        self.peakValue = peakValue
        self.coordinates = coordinates
        self.spectrumCoordinates = spectrumCoordinates
        #(int L, int T, int R, int B)
        self.boundingBox = boundingBox
        self.timeRange = timeRange
        self.frequencyRange = frequencyRange
        self.data = data

        self.timeSpan = len(timeRange)
        self.frequencySpan = len(frequencyRange)
        self.timeBins = boundingBox.R - boundingBox.L + 1
        self.frequencyBins = boundingBox.B - boundingBox.T + 1

        # CoordinatesInBox => (Coordinates.Row - BoundingBox.B, Coordinates.Col - BoundingBox.L);
        self.coordinatesBox = (coordinates[0] - boundingBox.B, coordinates[1] - boundingBox.L)


class Peak:
    def __init__(self, value, row, col, boundingBox=None):
        self.value = value
        self.row = row
        self.col = col
        self.boundingBox = boundingBox
        self.coordinates = (row, col)


class OEvents:

    def evaluateMedianThr(self, data, medianThrFactor=4, eventAmpThr=0.5, perFrequencyMedians=None, includeDataCrops=False):
        self.peakThreshold = medianThrFactor
        self.eventAmpThreshold = eventAmpThr

        self.processInputMedianPerFreqThreshold(data.powerValues, perFrequencyMedians)

        self.extractPeaks()

        for peak in self.peaks:
            self.computeBoundingBox(peak)
        
        self.mergeBoundingBoxes()


        self.events = []
        for peak in self.peaks:
            event = Event(
                peakValue=peak.value,
                coordinates=(peak.row, peak.col),
                spectrumCoordinates=(data.binToFrequency(peak.row), data.binToTimepoint(peak.col)),
                boundingBox=peak.boundingBox,
                timeRange=(data.binToTimepoint(peak.boundingBox.L), data.binToTimepoint(peak.boundingBox.R)),
                frequencyRange=(data.binToFrequency(peak.boundingBox.B), data.binToFrequency(peak.boundingBox.T))
            )

            if includeDataCrops:
                pass

            self.events.append(event)


    def processInputGlobalMedian(self, data):
        self.processedData = data / np.median(data)


    def processInputMedianPerFreqThreshold(self, data, perFrequencyMedians):
        self.processedData = np.zeros_like(data)

        if perFrequencyMedians == None:
            for i in range(len(data)):
                self.processedData[i] = data[i] / np.median(data[i])
        else:
            for i in range(len(data)):
                self.processedData[i] = data[i] / perFrequencyMedians[i]

    def extractPeaks(self):

        self.peaks = []
        _, maxima = find_extrema_2D(self.processedData, 1, 'and')

        indexes = np.where(maxima==True)
        maxima_locs = np.vstack((indexes[0], indexes[1]))
        for max_list in maxima_locs.T:
            max = (max_list[0], max_list[1])
            if self.processedData[max] > self.peakThreshold:
                self.peaks.append(
                    Peak(
                        row=max[0],
                        col=max[1],
                        value=self.processedData[max]
                    )
                )

        # sort peaks by descending values
        self.peaks.sort(key=lambda x: x.value)

    def computeBoundingBox(self, peak):
        L = peak.col
        T = peak.row
        R = peak.col
        B = peak.row

        data = self.processedData
        localThr = min(self.eventAmpThreshold * peak.value, self.peakThreshold)

        while L > 0                 and data[peak.row, L-1] > localThr:  L -= 1
        while T < data.shape[0] - 1 and data[T+1, peak.col] > localThr:  T += 1
        while R < data.shape[1] - 1 and data[peak.row, R+1] > localThr:  R += 1
        while B > 0                 and data[B-1, peak.col] > localThr:  B -= 1

        peak.boundingBox = BoundingBox(L=L, T=T, R=R, B=B)

    def mergeBoundingBoxes(self):
        conflicts = []

        while True:
            oldPeaks = self.peaks.copy()
            self.peaks = []

            for conflict in conflicts:
                self.peaks.append(self.mergePeaks(oldPeaks[conflict[0]], oldPeaks[conflict[1]]))

            for peak_id, peak in enumerate(oldPeaks):

                peakMerged = False
                for conflict in conflicts:
                    if conflict[0] == peak_id or conflict[1] == peak_id:
                        peakMerged = True
                        break

                if peakMerged == False: 
                    self.peaks.append(peak)

            conflicts = []
            for i in range(len(self.peaks)-1):
                for j in range(i+1, len(self.peaks)):
                    if self.checkRectOverlap(self.peaks[i].boundingBox, self.peaks[j].boundingBox):
                        conflicts.append((i, j))
                        break

            if len(conflicts) == 0:
                break

        self.peaks.sort(key=lambda x: x.value)


    def mergePeaks(self, peak1, peak2):
        maxPeak = peak1 if peak1.value > peak2.value else peak2

        L = min(peak1.boundingBox.L, peak2.boundingBox.L)
        T = max(peak1.boundingBox.T, peak2.boundingBox.T)
        R = max(peak1.boundingBox.R, peak2.boundingBox.R)
        B = min(peak1.boundingBox.B, peak2.boundingBox.B)

        return Peak(
            row=maxPeak.row,
            col=maxPeak.col,
            value=maxPeak.value,
            boundingBox=BoundingBox(L=L, T=T, R=R, B=B)
        )


    def checkRectOverlap(self, rect1, rect2, overlapRatio=0.5):
        x1 = max(rect1.L, rect2.L)
        y2 = min(rect1.T, rect2.T)
        x2 = min(rect1.R, rect2.R)
        y1 = max(rect1.B, rect2.B)

        if x2 >= x1 and y2 >= y1:
            rect1Area = (rect1.T - rect1.B) * (rect1.R - rect1.L)
            rect2Area = (rect2.T - rect2.B) * (rect2.R - rect2.L)
            intersectionArea = (x2-x1) * (y2-y1)

            if intersectionArea > overlapRatio * min(rect1Area, rect2Area):
                return True

        return False