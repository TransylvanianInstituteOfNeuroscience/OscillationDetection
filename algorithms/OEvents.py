import numpy as np

from common.extrema import find_extrema_2D
from algorithms.structs import BoundingBox


class Event:
    def __init__(self, peakValue, coordinates, spectrumCoordinates, boundingBox, timeRange, frequencyRange, data=None):
        """

        :param peakValue: The value of the peak.
        :param coordinates: The coordinates in image space.
        :param spectrumCoordinates: The coordinates in time-frequency space.
        :param boundingBox: The bounding box in image space.
        :param timeRange: The time range in time-frequency space.
        :param frequencyRange: The frequency range in time-frequency space.
        :param data: A cutout of the spectrum data in the bounding box.
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
        """
        Encapsulates a peak.
        :param value: float - power value of the peak
        :param row: int - row index for the peak
        :param col: int - column index for the peak
        :param boundingBox: struct based on 4 values - representing the bouding box
        """
        self.value = value
        self.row = row
        self.col = col
        self.boundingBox = boundingBox
        self.coordinates = (row, col)


class OEvents:

    def evaluateMedianThr(self, data, medianThrFactor=4, eventAmpThr=0.5, perFrequencyMedians=None, includeDataCrops=False):
        """
        Evaluate the input spectrum.
        :param data: The input spectrum.
        :param medianThrFactor: The threshold at which peaks are considered.
        :param eventAmpThr: The amplitude threshold for cardinal expansion.
        :param perFrequencyMedians:
        :param includeDataCrops: If true, cutouts of the bounding boxes from the data will be included.
        :return:
        """
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
        """

        :param data:
        :return:
        """
        self.processedData = data / np.median(data)


    def processInputMedianPerFreqThreshold(self, data, perFrequencyMedians):
        """

        :param data:
        :param perFrequencyMedians:
        :return:
        """
        self.processedData = np.zeros_like(data)

        if perFrequencyMedians == None:
            for i in range(len(data)):
                self.processedData[i] = data[i] / np.median(data[i])
        else:
            for i in range(len(data)):
                self.processedData[i] = data[i] / perFrequencyMedians[i]

    def extractPeaks(self):
        """
        Extract peaks from the input spectrum.
        :return:
        """

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
        """
        Compute the bounding box of a peak by cardinal expansion.
        :param peak: The peak.
        :return:
        """
        L = peak.col
        T = peak.row
        R = peak.col
        B = peak.row

        data = self.processedData
        localThr = min(self.eventAmpThreshold * peak.value, self.peakThreshold)

        # cardinal expansion
        while L > 0                 and data[peak.row, L-1] > localThr:  L -= 1
        while T < data.shape[0] - 1 and data[T+1, peak.col] > localThr:  T += 1
        while R < data.shape[1] - 1 and data[peak.row, R+1] > localThr:  R += 1
        while B > 0                 and data[B-1, peak.col] > localThr:  B -= 1

        peak.boundingBox = BoundingBox(L=L, T=T, R=R, B=B)

    def mergeBoundingBoxes(self):
        """
        Perform iterations of merging peaks with overlapping bounding boxes.
        :return:
        """
        conflicts = []

        while True:
            # save the current state
            oldPeaks = self.peaks.copy()
            self.peaks = []

            # merge all overlapping peaks
            for conflict in conflicts:
                self.peaks.append(self.mergePeaks(oldPeaks[conflict[0]], oldPeaks[conflict[1]]))

            # fill in the rest of the peaks
            for peak_id, peak in enumerate(oldPeaks):

                # check that the peak has not already been merged
                peakMerged = False
                for conflict in conflicts:
                    if conflict[0] == peak_id or conflict[1] == peak_id:
                        peakMerged = True
                        break

                # add to the merged peaks list
                if peakMerged == False: 
                    self.peaks.append(peak)

            # check for conflicts
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
        """
        Merge two peaks (and their bounding boxes).
        :param peak1: The first peak.
        :param peak2: The second peak.

        :return: The new peak.
        """
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
        """
        Check if two bounding boxes overlap by a significant amount.
        :param rect1: The first bounding box.
        :param rect2: The second bounding box.
        :param overlapRatio: The minimum amount of overlap area for either bounding box.

        :return: True if they overlap and exceed the minimum amount.
        """
        x1 = max(rect1.L, rect2.L)
        y2 = min(rect1.T, rect2.T)
        x2 = min(rect1.R, rect2.R)
        y1 = max(rect1.B, rect2.B)

        # intersection
        if x2 >= x1 and y2 >= y1:
            rect1Area = (rect1.T - rect1.B) * (rect1.R - rect1.L)
            rect2Area = (rect2.T - rect2.B) * (rect2.R - rect2.L)
            intersectionArea = (x2-x1) * (y2-y1)

            if intersectionArea > overlapRatio * min(rect1Area, rect2Area):
                return True

        return False