import sys

import numpy as np

from common.extrema import find_extrema_2D
from structs import BoundingBox


class Blob:
    def __init__(self, ID, rowBounds, colBounds, center, pixelCount, mask, contour):
        self.ID = ID
        self.rowBounds = rowBounds
        self.colBounds = colBounds
        self.center = center
        self.pixelCount = pixelCount
        self.mask = mask
        self.contour = contour
        self.boundingBox = BoundingBox(colBounds[0], rowBounds[1], colBounds[1], rowBounds[0])


class BlobDetector:
    def __init__(self, image=None, valueThreshold=sys.float_info.min, pixelThreshold=20, style="None", computeContours=False):
        self.NoBlobID = 0
        self.DetectedBlobs = []
        if image is not None:
            self.detect(image, valueThreshold, pixelThreshold, style, computeContours)


    def detect(self, image, valueThreshold=sys.float_info.min, pixelThreshold=20, style="None", computeContours=False):
        self.IDs_image = np.full_like(image, self.NoBlobID)

        queue = []

        currentID = 1
        contour = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j] < valueThreshold or self.IDs_image[i, j] > self.NoBlobID:
                    continue

                queue.append((i, j))
                sumX=0
                sumY=0
                pixelCount=0
                maxX=0
                maxY=0
                minX = image.shape[0] + 1
                minY = image.shape[1] + 1

                while queue:
                    top = queue.pop(0)
                    self.IDs_image[top[0], top[1]] = currentID
                    pixelCount += 1

                    minX = top[0] if top[0] < minX else minX
                    minY = top[1] if top[1] < minY else minY
                    maxX = top[0] if top[0] > maxX else maxX
                    maxY = top[1] if top[1] > maxY else maxY
                    sumX += top[0]
                    sumY += top[1]

                    row_begin = max(top[0] - 1, 0)
                    row_end = min(top[0] + 1, image.shape[0] - 1)
                    col_begin = max(top[1] - 1, 0)
                    col_end = min(top[1] + 1, image.shape[1] - 1)

                    contourPoint = False
                    for k in range(row_begin, row_end+1):
                        for l in range(col_begin, col_end+1):
                            if self.IDs_image[k, l] > self.NoBlobID:
                                continue

                            if image[k, l] < valueThreshold:
                                contourPoint = True
                                continue

                            self.IDs_image[k, l] = currentID

                            queue.append((k, l))

                    if contourPoint and contour != []:
                        contour.append([top[0], top[1]])


                if pixelCount < pixelThreshold:
                    continue

                mask = self.get_mask(image, minX, minY, maxX-minX+1, maxY-minY+1, style)
                self.DetectedBlobs.append(
                    Blob(
                        ID=currentID,
                        rowBounds=(minX, maxX),
                        colBounds=(minY, maxY),
                        center=(sumX/pixelCount, sumY/pixelCount),
                        pixelCount=pixelCount,
                        mask=mask,
                        contour=contour
                    )
                )

                currentID += 1



    def get_mask(self, image, row, col, rowCount, colCount, style):
        if style == "real":
            result = np.zeros((rowCount, colCount))
            for i in range(rowCount):
                for j in range(colCount):
                    if self.IDs_image[row + i, col + j] > self.NoBlobID:
                        result[i, j] = 1

        elif style == "logical":
            return image[row:row+rowCount, col:col+colCount]

        else:
            return None



class Segment:
    def __init__(self, ID, level, blob, peakIDs, refPeakIDs):
        self.ID = ID
        self.level = level
        self.blob = blob
        self.peakIDs = peakIDs
        self.refPeakIDs = refPeakIDs


class Peak:
    def __init__(self, ID=0, segmentID=0, level=0, value=0, row=0, col=0):
        self.ID = ID
        self.segmentID = segmentID
        self.level = level
        self.value = value
        self.row = row
        self.col = col
        self.prominence = None
        self.referenceLevel = None
        self.referenceSegmentID = None
        self.coordinates = (row, col)



class TFPF:
    def __init__(self):
        self.blob_detector = BlobDetector()


    def slice_n_dice(self, sliceLevel):
        self.labels = np.zeros_like(self.quantization)
        self.final_peaks = []
        self.final_rects = []
        self.final_masks = []


        count = 0
        for i in range(len(self.segmentations)):
            if self.segmentations[i] is not None:
                break
            count+=1

        sliceLevel = max(count, min(len(self.level_thresholds)-1, sliceLevel))

        segmentation = self.segmentations[sliceLevel]
        labelCount=0
        for segment in self.segments:
            if segment.level != sliceLevel:
                continue

            maxPeak = Peak(value=self.input_range[0])
            for peak in self.peaks:
                if segmentation[peak.coordinates] == segment.ID and peak.value > maxPeak.value:
                    maxPeak = peak

            labelCount += 1

            mask = segment.blob.mask
            row_begin = segment.blob.rowBounds[0]
            col_begin = segment.blob.colBounds[0]

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] > 0:
                        self.labels[row_begin+i, col_begin+j] = labelCount

            self.final_peaks.append((maxPeak.row, maxPeak.col))
            self.final_rects.append(segment.blob.boundingBox)
            self.final_masks.append(segment.blob.mask)

        return labelCount



    def evaluate(self, input, quantizationMethod="linear", quantizationLevels=30, skipLevels=1):
        self.quantize(input, quantizationMethod, quantizationLevels)
        self.parse_image(input, skipLevels)
        self.compute_peak_references(skipLevels)


    def compute_peak_references(self, skipLevels):
        for peak in self.peaks:
            currentLevel = peak.level
            peak.prominence = self.get_prominence(peak.value, self.level_thresholds[currentLevel])
            peak.referenceLevel = currentLevel
            peak.referenceSegmentID = self.segmentations[currentLevel][peak.coordinates]

            while currentLevel > skipLevels:
                higherPeak = False

                for candidate in self.peaks:
                    candidateSegIDinLvl = self.segmentations[currentLevel][candidate.coordinates]

                    if candidateSegIDinLvl == peak.referenceSegmentID and candidate.value > peak.value:
                        higherPeak = True
                        break

                if higherPeak:
                    break

                peak.prominence = self.get_prominence(peak.value, self.level_thresholds[currentLevel])
                peak.referenceLevel = currentLevel
                peak.referenceSegmentID = self.segmentations[currentLevel][peak.coordinates]

                currentLevel -= 1

            self.segments[peak.referenceSegmentID].refPeakIDs.append(peak.ID)


    def get_prominence(self, peakValue, referenceLine):
        return peakValue - referenceLine


    def parse_image(self, input, skipLevels):

        self.segmentations = []
        self.segments = []
        self.peaks = []

        for iLevel in range(len(self.level_thresholds), 0, -1):
            if iLevel <= skipLevels:
                break

            self.blob_detector.detect(self.quantization, iLevel, style="logical")

            segmentation = np.zeros_like(input)
            for blob in self.blob_detector.DetectedBlobs:
                segment = Segment(
                    ID=len(self.segments)+1,
                    level=iLevel,
                    blob=blob,
                    peakIDs=[],
                    refPeakIDs=[]
                )

                for i in range(blob.mask.shape[0]):
                    for j in range(blob.mask.shape[1]):
                        if blob.mask[i, j] > 0:
                            segmentation[blob.rowBounds[0]+i, blob.colBounds[0]+j] = segment.ID

                self.segments.append(segment)

            self.segmentations.insert(0, segmentation)

            self.fill_mask(input, self.blob_detector.DetectedBlobs)
            _, maxima = find_extrema_2D(input, 1, 'and')


            for max in maxima:
                pkRow = max[0]
                pkCol = max[1]

                if self.quantization[pkRow, pkCol] == iLevel:
                    peak = Peak(
                        ID=len(self.peaks) + 1,
                        level=iLevel,
                        segmentID=segmentation[pkRow, pkCol],
                        value=input[pkRow, pkCol],
                        row=pkRow,
                        col=pkCol
                    )

                    self.segmentations[peak.segmentID].peakIDs.append(peak.ID)
                    self.peaks.append(peak)

        for iLevel in range(skipLevels, 0, -1):
            if iLevel <= 0:
                break
            m = np.zeros_like(input)
            m[0, 0] = -1
            self.segmentations.insert(iLevel, m)


    def fill_mask(self, input, blobs):
        self.masked_image = np.full_like(input, self.input_range[0])

        for blob in blobs:
            mask = blob.mask
            row_begin = blob.rowBounds[0]
            col_begin = blob.colBounds[0]

            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j] > 0:
                        self.masked_image[row_begin+i, col_begin+j] = input[row_begin+i, col_begin+j]


    def quantize(self, input, quantizationMethod, quantizationLevels):
        self.input_range = (np.amin(input), np.amax(input))

        #TODO distribution

        if quantizationMethod == "linear":
            self.level_thresholds = self.linear_levels(quantizationLevels)
        else:
            return None

        self.quantization = np.zeros_like(input)
        for q in self.level_thresholds:
            # c = 0
            for j in range(input.size):
                index = np.unravel_index(j, input.shape)
                if input[index] > q:
                    self.quantization[index] += 1
                    # c += 1




    def linear_levels(self, levels):
        return np.linspace(self.input_range[0], self.input_range[1], num=levels+1)