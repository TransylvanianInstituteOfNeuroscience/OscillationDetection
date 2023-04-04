# OscillationDetection


## Project Structure: 
- "main.py" file for running the algorithms on example data <br />
- requirements.txt - general library management file, when a new library is added, this file will be updated with the library name and its version <br />
- (folder) algorithms	: folder that contains each of the 3 algorithms implemented in Python: OEvents, TFPF and TFBM each in its own .py file, additionally there is a structs.py file that contains structures used in the algorithms
- (folder) common		: contains utility functions, for example: functions for finding neighbours
- (folder) data			: location in which datasets will be stored (it is not uploaded to git due to large file sizes)
- (folder) plotting 	: scripts used to create the plots in the paper (unusable without the data)
- (folder) preprocess 	: contains functions for preprocessing the data, for example: scaling