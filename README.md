# Classification of Lathe's Cutting Tool Wear Base on an Autonomous Machine Learning Model

Machining processes are of considerable significance to industries such as aviation, power generation, oil, and gas since a significant part of the industrial mechanical components went through a machining process during its manufacturing. Therefore, using worn cutting tools can lead to operational interruptions, accidents, and potential economic losses during these processes. Concerning these consequences, real-time monitoring can result in cost reduction, along with productivity and safety increase. This paper aims to discuss an autonomous model based on the Self-Organised Direction Aware Data Partitioning Algorithm (SODA) and machine learning techniques, including time series Feature Extraction based on Scalable Hypothesis tests (TSFRESH), to solve this problem. The model proposed in this work was tested in a data set recorded in a real machining system at the Manufacturing Processes Laboratory of the Federal University of Juiz de Fora (UFJF) in collaboration with the Laboratory of Industrial Automation and Computational Intelligence (LAIIC). This model can identify the patterns that distinguish the cutting tool operations as healthy or inadequate, achieving satisfactory performances in all cases presented in this work and potentially allowing to prevent faulty pieces fabrication.

### Dependencies

 - numpy
 - pandas
 - pickle
 - sklearn
 - datetime
 - matplotlib
 - tsfresh
 
### Files and Directories Overview

##### Directories

 - /Input
 - - This directory contains six different input datasets (see /Input/README.md for more information).
 - /Results
 - - This directory contains the figures and Classifier results
 
##### Files

 - lathes_model.py
 - - This python file contains the proposed model class.
 - SODA.py
 - - This python file contains the SODA algorithm.
 - model_example.ipynb
 - - This notebook file presents an example of the proposed model.
