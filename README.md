# An Investigation of Ensemble Methods in Emotion Recognition
In this work, we present unimodal systems based on audio for continuous recognition of human
affect. Automatic analysis of human affective state has increased popularity over the last few
years. Particularly, predicting the continuous level of affect of a person based on his or her
voice has higher performance in real-world experiments compared to discrete level of affect.
However, the variation across different people i.e evaluators and subjects, along with variation
across different audio capture condition, cause disparity in the annotation.
This lack of robustness motivated us to investigate the impact of different ratings for emotional
data. Therefore, we employ an ensemble of regressors to leverage the robustness of ensemble
method. For this purpose, we investigate the impact of Dynamic Selection in emotion recognition and the adaptability 
of the database used, RECOLA, in dynamic selection techniques. We
also compared differently designed systems for decision level fusion of the regressors; there are
three approaches proposed for integration step, namely, linear regression (as fusion decision),
arithmetic mean and a weighted average based on the merits of each learner given by dynamic
selection oracle. Furthermore, we used powerful learning algorithms such as Long Short-Term
Memory (LSTM) and Support Vector Regression (SVR) to learn from data.
The promising result in decision fusion approaches was obtained using the average between
homogeneous and heterogeneous regressors. This shows the effectiveness of this method, as
we achieved the highest accuracy by applying this approach. The result from Oracle in dynamic
selection also suggested that we may achieve higher performance with the use of different
ratings for prediction of emotion.

