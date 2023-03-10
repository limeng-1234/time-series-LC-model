# time-series-LC-model

## The LSTM_1 module contains two parts: 1)highD data processing and 2)modeling of LC decision based on LSTM network

## Attention_1 module is modeling of LC decision based on attention network


We mainly use the following information (as shown in Table \ref{VARIABLE DESCRIPTION_1}) in the highD dataset, vehicle id (id), The lane where the vehicle is located (laneId), time stamp (frame, 40ms per frame), longitudinal/lateral position (x/y), longitudinal/lateral speed (xVelocity/yVelocity). Fig. \ref{new2} shows the trajectory of some of the LC vehicles, where we record the moment of LC decision completion (LC Time) based on the laneId of the vehicle. The LC point (as shown by the red dot) records the LC decision completion moment (when the vehicle arrives here, it means that the vehicle has in fact made the lane change). We use the vehicle state before LC point as input data for the LC decision modeling. Fig. \ref{new3} shows the trajectories of two LH (laneId not changed) vehicles. From fig. \ref{new3} (a), we can find that although its laneId has not changed, however, its lateral velocity is larger, so the trajectory is highly likely to be a LC trajectory. Therefore, in order to avoid introducing abnormal data in the training process, this paper eliminates this type of data when extracting the LH data. We use the vehicle states that are similar to the trajectory shown in Fig. \ref{new3} (b) as the data for LH decision.

We set the time window $tw$ of input data to 10, and it should be noted that each time interval is 0.16 s (to reduce the complexity of model training, we extract a set of data every 4 frames), so the actual time-series is 1.6 s. The decision targe is the value $c \in \{0, 1\} = \mathbb C $ is 0 (1) for LH (LC) decision, resulting in a total of 772473 sets of time-series data. In this work, instead of normalizing the data, we subtract the data feature values at $tw$ moments from the time-series data.
