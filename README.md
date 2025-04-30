## Data Context

The data capture correspond to individual wearing an IMU on their shin while walking on Centennial campus. Their gait motion was captured while they were standing, walking on solid even terrain and softer uneven terrain (grass in our case), and climbing up and down stairs. This data was captured to train a gait recognition model to aid on the development of robotic prosthesis.

## Dataset Description

Here is a brief description of the data files:

  - ".x.v" files contain the xyz accelerometers and xyz gyroscope measurements from the lower limb.
  - ".x.t" files contain the time stamps for the accelerometer and gyroscope measurements. The units are in seconds and the sampling rate is 40 Hz.
  - ".y.v" files contain the labels. (0) indicates standing or walking in solid ground, (1) indicates going down the stairs, (2) indicates going up the stairs, and (3) indicates walking on grass.
  - ".y.t" files contain the time stamps for the labels. The units are in seconds and the sampling rates is 10 Hz.

=====================================
## Proposed Framework:

Using windows of size 6xN, the windows are strided over the time series and capture a matrix with the formentioned dimension having a label corresponding to the last timestamp related to the window. Multiple training
samples are gathered using this approach and a CNN with a specific architecture is trained.
