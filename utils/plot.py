# This code is used to plot the 3d graphs of all the videos

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
results_folder = '/content/drive/MyDrive/BTPK/Deep-Fish-Tracker-Network/Result-old/'
data_folder = '/content/drive/MyDrive/BTPK/Deep-Fish-Tracker-Network/TrackerData/'
save_folder = '/content/drive/MyDrive/BTPK/Deep-Fish-Tracker-Network/Videos/'
files = os.listdir(data_folder)

for file in files:
    file = file.split('.')[0]

    f = pyplot.figure()
    ax = f.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('time(frame)')
    data = pd.read_csv(results_folder+file+'.txt',names=['frame','track','x','y','w','h','c','wx','wy','wz'],header=None)
    seen = set()
    print(data)
    for i in range(len(data['track'])):
        if data['track'][i] not in seen:
            seen.add(data['track'][i])
            track_data = data[data['track'] == data['track'][i]]
            ax.plot(track_data['x'],track_data['y'],track_data['frame'])


    pyplot.savefig(save_folder+file+'.png')
