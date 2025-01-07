import numpy as np
import re


msg_str = "fr 518169\r\nts 30331.359222\r\n6d 1 [45 1.000][-5.026 695.186 763.716 -7.7879 -10.9331 -3.3467][0.980175 -0.032183 0.195503 0.057318 0.990587 -0.124306 -0.189662 0.133047 0.972793]\r\n"

dtrackBody = 45

class ListenerTrackingData:

    def __init__(self):
        self.frame = 0
        self.timeStamp = 0.0
        self.localPos = np.zeros(3)
        self.localRot = np.identity(3)

    def __str__(self):
        return f"ListenerTrackingData(frame='{self.frame}',timeStamp='{self.timeStamp}',localPos='{self.localPos}',localRot='{self.localRot})"

trackingData = ListenerTrackingData()

lines = msg_str.splitlines()
for line in lines:

    if line.startswith("fr"):
        splitline = line.split()
        trackingData.frame = int(splitline[1])
    elif line.startswith("ts"):
        splitline = line.split()
        trackingData.timeStamp = float(splitline[1])
    elif line.startswith("6d"):
        # split line by bracket contents
        bracket_contents = re.findall(r'\[(.*?)\]', line)

        # get body id (first number in first set of brackets)
        bodyId = int(bracket_contents[0].split()[0])
        if bodyId != dtrackBody:
            continue
        
        # get position
        pos_and_euler_angles = bracket_contents[1].split()
        pos = []
        for i in range(3):
            pos.append(float(pos_and_euler_angles[i])) 
        trackingData.localPos = np.array(pos)


        rot_matrix = bracket_contents[2].split()
        rot = []
        for i in range(9):
            rot.append(float(rot_matrix[i])) 
        trackingData.localRot = np.array(rot).reshape(3,3)

print(trackingData)