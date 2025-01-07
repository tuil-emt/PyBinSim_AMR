import logging
import socket
import threading
import numpy as np
import re
import copy
import datetime

from scipy.spatial.transform import Rotation

# container for tracking data from a single tracked body
# will be populated with raw tracking data in tracking space coordinate system 

class ListenerTrackingData:

    def __init__(self):
        self.frame = 0
        self.timeStamp = 0.0
        self.localPos = np.zeros(3)
        self.localRot = np.identity(3)

    def __str__(self):
        return f"ListenerTrackingData(frame='{self.frame}',timeStamp='{self.timeStamp}',localPos='{self.localPos}',localRot='{self.localRot})"

# calculate latency from when tracking data was captured
# time stamp is assumed to be the number of seconds since UTC midnight

def calculate_latency_sec(timeStamp):
    current_time = datetime.datetime.utcnow().time()
    seconds_since_midnight = (current_time.hour * 3600) + (current_time.minute * 60) + current_time.second + (current_time.microsecond / 1000000)
    latency = seconds_since_midnight - timeStamp
    return latency


# receives tracking data from tracking controller 
# parses tracking data for single tracked body with given ID (intended to be used for tracking head position of listener)

class DTrackInterface(object):
    def __init__(self, location, user):
        self.log = logging.getLogger("pybinsim.Filter")

        # location, user = self.load_location_and_user_data()
        self.body = user.headTrackingID - 1

        self.globalTranslationMat = np.eye(4)
        self.globalTranslationMat[:3, 3] = location.globalTranslation
        
        globalRotationMatQ = Rotation.from_euler('xyz', location.globalRotation, degrees=True).as_quat()
        globalRotationMat3x3 = Rotation.from_quat(globalRotationMatQ).as_matrix()
        self.globalRotationMat = np.eye(4)
        self.globalRotationMat[:3,:3] = globalRotationMat3x3

        self.log.info("DTrackInterface: Opening UDP socket on port " + str(location.dTrackPort))
        self.log.info("DTrackInterface: listening for tracking data for tracked body " + str(self.body + 1) + " (1-indexed ID)")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)
        self.ip = ''
        self.sock.bind((self.ip, location.dTrackPort))

        self.recv_buffer_size = 4096
        self.recv_buffer = bytearray(self.recv_buffer_size)

        self.current_tracking_data = ListenerTrackingData()

        self.tracking_data_lock = threading.Lock()

        self.run_receive_thread = True;
        self.rcv_thread = threading.Thread(target=self.receive_tracking_data_loop)
        self.rcv_thread.deamon = True
        self.rcv_thread.start()

    def receive_tracking_data_loop(self):

        while self.run_receive_thread:
            try:
                msg_length_bytes = self.sock.recv_into(self.recv_buffer)

                received_tracking_data = ListenerTrackingData()

                success = self.parse(self.recv_buffer[:msg_length_bytes], received_tracking_data)

                if (success):
                    with self.tracking_data_lock:
                        self.current_tracking_data = received_tracking_data

                    # print("Latency: " + str(calculate_latency_sec(self.current_tracking_data.timeStamp)))
                    # print("Received frame " + str(self.current_tracking_data))

            except socket.timeout as e:
                print("DTrackInterface::receive_tracking_data_loop error:", str(e))


    def parse(self, msg_bytes, received_tracking_data):
        msg_str = msg_bytes.decode('utf-8')  # You can specify the appropriate encoding
        lines = msg_str.splitlines()
        found_body = False
        bodies_found = []

        for line in lines:

            if line.startswith("fr"):
                splitline = line.split()
                received_tracking_data.frame = int(splitline[1])
            elif line.startswith("ts"):
                splitline = line.split()
                received_tracking_data.timeStamp = float(splitline[1])
            elif line.startswith("6d"):
                # line contains all 6dof body data

                # split line by bracket contents
                bracket_contents = re.findall(r'\[(.*?)\]', line)

                # each group of 3 sets of brackets contains body pose
                for b in range(int(len(bracket_contents) / 3)):

                    # get body id (first number in first set of brackets)
                    bodyId = int(bracket_contents[b * 3].split()[0])
                    bodies_found.append( (bodyId + 1) )

                    if bodyId != self.body:
                        continue

                    found_body = True

                    # get position
                    pos_and_euler_angles = bracket_contents[b * 3 + 1].split()
                    pos = []
                    for i in range(3):
                        pos.append(float(pos_and_euler_angles[i])) 
                    received_tracking_data.localPos = np.array(pos)


                    rot_matrix = bracket_contents[b * 3 + 2].split()
                    rot = []
                    for i in range(9):
                        rot.append(float(rot_matrix[i])) 

                    # rotation matrix from dtrack is column-major
                    received_tracking_data.localRot = np.array(rot).reshape(3,3).transpose()


        if received_tracking_data.frame == 0:
            self.log.info("DTrackInterface: Could not parse frame ID")
            return False
        if received_tracking_data.timeStamp == 0.0:
            self.log.info("DTrackInterface: Could not parse timestamp")
            return False
        if not found_body:
            self.log.info("DTrackInterface: Did not detect body " + str(self.body + 1) + " (found " + str(bodies_found) + ", 1-indexed IDs)")
            return False

        return True


    def get_latest_listener_transform_matrix(self):
        
        with self.tracking_data_lock:
            latest_data = copy.deepcopy(self.current_tracking_data)
        
        # convert from tracking space to unity space
        # https://github.com/ar-tracking/UnityDTrackPlugin/blob/master/Source/Util/Transforms.cs#L59
        localTranslation = latest_data.localPos * np.array([0.001, 0.001, -0.001])
        
        localRotationMat3x3 = Rotation.from_matrix(latest_data.localRot)
        rot_q = localRotationMat3x3.as_quat()
        # quat format:  (x, y, z, w)
        # dtrack sdk for unity says to negate z and scalar https://github.com/ar-tracking/UnityDTrackPlugin/blob/master/Source/Util/Transforms.cs#L93
        # but this appears to be the way that produces the same matrices:
        rot_q_unity = np.array( [ rot_q[0], rot_q[1], -rot_q[2], -rot_q[3]] )

        localRotationMat3x3 = Rotation.from_quat(rot_q_unity).as_matrix()
        localRotationMat = np.eye(4)
        localRotationMat[:3,:3] = localRotationMat3x3

        # calculate listener transform as 4x4 matrix
        localTranslationMat = np.eye(4)
        localTranslationMat[:3, 3] = localTranslation

        listenerTransform = self.globalTranslationMat @ self.globalRotationMat @ localTranslationMat @ localRotationMat

        return listenerTransform


    def close(self):
        self.log.info('DTrackInterface: close()')
        self.run_receive_thread = False
        self.rcv_thread.join(timeout=3)
        self.sock.close();
        self.log.info('DTrackInterface: close() completed')
