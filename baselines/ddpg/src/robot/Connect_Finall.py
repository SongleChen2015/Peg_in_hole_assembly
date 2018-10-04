# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     Connect
   Description :   The class for control the robot; especially, the force sensor need to be calibrated firstly
   and test the detected force and moment then.
   Author :       Zhimin Hou
   date：         18-1-7
-------------------------------------------------
   Change Activity:
                   18-1-7
-------------------------------------------------
"""

import socket
import time
import numpy as np
HOST = '192.168.125.1'
PORT = 1502

class Robot_Control(object):
    def __init__(self):
        """The Matrix of Peg and hole"""
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        """The hole in world::::Tw_h=T*Tt_p"""
        self.Tw_h = np.array([[0.9941, -0.1064, 0.0218, 531.3748],
                              [0.1064, 0.9943, -0.0032, -44.1109],
                              [-0.0213, 0.0055, 0.9998, 70.3943],
                              [0, 0, 0, 1.0000]])

        # self.Tw_h = np.array([[9.94127729e-01,  -1.06342739e-01,   2.00328825e-02,   5.31604700e+02],
        #                       [1.06426394e-01,   9.94315580e-01,  -3.16269280e-03,  -4.43183873e+01],
        #                       [-1.95826778e-02,   5.26775395e-03,   9.99794319e-01,   7.06768898e+01],
        #                       [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

        # self.Tw_h = np.array([[9.93591091e-01, -1.09760991e-01, 2.70058995e-02, 5.30701537e+02],
        #                       [1.09886144e-01, 9.93939061e-01, -3.19878826e-03, -4.43103953e+01],
        #                       [-2.64911163e-02, 6.13746895e-03, 9.99630156e-01, 7.13090837e+01],
        #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.Tw_h = np.array([[9.94045497e-01, - 1.06350680e-01, 2.37302517e-02, 5.31604700e+02],
        #                       [1.06456211e-01, 9.94312159e-01, - 3.23400662e-03, -4.43183873e+01],
        #                       [-2.32513390e-02, 5.73258896e-03, 9.99713167e-01, 7.06768898e+01],
        #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        # self.Tw_h = np.array([[9.93966257e-01, -1.06298732e-01, 2.70500831e-02, 5.31283487e+02],
        #                       [1.06425708e-01, 9.94315198e-01, - 3.30302042e-03, - 4.41046845e+01],
        #                       [-2.65452019e-02, 6.15352237e-03, 9.99628623e-01, 7.11462325e+01],
        #                       [0.0, 0.0, 0.0, 1.0000]])

        # self.Tw_h = np.array([[9.94045497e-01, - 1.06350680e-01, 2.37302517e-02, 5.31279654e+02],
        #                       [1.06456211e-01, 9.94312159e-01, - 3.23400662e-03, - 4.43022530e+01],
        #                       [-2.32513390e-02, 5.73258896e-03, 9.99713167e-01, 7.14128785e+01],
        #                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        """The peg in tool"""
        self.Tt_p = np.array([[0.992324636460983, -0.123656366940134, -0.000957450195690770, 1.41144492938517],
                             [-0.123650357439767, -0.992313999324900, 0.00485764354118817, -0.848850965882200],
                             [-0.00155076978095379, -0.00469357455783674, -0.999987743210303, 129.363218736061],
                             [0, 0, 0, 1]])

        """"""
        self.T_tt = np.array([[1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 180],
                              [0, 0, 0, 1]])

        """Tw_p=T*Tt_p"""
        self.Tw_t_0 = np.array([[-0.059425229, - 0.984803643, 0.163188318, 102.360878],
                               [-0.336890243, 0.173671486, 0.925388124, 580.456787],
                               [-0.939666753, 1.48E-05, - 0.342091206, 490.809845],
                               [0, 0, 0, 1]])

        """Start_position:[5.328795780000000e+02, -44.728722000000000, 2.007725980000000e+02]"""
        """Start_euler [1.798860170000000e+02,1.119398000000000,-0.993206000000000]"""

        self.Connect()

    """Connect with robot"""
    def Connect(self):
        self.s.connect((HOST, PORT))

    """Disconnect"""
    def DisConnect(self):
        self.s.close()

    """move line to the target"""
    def MoveToolTo(self, position, euler, vel):

        # for i in range(3):
        #     if abs(position[i])< 0.001:
        #         position[i] = 0.
        #     if abs(euler)
        # change euler to quter
        Q = self.EulerToQuternion(euler)
        # send the code head
        swrite = '#FileHead@'
        self.s.send(swrite.encode())
        Filecounter = 0

        # The send module
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'MODULE movproc' + '@'
        self.s.send(swrite.encode())

        # The target point
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'CONST robtarget Target_1000:=[[' + str('%0.5f' %position[0]) + ',' + \
                 str('%0.5f' %position[1]) + ',' + str('%0.5f' %position[2]) + '],' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + '[' + str('%0.5f' %Q[0]) + ',' + str('%0.5f' %Q[1]) +\
                 ',' + str('%0.5f' %Q[2]) + ',' + str('%0.5f' %Q[3]) + '],' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + '[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,0]];' + '@'
        self.s.send(swrite.encode())

        # The start of program
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'PROC Path_10()' + chr(10) + '@'
        self.s.send(swrite.encode())

        # SingArea \Wrist
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'SingArea  ' + chr(92) + 'Wrist;' + '@'
        self.s.send(swrite.encode())

        # ConfL \off
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'ConfL ' + chr(92) + 'Off;' + '@'
        self.s.send(swrite.encode())

        # Move intruction
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'MoveL Target_1000,userspeed' + chr(92) + 'V:=' +\
                  str('%.5f' %vel) + ',z100,Tool0' + chr(92) + 'WObj:=wobj0;' + chr(10) + '@'
        self.s.send(swrite.encode())

        # Move Finish
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + ' MovtionFinish;' + '@'
        self.s.send(swrite.encode())

        # Error_Move_Finish
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ERROR' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + ' MovtionFinish;' + '@'
        self.s.send(swrite.encode())

        # The finish of code
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ENDPROC' + '@'
        self.s.send(swrite.encode())

        # The finish of Module
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ENDMODULE' + '@'
        self.s.send(swrite.encode())

        # Send the finish of the document
        Filecounter += 1
        swrite = '#FileEnd@'
        self.s.send(swrite.encode())

        # Receive the message of the robot and send the instruct to control the robot
        recvbuf = ''
        time_out = 0
        while recvbuf.find('Receive Over!') == -1 and time_out < 20:
            recvbuf = self.s.recv(2048).decode()
            time_out +=1
            time.sleep(0.1)

        if time_out < 20:
            result = True
        else:
            result = False
            print('Receive Fail!\n')
            return result

        self.s.send('#WorkStart@'.encode())

        # Wait the message of the finish signal of the motion work
        recvbuf = ''
        time_out = 0
        while recvbuf.find('MotionFinish') == -1 and time_out < 100:
            recvbuf = self.s.recv(2048).decode()
            time_out += 1
            time.sleep(0.1)

        if time_out < 20:
            result = True
        else:
            result = False
            print('Move Tool Fail!\n')

    """move joint to the target"""
    def MoveJointTo(self, position, vel):
        # send the code head
        swrite = '#FileHead@'
        self.s.send(swrite.encode())
        Filecounter = 0

        # The send module
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'MODULE movproc' + chr(10) + '@'
        self.s.send(swrite.encode())

        # The target point
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'CONST jointtarget Target_1000:=' + '@'
        self.s.send(swrite.encode())
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + '[[' + str('%.5f', position[0]) +\
                 ',' + str('%.5f', position[1]) + ',' + str('%.5f', position[2]) + ',' + str('%.5f', position[3]) +\
                 ',' + str('%.5f', position[4]) + ',' + str('%.5f', position[5]) + '],' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + '[0,9E9,9E9,9E9,9E9,9E9]];' + '@'
        self.s.send(swrite.encode())

        # The beginning of program
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'PROC Path_10()' + chr(10) + '@'
        self.s.send(swrite.encode())

        # Move instruct
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + chr(9) + 'MoveAbsj Target_1000,userspeed' +\
                 chr(92) + 'V:=' + str('%.5f', vel) + ',z100,Tool0' + chr(92) + 'WObj:=wobj0;' + chr(10) + '@'
        self.s.send(swrite.encode())

        # MoveFinish
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + ' MovtionFinish;' + '@'
        self.s.send(swrite.encode())

        # ERROR_MovtionFinish
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ERROR' + '@'
        self.s.send(swrite.encode())

        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + ' MovtionFinish;' + '@'
        self.s.send(swrite.encode())

        # The end of the code
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' ' + 'ENDPROC' + '@'
        self.s.send(swrite.encode())

        # The end of the module
        Filecounter += 1
        swrite = '#FileData ' + str(Filecounter) + ' '  'ENDMODULE' '@'
        self.s.send(swrite.encode())

        # send the end of the document
        swrite = '#FileEnd@'
        self.s.send(swrite.encode())

        # Receive the message of the robot and send the instruct to control the robot
        recvbuf = ''
        time_out = 0
        while recvbuf.find('Receive Over!') == -1 and time_out < 20:
            recvbuf = self.s.recv()
            time_out += 1
            time.sleep(0.1)

        if time_out < 20:
            result = True
        else:
            result = False
            print('Receive Fail!\n')
            return result

        self.s.send('#WorkStart@')

        # Wait the message of the finish signal of the motion work
        recvbuf = ''
        time_out = 0
        while recvbuf.find('MotionFinish') == -1 and time_out < 100:
            recvbuf = self.s.recv(2048).decode()
            time_out += 1
            time.sleep(0.1)

        if time_out < 20:
            result = True
        else:
            result = False
            print('Move Tool Fail!\n')

    """"""
    # def Rotate_x(self):

    """"excute the aligh action for the pegs and holes"""
    def Align_PegHole(self):

        Tt_p = np.array([[0.992324636460983, -0.123656366940134, -0.000957450195690770, 1.41144492938517],
                        [-0.123650357439767, -0.992313999324900, 0.00485764354118817, -0.848850965882200],
                        [-0.00155076978095379, -0.00469357455783674, -0.999987743210303, 129.363218736061],
                        [0, 0, 0, 1]])

        # Get the current position, euler and matrix T
        _, _, Tw_t = self.GetCalibTool()
        Tw_p = np.dot(Tw_t, Tt_p)
        Th_p = np.dot(np.linalg.inv(self.Tw_h), Tw_p)

        # Calculate the position and euler of pegs
        Tw_p_2 = self.Tw_h
        Tw_p_2[2, 3] = Tw_p[2, 3]
        dT = np.dot(np.linalg.inv(Tw_p), Tw_p_2)
        dT_2 = np.dot(np.dot(Tt_p, dT), np.linalg.inv(Tt_p))
        Tw_t2 = np.dot(Tw_t, dT_2)
        [position_t2, eulerang_t2] = self.MatrixToEuler(Tw_t2)

        # Get the force of the pegs
        myForceVector = self.GetFCForce()
        # print(myForceVector)

        # Move the pegs to target
        self.MoveToolTo(position_t2, eulerang_t2, 30)

        return position_t2, eulerang_t2

    """require the position and eluer current"""
    def GetCalibTool(self):
        recvbuf = ''
        Euler = np.zeros(3, dtype=float)
        Position = np.zeros(3, dtype=float)

        self.s.send('#GetCalibPar@'.encode())
        time.sleep(0.3)

        time_out = 0
        while len(recvbuf) < 70 and time_out < 30:
            recvbuf += self.s.recv(2048).decode()
            time_out += 1
            time.sleep(0.1)

        if time_out >= 30:
            print('Get Tool fail!\n')
            result = False
            return result
        # print(recvbuf)
        px = recvbuf.find('PX')
        endnum_px = recvbuf.find('*', px)
        py = recvbuf.find('PY')
        endnum_py = recvbuf.find('*', py)
        pz = recvbuf.find('PZ')
        endnum_pz = recvbuf.find('*', pz)
        ex = recvbuf.find('EX')
        endnum_ex = recvbuf.find('*', ex)
        ey = recvbuf.find('EY')
        endnum_ey = recvbuf.find('*', ey)
        ez = recvbuf.find('EZ')
        endnum_ez = recvbuf.find('*', ez)

        Position[0] = float(recvbuf[(px + 2):endnum_px - 1])
        Position[1] = float(recvbuf[(py + 2):endnum_py - 1])
        Position[2] = float(recvbuf[(pz + 2):endnum_pz - 1])
        Euler[0] = float(recvbuf[(ex + 2):endnum_ex - 1])
        Euler[1] = float(recvbuf[(ey + 2):endnum_ey - 1])
        Euler[2] = float(recvbuf[(ez + 2):endnum_ez - 1])

        # Euler = Euler*np.pi/180

        # T = np.zeros((4, 4), dtype=float)
        # T[0, 0] = np.cos(Euler[2]) * np.cos(Euler[1])
        # T[0, 1] = np.cos(Euler[2]) * np.sin(Euler[1]) * np.sin(Euler[0]) - np.sin(Euler[2]) * np.cos(Euler[0])
        # T[0, 2] = np.cos(Euler[2]) * np.sin(Euler[1]) * np.cos(Euler[0]) - np.sin(Euler[2]) * np.sin(Euler[0])
        # T[2, 1] = np.sin(Euler[2]) * np.cos(Euler[1])
        # T[2, 2] = np.sin(Euler[2]) * np.sin(Euler[1]) * np.sin(Euler[0]) + np.cos(Euler[2]) * np.cos(Euler[0])
        # T[2, 3] = np.sin(Euler[2]) * np.sin(Euler[1]) * np.cos(Euler[0]) - np.cos(Euler[2]) * np.sin(Euler[0])
        # T[3, 1] = -np.sin(Euler[1])
        # T[3, 2] = np.cos(Euler[1]) * np.sin(Euler[0])
        # T[3, 3] = np.cos(Euler[1]) * np.cos(Euler[0])
        # T[0, 3] = Position[0]
        # T[1, 3] = Position[1]
        # T[2, 3] = Position[2]

        T = self.EulerToMatrix(Position, Euler)

        return Position, Euler, T

    """require the force and moment"""
    def GetFCForce(self):
        recvbuf = ''
        myForceVector = np.zeros(6, dtype=float)
        self.s.send('#GetFCForce@'.encode())
        time.sleep(0.3)

        time_out = 0
        ind = 1

        while len(recvbuf) < 70 and time_out < 30:
            recvbuf += self.s.recv(2048).decode() # join decode()
            time_out += 1
            time.sleep(0.1)

        if time_out < 30:
            print('Get Force Success!\n')
        else:
            result = False
            print('Get Force Fail!\n')
            return result
        # print(recvbuf)
        fx = recvbuf.find('Fx')
        # print(fx)
        endnum_fx = recvbuf.find('*', fx)
        # print(endnum_fx)
        fy = recvbuf.find('Fy')
        # print(fy)
        endnum_fy = recvbuf.find('*', fy)
        # print(endnum_fy)
        fz = recvbuf.find('Fz')
        endnum_fz = recvbuf.find('*', fz)
        tx = recvbuf.find('Tx')
        endnum_tx = recvbuf.find('*', tx)
        ty = recvbuf.find('Ty')
        endnum_ty = recvbuf.find('*', ty)
        tz = recvbuf.find('Tz')
        endnum_tz = recvbuf.find('*', tz)

        myForceVector[0] = float(recvbuf[(fx + 2):endnum_fx - 1])
        # print(recvbuf[(fx + 2):endnum_fx])
        # print(recvbuf[(fx + 2):endnum_fx - 1])
        myForceVector[1] = float(recvbuf[(fy + 2):endnum_fy - 1])
        myForceVector[2] = float(recvbuf[(fz + 2):endnum_fz - 1])
        myForceVector[3] = float(recvbuf[(tx + 2):endnum_tx - 1])
        myForceVector[4] = float(recvbuf[(ty + 2):endnum_ty - 1])
        myForceVector[5] = float(recvbuf[(tz + 2):endnum_tz - 1])

        return  myForceVector

    """calibration the force controller"""
    def CalibFCforce(self):
        swrite = '#CalibFCForce@"'
        self.s.send(swrite.encode())
        CalibResult = True
        return CalibResult

    """"""
    def EulerToMatrix(self, position, euler):

        euler = euler * np.pi/180
        Wx = euler[0]
        Wy = euler[1]
        Wz = euler[2]
        Mz = np.array([[np.cos(Wz), -np.sin(Wz), 0, 0],
                       [np.sin(Wz), np.cos(Wz), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        My = np.array([[np.cos(Wy), 0, np.sin(Wy), 0],
                       [0, 1, 0, 0],
                       [-np.sin(Wy), 0, np.cos(Wy), 0],
                       [0, 0, 0, 1]])
        Mx = np.array([[1, 0, 0, 0],
                       [0, np.cos(Wx), -np.sin(Wx), 0],
                       [0, np.sin(Wx), np.cos(Wx), 0],
                       [0, 0, 0, 1]])
        Matrix_three = np.dot(np.dot(Mz, My), Mx)  # Mz * My * Mx
        Unit = np.array([[1, 0, 0, position[0]],
                         [0, 1, 0, position[1]],
                         [0, 0, 1, position[2]],
                         [0, 0, 0, 1]])
        Matrix = np.dot(Unit, Matrix_three)

        return Matrix

    """"""
    def EulerToQuternion(self, euler):

        euler = euler * np.pi/180
        x1 = np.cos(euler[1]) * np.cos(euler[2])
        x2 = np.cos(euler[1]) * np.sin(euler[2])
        x3 = -np.sin(euler[1])

        y1 = -np.cos(euler[0]) * np.sin(euler[2]) + np.sin(euler[0]) * np.sin(euler[1]) * np.cos(euler[2])
        y2 = np.cos(euler[0]) * np.cos(euler[2]) + np.sin(euler[0]) * np.sin(euler[1]) * np.sin(euler[2])
        y3 = np.sin(euler[0]) * np.cos(euler[1])

        z1 = np.sin(euler[0]) * np.sin(euler[2]) + np.cos(euler[0]) * np.sin(euler[1]) * np.cos(euler[2])
        z2 = -np.sin(euler[0]) * np.cos(euler[2]) + np.cos(euler[0]) * np.sin(euler[1]) * np.sin(euler[2])
        z3 = np.cos(euler[0]) * np.cos(euler[1])

        Q = np.zeros(4, dtype=float)
        Q[0] = np.sqrt(x1 + y2 + z3 + 1)/2
        if y3 > z2:
            Q[1] = np.sqrt(x1 - y2 - z3 + 1)/2
        else:
            Q[1] = -np.sqrt(x1 - y2 - z3 + 1)/2

        if z1 > x3:
            Q[2] = np.sqrt(y2 - x1 - z3 + 1)/2
        else:
            Q[2] = -np.sqrt(y2 - x1 - z3 + 1)/2

        if x2 > y1:
            Q[3] = np.sqrt(z3 - x1 - y2 + 1)/2
        else:
            Q[3] = -np.sqrt(z3 - x1 - y2 + 1)/2

        return Q

    """calculate the traslation Matrix T with the position and euler"""
    def MatrixToEuler(self, T):
        Position = np.zeros(3, dtype=float)
        Position[0] = T[0, 3]
        Position[1] = T[1, 3]
        Position[2] = T[2, 3]

        Euler = np.zeros(3, dtype=float)
        Euler[2] = np.arctan2(T[1, 0], T[0, 0])
        Euler[1] = np.arctan2(-T[2, 0], np.cos(Euler[2]) * T[0, 0] + np.sin(Euler[2]) * T[1, 0])
        Euler[0] = np.arctan2(np.sin(Euler[2]) * T[0, 2] - np.cos(Euler[2]) * T[1, 2],
                         -np.sin(Euler[2]) * T[0, 1] + np.cos(Euler[2]) * T[1, 1])

        Euler = Euler * 180 / np.pi

        return Position, Euler


def Test_calibrate():

    """define a robot_control class"""
    Controller = Robot_Control()

    """Test the detected force and check whether the force approach to zero"""
    force = Controller.CalibFCforce()
    print(force)

    """"""
    position_t2, eulerang_t2 = Controller.Align_PegHole()
    print(position_t2)
    print(eulerang_t2)
    force = Controller.GetFCForce()
    print(force)

    """used to search the initial position and euler; please note the other code"""
    init_position = np.array([6.328895870000000e+02, -44.731415000000000, 3.497448430000000e+02])
    init_euler = np.array([1.798855440000000e+02, 1.306262000000000, -0.990207000000000])
    position, euler, T = Controller.GetCalibTool()
    Controller.MoveToolTo(position +[0., 0., 200.], euler + [0., 0.1, 0.1], 3)
    Controller.MoveToolTo(init_position, init_euler, 3)
    print(position)
    print(euler)
    print(np.dot(T, Controller.Tt_p))


if __name__ == "__main__":
    Test_calibrate()