import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pi = np.pi

class OnePeginHole(object):
    def __init__(self, transformParas):
        self.transformParas = transformParas
        self.diameterHole = 30
        self.diameterPeg = 29.9
        self.Center = [0, 0, 0]
        self.height = 100
        self.K = 200
        self.u = 0.01
        self.Centers_pegs, self.Centers_holes, self.pegs, self.holes = self.createPegInHoleModel()
        self.Centers_pegs, self.pegs = self.transformPegInHoleModel(self.Centers_pegs, self.pegs, self.transformParas)
        self.pegInHole = np.mat(self.calPegInHole(self.Centers_pegs))
        self.holeInPeg = self.pegInHole.I
        self.sumContactForceInPeg = self.getSumContactForceInPeg(self.Centers_pegs, self.Centers_holes, self.pegs, self.holes, self.diameterHole / 2,
                                                   self.diameterPeg / 2, self.K, self.u, self.holeInPeg)
        self.Position = self.MatrixToEuler(self.pegInHole)


    def ChangeOneStep(self, transformParas):
        self.Centers_pegs, self.pegs = self.transformPegInHoleModel(self.Centers_pegs, self.pegs, transformParas)
        self.pegInHole = np.mat(self.calPegInHole(self.Centers_pegs))
        self.holeInPeg = np.linalg.inv(self.pegInHole)
        self.Position = self.MatrixToEuler(self.pegInHole)
        self.sumContactForceInPeg = self.getSumContactForceInPeg(self.Centers_pegs, self.Centers_holes, self.pegs,
                                                                 self.holes, self.diameterHole / 2,
                                                                 self.diameterPeg / 2, self.K, self.u, self.holeInPeg)

        return self.sumContactForceInPeg, self.Position


    def MatrixToEuler(self, T):
        Position = np.zeros(6)
        Position[0] = T[0, 3]
        Position[1] = T[1, 3]
        Position[2] = T[2, 3]

        Position[5] = np.arctan2(T[1, 0], T[0, 0])*180/pi
        Position[4] = np.arctan2((-1)*T[2, 0], np.cos(Position[5])*T[0, 0]+np.sin(Position[5])*T[1, 0])*180/pi
        Position[3] = np.arctan2(np.sin(Position[5])*T[0, 2] - np.cos(Position[5]) * T[1, 2],  (-1)*np.sin(Position[5]) * T[0, 1] + np.cos(Position[5])*T[1, 1])*180/pi
        return Position


    def Cross(self, A, B):
        C = np.zeros(3)
        if np.isnan(A).any() == True or np.isnan(B).any() == True:
            A = np.nan_to_num(A)
            B = np.nan_to_num(B)

        if np.isinf(A).any() == True or np.isinf(B).any() == True:
            A = np.nan_to_num(A)
            B = np.nan_to_num(B)
        C[0] = A[1] * B[2] - A[2] * B[1]
        C[1] = A[2] * B[0] - A[0] * B[2]
        C[2] = A[0] * B[1] - A[1] * B[0]
        return C

    ## creat single circle
    def createCircle(self, center, radius):
        theta = np.arange(0, 2 * pi, 2 * pi / 20000)

        circle = np.zeros([3, theta.size])
        circle[0, :] = radius * np.cos(theta) + np.repeat(center[0], theta.size)
        circle[1, :] = radius * np.sin(theta) + np.repeat(center[1], theta.size)
        circle[2, :] = np.repeat(center[2], theta.size)
        return circle

    ## creat one hole and one peg model
    def createPegInHoleModel(self):
        diameterHole = self.diameterHole
        diameterPeg = self.diameterPeg
        Center = self.Center
        height = self.height
        Centers_pegs = np.array([[Center[0], Center[0], 30., 0.],
                                 [Center[1], Center[1], 0., 30.],
                                 [2 * height + 1, height + 1, 2 * height + 1, 2 * height + 1]])
        # Centers_pegs = np.array([[Center[0], Center[1], 2 * height + 1],
        #                          [Center[0], Center[1], height + 1],
        #                          [30., 0., 2 * height + 1],
        #                          [0., 30., 2 * height + 1]])
        # Centers_holes = np.array([[Center[0], Center[1], height],
        #                           [Center[0], Center[1], 0]])
        Centers_holes = np.array([[Center[0], Center[0]],
                                  [Center[1], Center[1]],
                                  [height, 0]])
        pegs = np.zeros((2, 3, 20000))
        holes = np.zeros((2, 3, 20000))
        for i in range(2):
            pegs[i] = self.createCircle(Centers_pegs[:, i], diameterPeg / 2)
            holes[i] = self.createCircle(Centers_holes[:, i], diameterHole / 2)

        return Centers_pegs, Centers_holes, pegs, holes

    ## plot the model
    def plot(self, Centers_pegs, Centers_holes, pegs, holes):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        axis_z = Centers_pegs[:, 0:2]
        axis_x = np.array([Centers_pegs[:, 0],
                           Centers_pegs[:, 2]]).transpose()
        axis_y = np.array([Centers_pegs[:, 0],
                           Centers_pegs[:, 3]]).transpose()

        ax.plot(axis_x[0, :], axis_x[1, :], axis_x[2, :], c='g', label='parametric curve')
        ax.plot(axis_y[0, :], axis_y[1, :], axis_y[2, :], c='y', label='parametric curve')
        ax.plot(axis_z[0, :], axis_z[1, :], axis_z[2, :], c='r', label='parametric curve')
        ax.plot(Centers_holes[0, :], Centers_holes[1, :], Centers_holes[2, :], c='b', label='parametric curve')
        for i in range(2):
            ax.plot(pegs[i][0, :], pegs[i][1, :], pegs[i][2, :], c='r', label='parametric curve')
            ax.plot(holes[i][0, :], holes[i][1, :], holes[i][2, :], c='b', label='parametric curve')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        # plt.xlim(-20, 20)
        # plt.ylim(-20, 20)
        # plt.axis('equal')
        plt.axis([-40, 40, -40, 40])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    ## get the transformation
    def calPegInHole(self, pegCentersInHole):

        pegInHole = np.eye(4, dtype='float64')
        rotPegInHole = np.eye(3, dtype='float64')
        vector_z = pegCentersInHole[:, 1] - pegCentersInHole[:, 0]
        rotPegInHole[:, 2] = vector_z / np.linalg.norm(vector_z)
        vector_x = pegCentersInHole[:, 2] - pegCentersInHole[:, 0]
        rotPegInHole[:, 0] = vector_x / np.linalg.norm(vector_x)
        vector_y = self.Cross(rotPegInHole[:, 2], rotPegInHole[:, 0])  # caculate the vector_y
        # vector_y = pegCentersInHole[:, 3] - pegCentersInHole[:, 0]
        rotPegInHole[:, 1] = vector_y / np.linalg.norm(vector_y)
        pegInHole[0:3, 0:3] = rotPegInHole
        pegInHole[0:3, 3] = pegCentersInHole[:, 0]

        return pegInHole

    ## transformat the matrix
    def transformPegInHoleModel(self, Centers_pegs, pegs, transformParas):

        rx = transformParas[0] * pi / 180.0
        ry = transformParas[1] * pi / 180.0
        rz = transformParas[2] * pi / 180.0
        rotx = np.mat([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        roty = np.mat([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        rotz = np.mat([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        rotationInPeg = rotz * roty * rotx

        pegInHole = self.calPegInHole(Centers_pegs)
        pegInHole = np.mat(pegInHole)
        HoleInpeg = pegInHole.I
        rotationInHole = pegInHole[0:3, 0:3] * rotationInPeg
        translationInHole = pegInHole[0:3, 0:3] * \
                            np.mat([transformParas[3], transformParas[4], transformParas[5]]).T + \
                            pegInHole[0:3, 3]

        inPointsInPegs_center = HoleInpeg[0:3, 0:3] * \
                                np.mat(Centers_pegs) + HoleInpeg[0:3, 3]
        # print(inPointsInPegs_center)
        Newcenters_pegs = np.array(rotationInHole * inPointsInPegs_center + translationInHole)
        # print(Newcenters_pegs)
        inPointsInPegs = np.zeros((2, 3, 20000))
        Newpegs = np.zeros((2, 3, 20000))
        for i in range(2):
            inPointsInPegs[i] = HoleInpeg[0:3, 0:3] * \
                                np.mat(pegs[i]) + HoleInpeg[0:3, 3]
            Newpegs[i] = rotationInHole * inPointsInPegs[i] + translationInHole

        return Newcenters_pegs, Newpegs


    def Find_DownPoints(self, Center_hole, r_hole, circle_Peg):
        # circle = createCircle(Center, diameterPeg/2).transpose()
        # circle_new = np.add(circle, [0.005, 0., 0.])
        Ellipse_new = circle_Peg.transpose()
        theta = np.arange(0, 2 * pi, 2 * pi / 20000)
        Distance = np.zeros([theta.size, 5])
        Distance[:, 0:2] = np.subtract(Ellipse_new[:, 0:2], Center_hole[0:2])
        Distance[:, 2] = np.sqrt(np.square(Distance[:, 0]) + np.square(Distance[:, 1]))
        Distance[:, 3] = np.subtract(Distance[:, 2], r_hole)
        Distance[:, 3][np.where(Distance[:, 3]<0.)] = 0.
        Dismax = np.max(Distance[:, 3])
        num = np.where((Distance[:, 3] ==Dismax) & (Distance[:, 3] >0.))
        length = num[0].size
        pointsDown = np.zeros((length, 5))
        for i in range(length):
            pointsDown[i, 0:3] = Ellipse_new[num[0][i], :].astype(np.float64)
            pointsDown[i, 3] = np.abs(Distance[num[0][i], 3])
            pointsDown[i, 4] = 0.
        return pointsDown

        # Points_all = np.where(np.abs(Distance[:, 2] - Dismax) < 0.00001)
        # a = Points_all - num[0]
        #
        # Point_another = np.where(np.abs(a) > 200 and np.abs(a) < Ellipse_new.shape[0])
        # print (Point_another)
        # and np.abs(Points_all - num[0])<Ellipse_new.shape[0]))

        # column_dis = Distance(num, 3)
        # pointsDown1 = np.column_stack((Ellipse_new[num,:], column_dis))
        # ## 底部接触点的竖直方向变形为0
        # pointsDown1 = np.column_stack((pointsDown1, [0]))
        # if (Point_another.size > 1): ## 有多个接触点
        #     pointsDown2 = np.column_stack(Ellipse_new[Point_another[0],:], Distance(Point_another[0], 3))
        #     pointsDown2 = np.column_stack(pointsDown2, [0])
        #     pointsDown1 = np.row_stack(pointsDown1, pointsDown2)


    def Find_UpPoints(self, pegCenter, r_Peg, circle_Hole):

        circle_Hole = circle_Hole.transpose()
        Distance = np.zeros([circle_Hole.shape[0], 5])
        Distance[:, 0:2] = np.subtract(circle_Hole[:, 0:2], pegCenter[0:2])
        Distance[:, 2] = np.sqrt(np.square(Distance[:, 0]) + np.square(Distance[:, 1]))
        Distance[:, 3] = np.subtract((-1)*Distance[:, 2], (-1)*r_Peg)
        Distance[:, 3][np.where(Distance[:, 3]<0.)] = 0.
        Dismin = np.min(Distance[:, 2])
        num = np.where((Distance[:, 2] == Dismin) & (Distance[:, 2]<15.))
        length = num[0].size
        pointsUp = np.zeros((length, 5))
        for i in range(length):
            pointsUp[i, 0:3] = circle_Hole[num[0][i], :].astype(np.float64)
            pointsUp[i, 3] = np.abs(Distance[num[0][i], 3])
            pointsUp[i, 4] = 0.
        return pointsUp


    def Get_Threeforces(self, Point, K, u):
        Fn_xy = K * Point[3]
        Fn_z = K * Point[4]
        Fx = (-1) * Fn_xy * Point[0] / np.sqrt(np.square(Point[0]) + np.square(Point[1]))
        Fy = (-1) * Fn_xy * Point[1] / np.sqrt(np.square(Point[0]) + np.square(Point[1]))
        # 向下运动的过程中摩擦力向上
        Ff = Fn_xy * u
        Fz = Ff + Fn_z
        F = np.array([Fx, Fy, Fz])
        return F


    def getDownContactForce(self, inputContour, holeCenter, holeRadius, K, u, holeInPeg):
        holeCenter2d = holeCenter[0:2]
        contactDownPoints = self.Find_DownPoints(holeCenter2d, holeRadius, inputContour)
        # print(contactDownPoints)
        contactForceInPeg = np.zeros(6)
        for i in range(contactDownPoints.shape[0]):
            contactForce = self.Get_Threeforces(contactDownPoints[i, :], K, u).transpose()
            # print (contactForce)
            contact = np.hstack((contactDownPoints[i, 0:3], 1))
            contactPointInPeg = np.array(np.dot(holeInPeg, contact.transpose()))[0]
            contactForceInPeg[0:3] = contactForceInPeg[0:3] + np.dot(holeInPeg[0:3, 0:3], contactForce)
            Forces = np.array(np.dot(holeInPeg[0:3, 0:3], contactForce))
            contactForceInPeg[3:6] = contactForceInPeg[3:6] + self.Cross(contactPointInPeg[0:3] / 1000., Forces[0])   # Nm
        # print (contactForceInPeg)
        return contactForceInPeg


    def getUpContactForce(self, circleHole, pegCenter, radiusPeg, K, u, holeInPeg):
        # a = holeInPeg[0:3, 3].T
        pegCenterInPeg = np.dot(holeInPeg[0:3, 0:3], pegCenter.transpose()) \
                         + holeInPeg[0:3, 3].T

        circleHoleInPeg = np.dot(holeInPeg[0:3, 0:3], circleHole) + holeInPeg[0:3, 3]

        pegCenter2d = np.array(pegCenterInPeg)[0]
        contactUpPointsInPeg = self.Find_UpPoints(pegCenter2d, radiusPeg, circleHoleInPeg)
        # print(contactUpPointsInPeg)
        contactForceInPeg = np.zeros(6)
        for i in range(contactUpPointsInPeg.shape[0]):
            contactForce = self.Get_Threeforces(contactUpPointsInPeg[i, :], K, u).transpose()
            # print(contactForce)
            contactForce[2] = (-1) * contactForce[2]
            contactForceInPeg[0:3] = contactForceInPeg[0:3] + contactForce
            contactForceInPeg[3:6] = contactForceInPeg[3:6] + self.Cross(contactUpPointsInPeg.transpose() / 1000.,
                                                                    contactForce)                # Nm
        # print (contactForceInPeg)
        return contactForceInPeg


    def getSumContactForceInPeg(self, Centers_pegs, Centers_holes, pegs, holes, holeRadius, pegRadius, K, u, holeInPeg):

        sumContactForceInPeg = np.zeros((6))
        sumContactForceInPeg = sumContactForceInPeg + \
                               self.getDownContactForce(pegs[1], Centers_holes[:, 1], holeRadius, K, u, holeInPeg)
        sumContactForceInPeg = sumContactForceInPeg + \
                               self.getUpContactForce(holes[0], Centers_pegs[:, 0], pegRadius, K, u, holeInPeg)
        return sumContactForceInPeg

""" 
#--------------This code just for testing the simulation---------------------
# First_transformParas = np.array([0.05, 0., 0., 0., 0., 2.])
# model = OnePeginHole(First_transformParas)
# # Centers_pegs, Centers_holes, pegs, holes = createPegInHoleModel(diameterHole, diameterPeg, Center, height)
# model.plot(model.Centers_pegs, model.Centers_holes, model.pegs, model.holes)
# # print(pegs.shape)
# transformParas = np.array([0., 0.5, 0., 0., 0., 2.])
# Newcenters_pegs, Newpegs = transformPegInHoleModel(Centers_pegs, pegs, transformParas)
# # print(Newcenters_pegs[:, 0][0:2])
#
# # contactDownPoints = Find_DownPoints(Centers_holes[:, 1], diameterHole/2, Newpegs[1])
#
# # pegCenterInPeg = np.dot(holeInPeg[0:3, 0:3], Newcenters_pegs[:, 0].transpose()) + holeInPeg[0:3, 3]
# # circleHoleInPeg = np.dot(holeInPeg[0:3, 0:3], holes[0]) + holeInPeg[0:3, 3]
# # contactUpPointsInPeg = Find_UpPoints(Newcenters_pegs[:, 0], diameterPeg/2, circleHoleInPeg)
# # print(contactUpPointsInPeg)
# sumContactForceInPeg = getSumContactForceInPeg(Newcenters_pegs, Centers_holes, Newpegs, holes, diameterHole / 2,
#                                                diameterPeg / 2, K, u, holeInPeg)
# print(sumContactForceInPeg)

    # pegInHole = np.mat(pegInHole)
# HoleInpeg = pegInHole.I
# # rotationInHole = pegInHole[1:3,1:3] * rotationInPeg
# # print (HoleInpeg)
# plot(Newcenters_pegs, Centers_holes, Newpegs, holes)

# A = np.array([0, 0, -1])
# B = np.array([1, 0, 0])
# print(Cross(A, B))
# # a = np.array([[[1, 2, 3], [2, 4, 5]],
# #               [[5, 7, 9], [0, 4, 5]]])
# print (a)
# # circle = createCircle(Center, diameterPeg/2).transpose()
# # print (circle.shape[0])

"""


class DualPegsinHoles(object):
    def __init__(self, transformParas):
        self.transformParas = transformParas
        self.diameterHole = 30
        self.diameterPeg = 29.9
        self.distanceAxes = 200
        self.Center = [0, 0, 0]
        self.height = 100
        self.K = 100  #20
        self.u = 0.05
        self.Centers_pegs, self.Centers_holes, self.pegs, self.holes = self.createDualPegsInHolesModel()
        self.Centers_pegs, self.pegs = self.transformPegInHoleModel(self.Centers_pegs, self.pegs, self.transformParas)
        self.pegInHole = np.mat(self.calPegInHole(self.Centers_pegs))
        self.holeInPeg = self.pegInHole.I
        self.sumContactForceInPeg = self.getSumContactForceInPeg(self.Centers_pegs, self.Centers_holes, self.pegs, self.holes, self.diameterHole / 2,
                                                   self.diameterPeg / 2, self.K, self.u, self.holeInPeg)
        self.Position = self.MatrixToEuler(self.pegInHole)
        # self.fig = plt.figure('Simulation')
        # self.plot(self.Centers_pegs, self.Centers_holes, self.pegs, self.holes)

    def ChangeOneStep(self, transformParas):
        self.Centers_pegs, self.pegs = self.transformPegInHoleModel(self.Centers_pegs, self.pegs, transformParas)
        self.pegInHole = np.mat(self.calPegInHole(self.Centers_pegs))
        self.holeInPeg = np.linalg.inv(self.pegInHole)
        self.Position = self.MatrixToEuler(self.pegInHole)
        self.sumContactForceInPeg = self.getSumContactForceInPeg(self.Centers_pegs, self.Centers_holes, self.pegs,
                                                                 self.holes, self.diameterHole / 2,
                                                                 self.diameterPeg / 2, self.K, self.u, self.holeInPeg)
        return self.sumContactForceInPeg, self.Position


    def MatrixToEuler(self, T):
        Position = np.zeros(6)
        Position[0] = T[0, 3]
        Position[1] = T[1, 3]
        Position[2] = T[2, 3]

        Position[5] = np.arctan2(T[1, 0], T[0, 0])*180/pi
        Position[4] = np.arctan2((-1)*T[2, 0], np.cos(Position[5])*T[0, 0]+np.sin(Position[5])*T[1, 0])*180/pi
        Position[3] = np.arctan2(np.sin(Position[5])*T[0, 2] - np.cos(Position[5]) * T[1, 2],  (-1)*np.sin(Position[5]) * T[0, 1] + np.cos(Position[5])*T[1, 1])*180/pi
        return Position


    def Cross(self, A, B):
        C = np.zeros(3)
        if np.isnan(A).any() == True or np.isnan(B).any() == True:
            A = np.nan_to_num(A)
            B = np.nan_to_num(B)

        if np.isinf(A).any() == True or np.isinf(B).any() == True:
            A = np.nan_to_num(A)
            B = np.nan_to_num(B)
        C[0] = A[1] * B[2] - A[2] * B[1]
        C[1] = A[2] * B[0] - A[0] * B[2]
        C[2] = A[0] * B[1] - A[1] * B[0]
        return C

    ## creat single circle
    def createCircle(self, center, radius):
        theta = np.arange(0, 2 * pi, 2 * pi / 20000)

        circle = np.zeros([3, theta.size])
        circle[0, :] = radius * np.cos(theta) + np.repeat(center[0], theta.size)
        circle[1, :] = radius * np.sin(theta) + np.repeat(center[1], theta.size)
        circle[2, :] = np.repeat(center[2], theta.size)
        return circle

    ## creat one hole and one peg model
    def createDualPegsInHolesModel(self):
        diameterHole = self.diameterHole
        diameterPeg = self.diameterPeg
        Center = self.Center
        height = self.height
        distanceAxes = self.distanceAxes
        Centers_pegs = np.array([[-distanceAxes/2, distanceAxes/2, -distanceAxes/2, distanceAxes/2],
                                 [0, 0, 0., 0.],
                                 [2 * height + 1, 2 * height + 1, height + 1, height + 1]])
        Centers_holes = np.array([[-distanceAxes/2, distanceAxes/2, -distanceAxes/2, distanceAxes/2],
                                  [0, 0, 0., 0.],
                                  [height, height, 0., 0.]])
        pegs = np.zeros((4, 3, 20000))
        holes = np.zeros((4, 3, 20000))
        for i in range(4):
            pegs[i] = self.createCircle(Centers_pegs[:, i], diameterPeg / 2)
            holes[i] = self.createCircle(Centers_holes[:, i], diameterHole / 2)
        return Centers_pegs, Centers_holes, pegs, holes

    ## plot the model
    def plot(self, subfig, Centers_pegs, Centers_holes, pegs, holes):

        # fig = plt.figure()
        # ax = self.fig.add_subplot(121, projection='3d')
        ax = subfig
        ax.clear()
        axis_Peg1 = np.array([Centers_pegs[:, 0],
                           Centers_pegs[:, 2]]).transpose()
        axis_Peg2 = np.array([Centers_pegs[:, 1],
                           Centers_pegs[:, 3]]).transpose()
        axis_Hole1 = np.array([Centers_holes[:, 0],
                               Centers_holes[:, 2]]).transpose()
        axis_Hole2 = np.array([Centers_holes[:, 1],
                               Centers_holes[:, 3]]).transpose()
        ax.plot(axis_Peg1[0, :], axis_Peg1[1, :], axis_Peg1[2, :], c='r', label='parametric curve')
        ax.plot(axis_Peg2[0, :], axis_Peg2[1, :], axis_Peg2[2, :], c='r', label='parametric curve')
        ax.plot(axis_Hole1[0, :], axis_Hole1[1, :], axis_Hole1[2, :], c='b', label='parametric curve')
        ax.plot(axis_Hole2[0, :], axis_Hole2[1, :], axis_Hole2[2, :], c='b', label='parametric curve')

        for i in range(4):
            ax.plot(pegs[i][0, :], pegs[i][1, :], pegs[i][2, :], c='r', label='parametric curve')
            ax.plot(holes[i][0, :], holes[i][1, :], holes[i][2, :], c='b', label='parametric curve')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        ax.set_zlim(0, 200)
        ax.set_zlim(0, 200)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.set_aspect('equal', adjustable='box')
        plt.ion()
        # plt.axis([-150, 150, -150, 150])
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.pause(0.001)
        # plt.show()

    ## get the transformation
    def calPegInHole(self, pegCentersInHole):

        pegInHole = np.eye(4, dtype='float64')
        rotPegInHole = np.eye(3, dtype='float64')
        vector_z = pegCentersInHole[:, 2] - pegCentersInHole[:, 0]
        rotPegInHole[:, 2] = vector_z / np.linalg.norm(vector_z)
        vector_x = pegCentersInHole[:, 1] - pegCentersInHole[:, 0]
        rotPegInHole[:, 0] = vector_x / np.linalg.norm(vector_x)
        vector_y = self.Cross(rotPegInHole[:, 2], rotPegInHole[:, 0])
        # vector_y = pegCentersInHole[:, 3] - pegCentersInHole[:, 0]
        rotPegInHole[:, 1] = vector_y / np.linalg.norm(vector_y)
        pegInHole[0:3, 0:3] = rotPegInHole
        pegInHole[0:3, 3] = (pegCentersInHole[:, 0] + pegCentersInHole[:, 1])/2.

        return pegInHole

    ## transformat the matrix
    def transformPegInHoleModel(self, Centers_pegs, pegs, transformParas):

        rx = transformParas[0] * pi / 180.0
        ry = transformParas[1] * pi / 180.0
        rz = transformParas[2] * pi / 180.0
        # rx = transformParas[0]
        # ry = transformParas[1]
        # rz = transformParas[2]
        rotx = np.mat([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        roty = np.mat([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        rotz = np.mat([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        rotationInPeg = rotz * roty * rotx

        pegInHole = self.calPegInHole(Centers_pegs)
        pegInHole = np.mat(pegInHole)
        HoleInpeg = pegInHole.I
        rotationInHole = pegInHole[0:3, 0:3] * rotationInPeg
        translationInHole = pegInHole[0:3, 0:3] * \
                            np.mat([transformParas[3], transformParas[4], transformParas[5]]).T + \
                            pegInHole[0:3, 3]

        inPointsInPegs_center = HoleInpeg[0:3, 0:3] * \
                                np.mat(Centers_pegs) + HoleInpeg[0:3, 3]
        # print(inPointsInPegs_center)
        Newcenters_pegs = np.array(rotationInHole * inPointsInPegs_center + translationInHole)
        # print(Newcenters_pegs)
        inPointsInPegs = np.zeros((4, 3, 20000))
        Newpegs = np.zeros((4, 3, 20000))
        for i in range(4):
            inPointsInPegs[i] = HoleInpeg[0:3, 0:3] * \
                                np.mat(pegs[i]) + HoleInpeg[0:3, 3]
            Newpegs[i] = rotationInHole * inPointsInPegs[i] + translationInHole

        return Newcenters_pegs, Newpegs


    def Find_DownPoints(self, Center_hole, r_hole, circle_Peg):
        # circle = createCircle(Center, diameterPeg/2).transpose()
        # circle_new = np.add(circle, [0.005, 0., 0.])
        Ellipse_new = circle_Peg.transpose()
        theta = np.arange(0, 2 * pi, 2 * pi / 20000)
        Distance = np.zeros([theta.size, 5])
        Distance[:, 0:2] = np.subtract(Ellipse_new[:, 0:2], Center_hole[0:2])
        Distance[:, 2] = np.sqrt(np.square(Distance[:, 0]) + np.square(Distance[:, 1]))
        Distance[:, 3] = np.subtract(Distance[:, 2], r_hole)
        Distance[:, 3][np.where(Distance[:, 3]<0.)] = 0.
        Dismax = np.max(Distance[:, 3])
        num = np.where((Distance[:, 3] ==Dismax) & (Distance[:, 3] >0.))
        length = num[0].size
        pointsDown = np.zeros((length, 7))
        for i in range(length):
            pointsDown[i, 0:3] = Ellipse_new[num[0][i], :].astype(np.float64)
            pointsDown[i, 3] = np.abs(Distance[num[0][i], 3])
            pointsDown[i, 4] = 0.
            pointsDown[i, 4] = 0.
            pointsDown[i, 5:7] = Center_hole[0:2] - Ellipse_new[num[0][i], 0:2].astype(np.float64)
        # print(pointsDown)
        return pointsDown

        # Points_all = np.where(np.abs(Distance[:, 2] - Dismax) < 0.00001)
        # a = Points_all - num[0]
        #
        # Point_another = np.where(np.abs(a) > 200 and np.abs(a) < Ellipse_new.shape[0])
        # print (Point_another)
        # and np.abs(Points_all - num[0])<Ellipse_new.shape[0]))

        # column_dis = Distance(num, 3)
        # pointsDown1 = np.column_stack((Ellipse_new[num,:], column_dis))
        # ## 底部接触点的竖直方向变形为0
        # pointsDown1 = np.column_stack((pointsDown1, [0]))
        # if (Point_another.size > 1): ## 有多个接触点
        #     pointsDown2 = np.column_stack(Ellipse_new[Point_another[0],:], Distance(Point_another[0], 3))
        #     pointsDown2 = np.column_stack(pointsDown2, [0])
        #     pointsDown1 = np.row_stack(pointsDown1, pointsDown2)


    def Find_UpPoints(self, pegCenter, r_Peg, circle_Hole):

        circle_Hole = circle_Hole.transpose()
        Distance = np.zeros([circle_Hole.shape[0], 5])
        Distance[:, 0:2] = np.subtract(circle_Hole[:, 0:2], pegCenter[0:2])
        Distance[:, 2] = np.sqrt(np.square(Distance[:, 0]) + np.square(Distance[:, 1]))
        Distance[:, 3] = np.subtract((-1)*Distance[:, 2], (-1)*r_Peg)
        Distance[:, 3][np.where(Distance[:, 3]<0.)] = 0.
        Dismin = np.min(Distance[:, 2])
        num = np.where((Distance[:, 2] == Dismin) & (Distance[:, 2]<14.95))
        length = num[0].size
        pointsUp = np.zeros((length, 7))
        for i in range(length):
            pointsUp[i, 0:3] = circle_Hole[num[0][i], :].astype(np.float64)
            pointsUp[i, 3] = np.abs(Distance[num[0][i], 3])
            pointsUp[i, 4] = 0.
            pointsUp[i, 5:7] = pegCenter[0:2] - circle_Hole[num[0][i], 0:2].astype(np.float64)
        # print(pointsUp)
        return pointsUp


    def Get_Threeforces(self, Point, K, u):
        Fn_xy = K * Point[3]
        Fn_z = K * Point[4]
        Fx = Fn_xy * Point[5] / np.sqrt(np.square(Point[5]) + np.square(Point[6]))
        Fy = Fn_xy * Point[6] / np.sqrt(np.square(Point[5]) + np.square(Point[6]))
        # 向下运动的过程中摩擦力向上
        Ff = Fn_xy * u
        Fz = Ff + Fn_z
        F = np.array([Fx, Fy, Fz])
        return F


    def getDownContactForce(self, inputContour, holeCenter, holeRadius, K, u, holeInPeg):
        holeCenter2d = holeCenter[0:2]
        contactDownPoints = self.Find_DownPoints(holeCenter2d, holeRadius, inputContour)
        # print(contactDownPoints)
        contactForceInPeg = np.zeros((6))
        for i in range(contactDownPoints.shape[0]):
            # contactDownPoints[i, 0:2] = holeCenter2d - contactDownPoints[i, 0:2]
            contactForce = self.Get_Threeforces(contactDownPoints[i, :], K, u).transpose()
            # print (contactForce)
            contact = np.hstack((contactDownPoints[i, 0:3], 1))
            contactPointInPeg = np.array(np.dot(holeInPeg, contact.transpose()))[0]
            contactForceInPeg[0:3] = contactForceInPeg[0:3] + np.dot(holeInPeg[0:3, 0:3], contactForce)
            Forces = np.array(np.dot(holeInPeg[0:3, 0:3], contactForce))
            contactForceInPeg[3:6] = contactForceInPeg[3:6] + self.Cross(contactPointInPeg[0:3]/1000., Forces[0])
        # print (contactForceInPeg)
        return contactForceInPeg


    def getUpContactForce(self, circleHole, pegCenter, radiusPeg, K, u, holeInPeg):
        # a = holeInPeg[0:3, 3].T
        pegCenterInPeg = np.dot(holeInPeg[0:3, 0:3], pegCenter.transpose()) \
                         + holeInPeg[0:3, 3].T

        circleHoleInPeg = np.dot(holeInPeg[0:3, 0:3], circleHole) + holeInPeg[0:3, 3]

        pegCenter2d = np.array(pegCenterInPeg)[0]
        contactUpPointsInPeg = self.Find_UpPoints(pegCenter2d, radiusPeg, circleHoleInPeg)
        # print(contactUpPointsInPeg)
        contactForceInPeg = np.zeros((6))
        for i in range(contactUpPointsInPeg.shape[0]):
            # contactUpPointsInPeg[i, 0:2] = pegCenter2d[0:2] - contactUpPointsInPeg[i, 0:2]
            contactForce = self.Get_Threeforces(contactUpPointsInPeg[i, :], K, u).transpose()
            # print(contactForce)
            contactForce[2] = (-1) * contactForce[2]
            contactForceInPeg[0:3] = contactForceInPeg[0:3] + contactForce
            contactForceInPeg[3:6] = contactForceInPeg[3:6] + self.Cross(contactUpPointsInPeg.transpose()/1000., contactForce)
        # print (contactForceInPeg)
        return contactForceInPeg

    # Get_all_forces
    def getSumContactForceInPeg(self, Centers_pegs, Centers_holes, pegs, holes, holeRadius, pegRadius, K, u, holeInPeg):


        sumContactForceInPeg = np.zeros((6))

        # ContactForceInPeg_left = np.zeros(6)
        # ContactForceInPeg_right = np.zeros(6)
        # ContactPointsInPeg_left = np.zeros(3)
        # ContactForceInPeg_downleft, ContactPointsInPeg_downleft = self.getDownContactForce(pegs[2], Centers_holes[:, 2], holeRadius, K, u, holeInPeg)
        # ContactForceInPeg_downright, ContactPointsInPeg_downright = self.getDownContactForce(pegs[3], Centers_holes[:, 3], holeRadius, K, u, holeInPeg)
        #
        # ContactForceInPeg_upleft, ContactPointsInPeg_upleft = self.getUpContactForce(holes[0], Centers_pegs[:, 0], pegRadius, K, u, holeInPeg)
        # ContactForceInPeg_upright, ContactPointsInPeg_upright = self.getUpContactForce(holes[1], Centers_pegs[:, 1], pegRadius, K, u, holeInPeg)
        ContactForceInPeg_down = np.zeros((6))
        ContactForceInPeg_up = np.zeros((6))
        for i in range(2):
            ContactForceInPeg_down = self.getDownContactForce(pegs[i+2], Centers_holes[:,i+2], holeRadius, K, u, holeInPeg)
            ContactForceInPeg_up = self.getUpContactForce(holes[i], Centers_pegs[:, i], pegRadius, K, u, holeInPeg)
            if (ContactForceInPeg_down[0] * ContactForceInPeg_up[0] <= 0) and (
                            ContactForceInPeg_down[1] * ContactForceInPeg_up[1] <= 0):
                sumContactForceInPeg = sumContactForceInPeg + ContactForceInPeg_down + ContactForceInPeg_up
            else:
                sumContactForceInPeg = sumContactForceInPeg + ContactForceInPeg_down

            # if (ContactPointsInPeg_downleft[i,0]*ContactPointsInPeg_upleft[i, 0] > 0) and (ContactPointsInPeg_downleft[i,1]*ContactPointsInPeg_upleft[i, 1] > 0):
            #     sumContactForceInPeg = sumContactForceInPeg + ContactForceInPeg_downleft
            # else:
            #     sumContactForceInPeg = sumContactForceInPeg + ContactForceInPeg_downleft
            #     sumContactForceInPeg = sumContactForceInPeg + ContactForceInPeg_downleft
        # sumContactForceInPeg = sumContactForceInPeg + \
        #                        self.getDownContactForce(pegs[2], Centers_holes[:, 2], holeRadius, K, u, holeInPeg)
        # sumContactForceInPeg = sumContactForceInPeg + \
        #                        self.getDownContactForce(pegs[3], Centers_holes[:, 3], holeRadius, K, u, holeInPeg)

        # sumContactForceInPeg = sumContactForceInPeg + \
        #                        self.getUpContactForce(holes[0], Centers_pegs[:, 0], pegRadius, K, u, holeInPeg)
        # sumContactForceInPeg = sumContactForceInPeg + \
        #                        self.getUpContactForce(holes[1], Centers_pegs[:, 1], pegRadius, K, u, holeInPeg)
        sumContactForceInPeg[0] = np.random.normal(sumContactForceInPeg[0], 0.1 * np.abs(sumContactForceInPeg[0]))
        sumContactForceInPeg[1] = np.random.normal(sumContactForceInPeg[1], 0.1 * np.abs(sumContactForceInPeg[1]))
        sumContactForceInPeg[2] = np.random.normal(sumContactForceInPeg[2], 0.1 * np.abs(sumContactForceInPeg[2]))
        sumContactForceInPeg[2] = np.clip(sumContactForceInPeg[2], -100, 0)
        sumContactForceInPeg[3] = np.random.normal(sumContactForceInPeg[3], 0.1 * np.abs(sumContactForceInPeg[3]))
        sumContactForceInPeg[4] = np.random.normal(sumContactForceInPeg[4], 0.1 * np.abs(sumContactForceInPeg[4]))
        sumContactForceInPeg[5] = np.random.normal(sumContactForceInPeg[5], 0.1 * np.abs(sumContactForceInPeg[5]))

        return sumContactForceInPeg
# print (model.sumContactForceInPeg)
# print (model.Position)


# # # Centers_pegs, Centers_holes, pegs, holes = createPegInHoleModel(diameterHole, diameterPeg, Center, height)
# # model.plot(model.Centers_pegs, model.Centers_holes, model.pegs, model.holes)
# print(pegs.shape)
# transformParas = np.array([0., 0.5, 0., 0., 0., 2.])
# Newcenters_pegs, Newpegs = transformPegInHoleModel(Centers_pegs, pegs, transformParas)


# Krxyz = 0.001
# Kpxy = 0.0008
# Kpz = 0.015
# First_transformParas = np.array([ 0.0553905,  -0.08639118, -0.0799849, 0.01339554, 0.00722043, 3.])
# model = DualPegsinHoles(First_transformParas)
# refForece = [0, 0, -100, 0, 0, 0]
# for i in range(200):
#     sumContactForceInPeg = model.sumContactForceInPeg
#     # print(sumContactForceInPeg)
#     errorForce = sumContactForceInPeg - refForece
#     action = np.zeros(6)
#     action[0:3] = Krxyz * errorForce[3:6]
#     action[3:5] = Kpxy * errorForce[0:2]
#     action[5] = Kpz * errorForce[2]
#     print("action")
#     print(action)
#     sumContactForceInPeg, position = model.ChangeOneStep(action)
#     print("sensors")
#     print(sumContactForceInPeg)
#     print(position)
#     if position[2] < 100:
#         break
# print("Finish",i)


# s = np.array([20., 0.0, 0., 0., 0., 10.])
# dual_model = DualPegsinHoles(s)
# Centers_pegs, Centers_holes, pegs, holes = dual_model.createDualPegsInHolesModel()
# Centers_pegs_new, pegs_new = dual_model.transformPegInHoleModel(Centers_pegs, pegs, s)
# dual_model.plot(Centers_pegs_new, Centers_holes, pegs_new, holes)




