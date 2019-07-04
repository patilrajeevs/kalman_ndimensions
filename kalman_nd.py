import numpy as np


class KalmanNDimensions(object):
    """Class implements a kalman filter for measurements and predictions in "n" dimensions
    """
    def __init__(self):
        '''
        # Movement and velocity in 1 dimention, lets say x
        self.measurements = np.array([1, 2, 3], np.float) #measurement inputs recieved, lets say x co-ordinates
        self.x = np.array([[0.], [0.]], np.float) # initial state (location and velocity)
        self.P = np.array([[1000., 1.], [0., 1000.]], np.float) # initial uncertainty(covariance)
        self.u = np.array([[0.], [0.]], np.float) # external motion
        self.F = np.array([[1., 1.], [0, 1.]], np.float) # next state function
        self.H = np.array([[1., 0.]], np.float) # measurement function
        self.R = np.array([[1.]], np.float) # measurement uncertainty
        self.I = np.array([[1., 0.], [0., 1.]], np.float) # identity matrix
        '''

        #N dimension state, x,y, and velocity in x direction and y direction
        #frequency of measurement update
        self.dt = 0.1
        self.measurements = np.array([[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]], np.float)
        #name should be xy,or just "state" but keeping it as x to be compatible to previous versions
        #this represents x,y/veocity_x, velocity_y
        self.x = np.array( [[4.],[12.], [0.], [0.]], np.float) # initial state (location and velocity)
        self.F = np.array([ [1., 0., self.dt, 0.     ],\
                            [0., 1., 0.     , self.dt],\
                            [0., 0., 1.     , 0.      ],\
                            [0., 0., 0.     , 1.      ]], np.float) # next state function
        self.P = np.array([ [0., 0., 0.     , 0.     ],\
                            [0., 0., 0.     , 0.     ],\
                            [0., 0., 1000.  , 0.     ],\
                            [0., 0., 0.     , 1000.  ]], np.float) # co-variance matrix. Start with a large un certainty

        self.u = np.array([[0.], [0.], [0.], [0.]], np.float) # external motion
        self.H = np.array([[1., 0., 0., 0.],\
                           [0., 1., 0., 0.]], np.float) # measurement function
        self.R = np.array([[0.1, 0.0], [0.0, 0.1]], np.float) # measurement uncertainty
        self.I = np.identity(4, np.float) # identity matrix

    def predict(self):
        """Prediction of next state
        """
        self.x = np.dot(self.F, self.x) + self.u #state transition matrix + apply external motion
        self.P = np.dot(np.dot(self.F, self.P), np.transpose(self.F)) #Update the state un-certainty F*P*F_transpose

    def measure(self, measurement):
        """Get the measurement update
        """
        measurement = np.array([measurement])
        y = np.transpose(measurement) - np.dot(self.H, self.x) #Difference between the measured value, and previously predicted one
        s = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R  #Measurement uncertainty + measurement noise
        k = np.dot(np.dot(self.P, np.transpose(self.H)),np.linalg.inv(s)) #kalman gain
        self.x = self.x + np.dot(k, y) #new measurement = old measurement + kalman gain * difference
        self.P = np.dot((self.I - np.dot(k, self.H)), self.P) #uncertainty decreases with measurement.
                                                              #covariance is reduced

    def __str__(self):
        return "%s\n%s"%(self.x, self.P)

if __name__ == "__main__":
    obj = KalmanNDimensions()
    for m in obj.measurements:
        obj.measure(m)
        print "After measurement %s\n"%obj.P
        obj.predict()
        print "After prediction %s\n"%obj.P
    print (obj)
    # output should be:
    # x: [[3.9996664447958645], [0.9999998335552873]]
    # P: [[2.3318904241194827, 0.9991676099921091], [0.9991676099921067, 0.49950058263974184]]
