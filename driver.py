'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import torcs_MODELO as tm
import numpy as np

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage

        self.parser = msgParser.MsgParser()

        self.state = carState.CarState()

        self.control = carControl.CarControl()
        self.steer_lock = 0.785398
        self.max_speed = 350
        self.prev_rpm = None

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):
        new_msg = self.state.setFromMsg(msg)

        ## ADICIONAR O TREINAMENTO AQUI
        print(new_msg)
        action = tm.train(new_msg)
        #action = tm.train(np.array(n))
        print(action)
        # self.steer()
        #
        # self.gear()
        #
        # self.speed()

        return self.control.toMsg()

    def steer(self):
        angle = float(self.state.angle)
        dist = float(self.state.trackPos)
        self.control.setSteer((angle - dist*0.8)/self.steer_lock)

    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()

        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False

        if up and rpm > 8000:
            gear += 1

        if not up and rpm < 6000:
            gear -= 1

        self.control.setGear(gear)

    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()

        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0

        self.control.setAccel(accel)


    def onShutDown(self):
        pass

    def onRestart(self):
        pass
