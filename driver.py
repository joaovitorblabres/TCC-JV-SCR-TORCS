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
        action = tm.train(new_msg)
        #action = tm.train(np.array(n))
        self.steer(action%4, action%4)
        #
        self.gear()
        #
        self.speed(action%4, action%4)

        return self.control.toMsg()

    def steer(self, steeringLeft, steeringRight):
        self.control.setSteer(max(steeringLeft, steeringRight))

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

    def speed(self, accel, brake):
        self.control.setAccel(max(accel, brake))


    def onShutDown(self):
        tm.endTrain()
        pass

    def onRestart(self):
        pass
