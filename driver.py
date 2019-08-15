'''
Created on Apr 4, 2012

@author: lanquarden
'''

import msgParser
import carState
import carControl
import torcs_MODELO as tm
import numpy as np
import datetime

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
        self.rpmList = [0, 100,3000,4000,4800,5200,5600]

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

    def drive(self, msg, RL):
        new_msg = self.state.setFromMsg(msg)
        #print(new_msg)
        ## ADICIONAR O TREINAMENTO AQUI
        n = []
        for key, val in new_msg.items():
        	for i in val:
        		n.append(i)
        new = np.array(n)
        action = RL.choose_action(new)
        #action = tm.train(np.array(n))
        if action == 0:
            self.steer(1, 0)
        elif action == 1:
            self.steer(0, 1)
        #
        self.gear()
        #
        if action == 2:
            self.speed(1, 0)
        elif action == 3:
            self.speed(0, 1)

        return (self.control.toMsg(), action, new, new_msg)

    def atualiza(self, RL, state, action, reward, state_):
        RL.store_transition(state, action, reward, state_)

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

        if rpm < self.rpmList[gear]:
            gear -= 1

        self.control.setGear(gear)

    def speed(self, accel, brake):
        self.control.setAccel(max(accel, brake))


    def onShutDown(self):
        #tm.endTrain()
        pass

    def onRestart(self):
        pass
