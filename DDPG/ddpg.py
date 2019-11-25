from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import csv
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
from datetime import datetime
import os, subprocess

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 3  #Steering/Acceleration/Brake
    state_dim = 28  #of sensors input

    np.random.seed(1337)

    vision = True

    EXPLORE = 100000.
    episode_count = 100000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0
    maximumDistanceTraveled = -500
    maximumRewardRecorded = -500
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    now = datetime.now()
    track = "DSPEED_28I_F2"
    testTime = now.strftime(track+"_%d_%m_%Y_%H_%M_%S_FINAL")
    os.mkdir(testTime)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    #Now load the weight
    print("Now we load the weight")
    try:
        # actor.model.load_weights(testTime+"/actormodel.h5")
        # critic.model.load_weights(testTime+"/criticmodel.h5")
        # actor.target_model.load_weights(testTime+"/actormodel.h5")
        # critic.target_model.load_weights(testTime+"/criticmodel.h5")
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    currentStep = 0
    epSteps = 0
    fileName = testTime + '/eps.csv'
    with open(fileName, mode='a') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['Episode', 'Total Reward', 'Steps', 'Mean Reward', 'Traveled', 'Laps', 'Date'])

    # proc1 = subprocess.Popen(["python3", "graph.py", testTime, "1"], shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)
    # proc2 = subprocess.Popen(["python3", "graph.py", testTime, "4"], shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)
    for i in range(episode_count):
        valid = 1
        laps = []
        # if random.random() <= 0.02:
        #     epsilon = 1

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        total_reward = 0.
        currentStep = step
        if train_indicator == 1:
            proc3 = subprocess.Popen(["sh", "at.sh"], shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)
        else:
            proc3 = subprocess.Popen(["sh", "at2.sh"], shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)
        for j in range(max_steps):
            if ob.curLapTime < 0:
                j -= 1
                action_t = np.zeros([1, action_dim])
                ob, raw_reward_t, done, info = env.step(action_t[0])
                state_t1 = np.hstack((ob.angle, ob.track, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
                state_t = state_t1
                continue
            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            # noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            # #The following code do the stochastic brake
            # if random.random() <= 0.1:
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            # if j < 500 and i // 200 % 2 == 0:
            #     a_t[0][0] *= (j)/800
                # a_t[0][2] *= (j)/300
            lapA = ob.lastLapTime
            damA = ob.damage
            ob, r_t, done, info = env.step(a_t[0])
            lapB = ob.lastLapTime
            damB = ob.damage
            if damA != damB and valid == 1:
                valid = 0
                print("INVALIDADOOOOOOO!!")
            if lapA != lapB:
                laps.append((lapB, valid))
                print("JA FEZ " + str(len(laps)) + " VOLTA(S)")
                valid = 1

            s_t1 = np.hstack((ob.angle, ob.track, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer

            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1
            traveled = ob.distRaced
            # if step % 10 == 0:
            if step % 100 == 0:
                #print(ob.angle, ob.trackPos)
                print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        if len(laps) > 0:
            fileName = testTime + '/laps' + dt_string + '.csv'
            with open(fileName, mode='a') as csvFile:
                csvWriter = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for lap in laps:
                    csvWriter.writerow(lap)

        epSteps = step - currentStep
        maximumDistanceTraveled = max(traveled, maximumDistanceTraveled)
        maximumRewardRecorded = max(total_reward/epSteps, maximumRewardRecorded)
        print("==========================================")
        print("Episode:", i)
        print("Reward:", total_reward)
        print("Steps for this Episode:", epSteps)
        print("Mean Reward:", total_reward/epSteps)
        print("Distance traveled:", traveled)
        print("Max distance traveled so far:", maximumDistanceTraveled)
        print("Max mean reward so far:", maximumRewardRecorded)
        print("==========================================")
        fileName = testTime + '/eps.csv'
        with open(fileName, mode='a') as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvWriter.writerow([i, total_reward, epSteps, total_reward/epSteps, traveled, len(laps), dt_string])


    env.end()  # This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
