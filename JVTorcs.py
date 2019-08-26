from time import sleep
import sys
import argparse
import socket
import driver
import rewards as rw
import tensorflow as tf
import DQLearning as DQL
import numpy as np
import AC
import DDPG
import subprocess
import os
import math

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description = 'Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--alg', action='store', dest='alg', type=int, default=0,
                    help='Algoritmo (0 - DQLearning, 1 - AC, 2 - DDPG)')
parser.add_argument('--rest', action='store', dest='restore', type=int, default=0,
                    help='Restore Model (0 - No, 1 - Yes)')
parser.add_argument('--win', action='store', dest='system', type=int, default=0,
                    help='Restore Model (0 - win, 1 - linux)')
parser.add_argument('--save', action='store', dest='saveEp', type=int, default=2000,
                    help='Restore Model (default: 2000)')
parser.add_argument('--lra', action='store', dest='lra', type=int, default=0,
                    help='Learning Rate Actor/Commun')
parser.add_argument('--lrc', action='store', dest='lrc', type=int, default=1,
                    help='Learning Rate Critic')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = False
d = driver.Driver(arguments.stage)
sess = tf.Session()

maximumRewardRecorded = -5000000
maximumDistanceTraveled = -5000
traveled = -500
dirpath = os.getcwd()
algo = ''
lrAs = [0.0001, 0.001, 0.01, 0.1]
lrCs = [0.0001, 0.001, 0.01, 0.1]

lrActor = lrAs[arguments.lra]
lrCritic = lrCs[arguments.lrc]
#python JVTorcs.py --maxEpisodes=500000 --alg=2 --save=100 --port=3001 --rest=1 --lra=2 --lrc=3
#python JVTorcs.py --maxEpisodes=500000 --alg=1 --save=100 --port=3002 --rest=1 --lra=2 --lrc=3

restartN = 0
if arguments.alg == 0:
    algo = 'DQL'
elif arguments.alg == 1:
    algo = 'AC'
elif arguments.alg == 2:
    algo = 'DDPG'

if arguments.alg == 0:
    RL = DQL.DeepQNetwork(4, 30,
        learning_rate=lrActor,
        reward_decay=0.99,
        e_greedy=0.99,
        replace_target_iter=200,
        memory_size=2000,
        # output_graph=True
    )
    sess = RL.sess
elif arguments.alg == 1:
    sess = tf.Session()
    actor = AC.Actor(sess, n_features=AC.N_F, n_actions=AC.N_A, lr=lrActor)
    critic = AC.Critic(sess, n_features=AC.N_F, lr=AC.LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
    sess.run(tf.global_variables_initializer())
elif arguments.alg == 2:
    var = 3
    actor = DDPG.Actor(sess, DDPG.action_dim, DDPG.action_bound, lrActor, DDPG.REPLACEMENT)
    critic = DDPG.Critic(sess, DDPG.state_dim, DDPG.action_dim, lrCritic, DDPG.GAMMA, DDPG.REPLACEMENT, actor.a, actor.a_)
    actor.add_grad_to_graph(critic.a_grads)
    sess.run(tf.global_variables_initializer())
    M = DDPG.Memory(DDPG.MEMORY_CAPACITY, dims=2 * DDPG.state_dim + DDPG.action_dim + 1)
saver = tf.train.Saver()
if arguments.restore == 1:
    if arguments.system == 0:
        restore = dirpath + "\\" + algo + "\\LR" + str(lrActor) + "\\"
    else:
        restore = dirpath + "/" + algo + "/LR" + str(lrActor) + "/"
    f = open(restore + "checkpoint", 'r')
    line = f.readline()
    lastModel = line.split(' ')[1].replace('\"', '').replace('\n', '')
    saver.restore(sess, restore + lastModel)
    curEpisode = int(lastModel.split('_')[2]) + 1
with tf.device('/device:GPU:0'):
    while not shutdownClient:
        if arguments.system == 0:
            os.chdir(r'C:\\Program Files (x86)\\torcs\\')
            p = subprocess.Popen(['python', r'C:\Program Files (x86)\torcs\openTorcs.py ', str(arguments.host_port%3001)], stdin=None, stderr=None, stdout=None, shell=None)
            os.chdir(dirpath)
        else:
            #python3 JVTorcs.py --maxEpisodes=500000 --alg=1 --win=1
            #os.chdir(r'../torcs-1.3.7/BUILD/bin/')
            practice = "practice" + str(arguments.host_port%3001) + ".xml"
            p = subprocess.Popen('torcs -r ~/.torcs/config/raceman/'+ practice +' -nofuel -nodamage', shell=True)
            os.chdir(dirpath)

        while True:
            #print('Sending id to server: ', arguments.id)
            buf = arguments.id + d.init()
            #print('Sending init string to server:', buf)

            try:
                b = buf.encode()
                sock.sendto(b, (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                #print("Failed to send data...Exiting...")
                sys.exit(-1)

            try:
                buf, addr = sock.recvfrom(1000)
            except socket.error as msg:
                #print("didn't get response from server...")
                pass

            try:
                buf.encode()
                if buf.find('***identified***') >= 0:
                    print('Received: ', buf)
                    break
            except:
                if buf.find(b'***identified***') >= 0:
                    print('Received: ', buf)
                    break

        currentStep = 0
        episode_rewards_sum = 0
        oldStep = []
        state = []
        traveled = -500
        reward = 0

        while True:
            # wait for an answer from server
            buf = None
            oldStep = state
            try:
                buf, addr = sock.recvfrom(1000)
            except socket.error as msg:
                #print("didn't get response from server when executing...")
                subprocess.run(["killall", "torcs-bin"])
                restartN += 1
                break
                #pass

            if verbose:
                pass
                #print('Received: ', buf)

            try:
                buf.encode()
                if buf != None and buf.find('***shutdown***') >= 0:
                    d.onShutDown()
                    #shutdownClient = True
                    print('Client Shutdown')
                    break
            except:
                if buf != None and buf.find(b'***shutdown***') >= 0:
                    d.onShutDown()
                    #shutdownClient = True
                    print('Client Shutdown')
                    break

            try:
                buf.encode()
                if buf != None and buf.find('***restart***') >= 0:
                    #d.onRestart()
                    print('Client Restart')
                    break
            except:
                if buf != None and buf.find(b'***restart***') >= 0:
                    #d.onRestart()
                    print('Client Restart')
                    break

            currentStep += 1
            bufState = 0
            action = None
            if currentStep != arguments.max_steps:
                if buf != None:
                    if arguments.alg == 0:
                        buf, action, state, bufState = d.drive(buf.decode(), RL)
                    elif arguments.alg == 1:
                        buf, action, state, bufState = d.drive(buf.decode(), actor)
                    elif arguments.alg == 2:
                        buf, action, state, bufState = d.drive(buf.decode(), actor, 1, var)
            else:
                buf = '(meta 1)'

            if verbose:
                print('Sending: ', buf)
            #print(currentStep)
            if buf != None and oldStep != []:
                try:
                    b = buf.encode()
                    sock.sendto(b, (arguments.host_ip, arguments.host_port))
                except socket.error as msg:
                    print("Failed to send data...Exiting...")
                    sys.exit(-1)
                #print(bufState.decode())
                try:
                    traveled = max(float(bufState['distRaced'][0]), traveled)
                except:
                    pass
                reward = rw.lng_trans_heavy_penalty(bufState)
                episode_rewards_sum += reward
                if arguments.alg == 0:
                    d.atualiza(RL, oldStep, action, reward, state)
                    if (currentStep > 200) and (currentStep % 20 == 0):
                        RL.learn()
                elif arguments.alg == 1:
                    td_error = critic.learn(oldStep, reward, state)
                    actor.learn(oldStep, action, td_error)
                elif arguments.alg == 2:
                    M.store_transition(oldStep, action, reward[0] / 10, state)

                    if M.pointer > DDPG.MEMORY_CAPACITY:
                        var *= .99995    # decay the action randomness
                        b_M = M.sample(DDPG.BATCH_SIZE)
                        b_s = b_M[:, :DDPG.state_dim]
                        b_a = b_M[:, DDPG.state_dim: DDPG.state_dim + DDPG.action_dim]
                        b_r = b_M[:, -DDPG.state_dim - 1: -DDPG.state_dim]
                        b_s_ = b_M[:, -DDPG.state_dim:]

                        critic.learn(b_s, b_a, b_r, b_s_)
                        actor.learn(b_s)

        #print(bufState)
        #if bufState['distRaced'][0] == None:
            #bufState['distRaced'][0] = 0
        if curEpisode % arguments.saveEp == 0:
            saved_path = saver.save(sess, './' + algo + '/lr_'+str(lrActor)+'_'+str(curEpisode)+'')
        if math.isnan(reward) == False and currentStep > 0:
            maximumDistanceTraveled = max(traveled, maximumDistanceTraveled)
            maximumRewardRecorded = max(episode_rewards_sum/currentStep, maximumRewardRecorded)
            print("==========================================")
            print("Episode:", curEpisode)
            print("Reward:", episode_rewards_sum)
            print("Steps for this Episode:", currentStep)
            print("Mean Reward:", episode_rewards_sum/currentStep)
            print("Distance traveled:", traveled)
            print("Max distance traveled so far:", maximumDistanceTraveled)
            print("Max mean reward so far:", maximumRewardRecorded)
            print("Number of restars:", restartN)
            print("==========================================")

        curEpisode += 1

        if curEpisode == arguments.max_episodes:
            shutdownClient = True


sock.close()
