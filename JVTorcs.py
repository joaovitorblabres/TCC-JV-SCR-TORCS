from time import sleep
import sys
import argparse
import socket
import driver
import rewards as rw
import tensorflow as tf
import DQLearning as DQL
import AC
import subprocess
import os

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
                    help='Algoritmo (0 - DQLearning, 1 - AC)')
parser.add_argument('--rest', action='store', dest='restore', type=int, default=0,
                    help='Restore Model (0 - No, 1 - Yes)')

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
if __name__ == "__main__":
    if arguments.alg == 0:
        RL = DQL.DeepQNetwork(4, 30,
                      learning_rate=0.001,
                      reward_decay=0.99,
                      e_greedy=0.99,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                  )
        sess = RL.sess
    if arguments.alg == 1:
        N_F = 30
        N_A = 4
        sess = tf.Session()
        actor = AC.Actor(sess, n_features=AC.N_F, n_actions=AC.N_A, lr=AC.LR_A)
        critic = AC.Critic(sess, n_features=AC.N_F, lr=AC.LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
        sess.run(tf.global_variables_initializer())

maximumRewardRecorded = -5000000
maximumDistanceTraveled = -5000
traveled = -500
saver = tf.train.Saver()
dirpath = os.getcwd()
algo = ''
if arguments.alg == 0:
    algo = 'DQL'
elif arguments.alg == 1:
    algo = 'AC'

with tf.device('/device:GPU:0'):
    if arguments.restore == 1:
        restore = dirpath + "\\" + algo + "\\"
        saver.restore(sess, restore + 'lr_0.001_8191')
        curEpisode = 8192
    while not shutdownClient:
        os.chdir(r'C:\\Program Files (x86)\\torcs\\')
        p = subprocess.Popen(['python', r'C:\Program Files (x86)\torcs\openTorcs.py ', str(arguments.host_port%3001)], stdin=None, stderr=None, stdout=None, shell=None)
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
        while True:
            # wait for an answer from server
            buf = None
            oldStep = state
            try:
                buf, addr = sock.recvfrom(1000)
            except socket.error as msg:
                pass
                #print("didn't get response from server...")

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
                    if arguments.alg == 1:
                        buf, action, state, bufState = d.drive(buf.decode(), actor)
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
                reward = rw.lng_trans(bufState)
                episode_rewards_sum += reward
                if arguments.alg == 0:
                    d.atualiza(RL, oldStep, action, reward, state)
                    if (currentStep > 200) and (currentStep % 20 == 0):
                        RL.learn()
                if arguments.alg == 1:
                    td_error = critic.learn(oldStep, reward, state)
                    actor.learn(oldStep, action, td_error)

        #print(bufState)
        #if bufState['distRaced'][0] == None:
            #bufState['distRaced'][0] = 0
        if curEpisode + 1 > 10000 and curEpisode%2000 == 0:
            saved_path = saver.save(sess, './' + algo + '/lr_'+str(0.001)+'_'+str(curEpisode)+'')
        maximumDistanceTraveled = max(traveled, maximumDistanceTraveled)
        maximumRewardRecorded = max(episode_rewards_sum/currentStep, maximumRewardRecorded)
        print("==========================================")
        print("Episode:", curEpisode)
        print("Reward:", episode_rewards_sum)
        print("Steps for this Episode:", currentStep)
        print("Mean Reward:", episode_rewards_sum/currentStep)
        print("Distance traveled:", traveled)
        print("Max distance traveled so far:", maximumDistanceTraveled)
        print("Max reward so far:", maximumRewardRecorded)
        print("==========================================")

        curEpisode += 1

        if curEpisode == arguments.max_episodes:
            shutdownClient = True


sock.close()
