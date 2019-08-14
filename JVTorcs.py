import sys
import argparse
import socket
import driver
import rewards as rw
import DQLearning as DQL

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

if __name__ == "__main__":
    # maze game
    RL = DQL.DeepQNetwork(4, 30,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                  )

while not shutdownClient:
    while True:
        print('Sending id to server: ', arguments.id)
        buf = arguments.id + d.init()
        print('Sending init string to server:', buf)

        try:
            b = buf.encode()
            sock.sendto(b, (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)

        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error as msg:
            print("didn't get response from server...")

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
    oldStep = []
    state = []
    while True:
        # wait for an answer from server
        buf = None
        oldStep = state
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error as msg:
            print("didn't get response from server...")

        if verbose:
            print('Received: ', buf)

        try:
            buf.encode()
            if buf != None and buf.find('***shutdown***') >= 0:
                d.onShutDown()
                shutdownClient = True
                print('Client Shutdown')
                break
        except:
            if buf != None and buf.find(b'***shutdown***') >= 0:
                d.onShutDown()
                shutdownClient = True
                print('Client Shutdown')
                break

        try:
            buf.encode()
            if buf != None and buf.find('***restart***') >= 0:
                d.onRestart()
                print('Client Restart')
                break
        except:
            if buf != None and buf.find(b'***restart***') >= 0:
                d.onRestart()
                print('Client Restart')
                break

        currentStep += 1
        bufState = None
        action = None
        if currentStep != arguments.max_steps:
            if buf != None:
                buf, action, state, bufState = d.drive(buf.decode(), RL)
        else:
            buf = '(meta 1)'

        if verbose:
            print('Sending: ', buf)
        print(currentStep)
        if buf != None and oldStep != []:
            try:
                b = buf.encode()
                sock.sendto(b, (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
            #print(bufState.decode())
            reward = rw.lng_trans_prime(bufState)
            d.atualiza(RL, state, action, reward, oldStep)
            if (currentStep > 200):
                if (currentStep % 10 == 0):
                    RL.learn()

    curEpisode += 1

    if curEpisode == arguments.max_episodes:
        shutdownClient = True


sock.close()
