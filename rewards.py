import numpy as np

def lng_trans_prime(obs):
	"""
	Reward longitudal velocity projected on track axis,
	with a penality for transverse velocity.
	"""
	#print(obs['speedX'])
	speedX = np.array([float(i) for i in obs['speedX']])
	# Track distance
	trackPos = np.array([float(i) for i in obs['trackPos']])
	angle = float(obs['angle'][0])
	reward = speedX * np.cos(angle) - np.abs(speedX * np.sin(angle)) - np.abs(speedX * angle)
	return reward
