import tensorflow as tf
import numpy as np
import sys
import os
import time
import math
import datetime

learning_rate_index = 2
gamma_index = 2
#test_or_train = sys.argv[3]
# learning_rate_index = int(sys.argv[1])
# gamma_index = int(sys.argv[2])
# test_or_train = sys.argv[3]

## ENVIRONMENT Hyperparameters
state_size = 30
action_size = 4

## TRAINING Hyperparameters
min_episodes_exp = 9
max_episodes_exp = 32
max_episodes = 2**max_episodes_exp
max_steps = 1024 + 256
max_episodes_evaluate = 128

learning_rate_list = [
	2**(-3),
	2**(-5),
	2**(-7), # Main learning rate
	2**(-9),
	2**(-11),
	2**(-13),
	2**(-15)
]
learning_rate = learning_rate_list[learning_rate_index]

gamma_list = [
	1 - 2**(-2) - 2**(-3),
	1 - 2**(-4) - 2**(-5),
	1 - 2**(-6) - 2**(-7), # Main gamma
	1 - 2**(-8) - 2**(-9)
]
gamma = gamma_list[gamma_index] # Discount rate

def discount_and_normalize_rewards(episode_rewards):
	discounted_episode_rewards = np.zeros_like(episode_rewards)
	cumulative = 0.0
	for i in reversed(range(len(episode_rewards))):
		cumulative = cumulative * gamma + episode_rewards[i]
		discounted_episode_rewards[i] = cumulative

	mean = np.mean(discounted_episode_rewards)
	std = np.std(discounted_episode_rewards)
	discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

	return discounted_episode_rewards


with tf.name_scope("inputs"):
	input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
	actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
	discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")

	# Add this placeholder for having this variable in tensorboard
	mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")
	# and this
	episode_rewards_sum_ = tf.placeholder(tf.float32 , name="episode_rewards_sum")

	with tf.name_scope("fc1"):
		fc1 = tf.contrib.layers.fully_connected(inputs = input_,
												num_outputs = 600,
												activation_fn = tf.nn.relu,
												weights_initializer = tf.contrib.layers.xavier_initializer(seed=4937))

	with tf.name_scope("fc2"):
		fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
												num_outputs = 300,
												activation_fn = tf.nn.relu,
												weights_initializer = tf.contrib.layers.xavier_initializer(seed=1337))

	with tf.name_scope("fc3"):
		fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
												num_outputs = action_size,
												activation_fn = None,
												weights_initializer = tf.contrib.layers.xavier_initializer(seed=41))

	with tf.name_scope("softmax"):
		action_distribution = tf.nn.softmax(fc3)

	with tf.name_scope("loss"):
		# tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
		# If you have single-class labels, where an object can only belong to one class, you might now consider using
		# tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
		neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
		loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)

	with tf.name_scope("train"):
		train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def lng_trans_prime(obs):
	"""
	Reward longitudal velocity projected on track axis,
	with a penality for transverse velocity.
	"""
	speedX = np.array([float(i) for i in obs['speedX']])
	# Track distance
	trackPos = np.array([float(i) for i in obs['trackPos']])
	angle = float(obs['angle'][0])
	reward = speedX * np.cos(angle) - np.abs(speedX * np.sin(angle)) - np.abs(speedX * angle)
	return reward

allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

writer = tf.summary.FileWriter("./results/log/lr_" + str(learning_rate) + "/g_" + str(gamma) + "/")
step = 0
def train(msg, sess):
	global step
	with tf.device('/device:GPU:0'):
		np.random.seed(4937)
		## Losses
		tf.summary.scalar("Loss", loss)
		## Reward mean
		tf.summary.scalar("Reward_mean", mean_reward_)
		## Each Step Reward
		tf.summary.scalar("Reward", episode_rewards_sum_)
		write_op = tf.summary.merge_all()

		episode_rewards_sum = 0

		# Launch the game
		state = msg
		#(angle 0.00894148)(curLapTime -0.982)(damage 0)(distFromStart 5759.1)(distRaced 0)(gear 0)(lastLapTime 0)
		#(racePos 1)(rpm 942.478)(speedX 0.000308602)(speedY 0.00128389)(speedZ -0.000193009)
		#(track 7.33374 7.60922 8.50537 10.4385 14.7757 21.468 27.865 39.8075 70.6072 200 49.8655 21.1133 13.9621 10.5624 7.24717 5.14528 4.2136 3.78723 3.66663)
		#(trackPos -0.333363)(z 0.345256)

		# env.render()
		step += 1
		n = []
		for key, val in msg.items():
			for i in val:
				n.append(i)
		new = np.array(n)
		# Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
		action_probability_distribution = sess.run(action_distribution, feed_dict={input_: new.reshape([1, state_size])})
		t1 = datetime.datetime.now()
		#print(action_probability_distribution.ravel())

		action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
#		print(action_probability_distribution)
		t2 = datetime.datetime.now()
		print(t2-t1)
		# Perform a
		reward = lng_trans_prime(msg)

		# Store s
		episode_states.append(state)

		# One-Hot Encoding
		action_ = np.zeros(action_size)
		action_[action] = 1

		# Store a
		episode_actions.append(action_)

		# Store r
		episode_rewards.append(reward)
		return action

def endTrain():
	global step
	with tf.Session(config = config) as sess:
		# Calculate sum reward
		episode_rewards_sum = np.sum(episode_rewards)

		allRewards.append(episode_rewards_sum)

		total_rewards = np.sum(allRewards)

		# Mean reward
		mean_reward = np.divide(total_rewards, episode+1)

		maximumRewardRecorded = np.amax(allRewards)

		print("==========================================")
		print("Episode:", episode)
		print("Reward:", episode_rewards_sum)
		print("Steps for this Episode:", step)
		print("Mean Reward:", mean_reward)
		print("Max reward so far:", maximumRewardRecorded)

		# Calculate discounted reward
		discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

		# Feedforward, gradient and backpropagation
		loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
														 actions: np.vstack(np.array(episode_actions)),
														 discounted_episode_rewards_: discounted_episode_rewards
														})

		# Write TF Summaries
		summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
														 actions: np.vstack(np.array(episode_actions)),
														 discounted_episode_rewards_: discounted_episode_rewards,
															mean_reward_: mean_reward,
															episode_rewards_sum_: episode_rewards_sum
														})
		step = 0
		# if episode + 1 in [2**i for i in range(min_episodes_exp, max_episodes_exp + 1)]:
		# 	# Save Model
		# 	saver.save(sess, ckpt_paths[int(math.log(episode + 1, 2)) - min_episodes_exp])
		# 	print("==========================================")
		# 	print("Model saved")
		#
		# writer.add_summary(summary, episode)
		# writer.flush()
#
# def test():
# 	with tf.device('/device:GPU:0'):
#
# 		env.seed(4937)
# 		np.random.seed(71)
#
# 		with tf.Session() as sess:
# 			# Load the model
# 			saver.restore(sess, ckpt_path)
#
# 			rewards = []
# 			for episode in range(max_episodes_evaluate):
# 				state = env.reset()
# 				done = False
# 				total_rewards = 0
# 				print("****************************************************")
# 				print("EPISODE", episode)
#
# 				step = 0
# 				while True:
# 					step += 1
# 					# Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
# 					action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, state_size])})
# 					#print(action_probability_distribution)
# 					action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
#
# 					new_state, reward, done, info = env.step(action)
#
# 					total_rewards += reward
#
# 					if done or (step == max_steps):
# 						rewards.append(total_rewards)
# 						print("Score:", total_rewards)
# 						break
# 					state = new_state
#
# 			env.close()
# 			print("****************************************************")
# 			print("Score over time:", str(sum(rewards) / max_episodes_evaluate))
# 			print("****************************************************")
#
# def watch():
# 	with tf.Session() as sess:
# 		# Load the model
# 		saver.restore(sess, ckpt_path)
#
# 		while True:
# 			state = env.reset()
# 			done = False
# 			total_rewards = 0
#
# 			step = 0
# 			while True:
# 				step += 1
# 				env.render()
# 				# Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
# 				action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1, state_size])})
# 				#print(action_probability_distribution)
# 				action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
#
# 				new_state, reward, done, info = env.step(action)
#
# 				total_rewards += reward
#
# 				if done or (step == max_steps):
# 					print("Score:", total_rewards)
# 					break
#
# 				state = new_state
#
# # Setup TensorBoard Writer
# ckpt_folder = "results/lr_" + str(learning_rate) + "/g_" + str(gamma) + "/"
# ckpt_filename = "s_" + str(max_steps) + "-lr_" + str(learning_rate) + "-g_" + str(gamma)
# ckpt_paths = []
# for i in range(min_episodes_exp, max_episodes_exp + 1):
# 	ckpt_paths.append("./" + ckpt_folder + ckpt_filename + "_" + str(2**i) + ".ckpt")
#
# saver = tf.train.Saver()
#
# if test_or_train == 'train':
# 	if not os.path.exists(ckpt_folder):
# 		os.makedirs(ckpt_folder)
# 	train()
#
# else:
# 	test_episodes = int(sys.argv[4]) - min_episodes_exp
#
# 	if not os.path.isfile(ckpt_paths[test_episodes] + ".index"):
# 		print("Checkpoint not found!")
# 		exit()
#
# 	ckpt_path = ckpt_paths[test_episodes]
# 	test()
# 	watch()
