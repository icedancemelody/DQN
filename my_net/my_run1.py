from DQN.my_env import Env
from DQN.my_net.RL_brain1 import SumDQN
import tensorflow as tf
import numpy as np

def run_maze():
    step = 0
    reward_sum=0#每十回合统计的奖励
    step_sum=0#每十回合统计的步数
    for episode in range(1001):
        # initial observation
        observation = env.reset()

        step_counter = 1
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            #RL.store_transition(observation, action, reward, observation_)

            # if (step > 100) and (step % 5 == 0):
            #     RL.learn()

            # swap observation
            observation = observation_

            env.update_env(episode, step_counter,done,reward)
            reward_sum+=reward

            if done:
                step_sum += step_counter
                break
            step += 1
            step_counter+=1

        if (episode%10==0) &(episode!=0):
            env.Reward_memory(episode/10,reward_sum/10,step_sum/10)
            reward_sum=0
            step_sum=0
    # end of game
    env.draw()
    print('game over')


if __name__ == "__main__":
    # maze game
    env = Env()
    RL = SumDQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=10000,
                      double_q=True,
                      prioritized=True,
                      dueling=True
                      # output_graph=True
                      )
    run_maze()



    # saver=tf.train.Saver()
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     save_path=saver.save(sess,"../my_net/save_net.ckpt")
    #     print("Save to path:",save_path)



