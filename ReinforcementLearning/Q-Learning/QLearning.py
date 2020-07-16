import numpy as np
import pandas as pd
import time


class QLearning(object):

    def __init__(self,state_space, action_space, cur_state, learn_epsilon = 0.9, learn_alpha = 0.1, learn_gamma = 0.9):
        # 贪婪度
        self.learn_epsilon = learn_epsilon
        # 学习率
        self.learn_alpha = learn_alpha
        # 折扣率
        self.learn_gamma = learn_gamma
        # 定义状态空间和动作空间
        self.state_space = state_space
        self.action_space = action_space
        # 初始化当前状态：
        self.cur_state = cur_state
        # 选择的动作
        pass
        # 下一个状态
        pass
        # 奖励
        pass
        # 状态空间和动作空间的大小
        self.n_state = len(state_space)
        self.n_action = len(action_space)
        # 创建Q表
        self.q_table = pd.DataFrame(np.zeros((self.n_state,self.n_action)),columns=action_space,index=state_space)
    

    def chose_action(self, state):
        # 当前状态可用的动作集合对应的Q值
        en_states = self.q_table.iloc[state,:]
        if (np.random.uniform() > self.learn_epsilon) or (en_states.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
            action = np.random.choice(self.action_space)
        else:
            action = en_states.argmax()    # 贪婪模式
        return action


    def show_q_table(self):
        print(self.q_table)


    def show_rl_info(self):
        print("贪婪度为：", self.learn_epsilon)
        print("学习率为：", self.learn_alpha)
        print("折扣率为：", self.learn_gamma)


if __name__ == "__main__":
    # 最大回合数
    max_episodes = 13

    state_space = [0,1,2,3,4,5]
    action_space = ["left","right"]
    q_rl = QLearning(state_space,action_space)
    q_rl.show_rl_info()
    q_rl.show_q_table()
    #q_rl.chose_action(1)
