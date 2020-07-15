"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class SumDQN:
    # change
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.95,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            prioritized=True,
            dueling=True,
            sess=None,
    ):
        self.model_name = "my_net/save_net.ckpt"
        self.sess=tf.Session()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q  # decide to use double q or not
        self.prioritized = prioritized
        self.dueling = dueling

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        # def build_layers(s, n_l1, trainable=False):
        n_l1=20
        self.s=tf.placeholder(tf.float32, [None, self.n_features], name='s')

        with tf.variable_scope('eval_net'):
            with tf.variable_scope('l1'):
                # 建立w,b容器
                w1 = tf.Variable(np.arange(self.n_features * n_l1).reshape((self.n_features, n_l1)), dtype=tf.float32,name="w1")

                b1 = tf.Variable(np.arange(1 * n_l1).reshape((1, n_l1)), dtype=tf.float32, name="b1")
                # b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)

                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # change
            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w2 = tf.Variable(np.arange(n_l1*1).reshape((n_l1,1)),
                                     dtype=tf.float32, name="w2")
                    b2 = tf.Variable(np.arange(1 * 1).reshape((1, 1)), dtype=tf.float32, name="b2")
                    self.V = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Advantage'):
                    w2 = tf.Variable(np.arange(n_l1 * self.n_actions).reshape((n_l1, self.n_actions)), dtype=tf.float32,
                                     name="w2")
                    b2 = tf.Variable(np.arange(1 * self.n_actions).reshape((1, self.n_actions)), dtype=tf.float32,
                                     name="b2")
                    self.A = tf.matmul(l1, w2) + b2

                with tf.variable_scope('Q'):
                    self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))  # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w2 = tf.Variable(np.arange(n_l1 * self.n_actions).reshape((n_l1, self.n_actions)), dtype=tf.float32,
                                     name="w2")
                    b2 = tf.Variable(np.arange(1 * self.n_actions).reshape((1, self.n_actions)), dtype=tf.float32,
                                     name="b2")
                    self.q_eval = tf.matmul(l1, w2) + b2

        # return out

        # ------------------ build evaluate_net ------------------
        #self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        # with tf.variable_scope('eval_net'):
        #     c_names, n_l1, w_initializer, b_initializer = \
        #         ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
        #         tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

        #self.q_eval = build_layers(self.s, 20)

        # 读取模型参数
        #saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        #saver = tf.train.import_meta_graph("E:\Innovate projects\DQN-master\DQN\DQN\my_net\save_net.ckpt.meta")
        #saver=tf.train.Saver()
        saver=tf.train.Saver()
        saver.restore(self.sess, "save_net.ckpt")

        w1_info = tf.get_default_graph().get_tensor_by_name('eval_net/l1/w1:0')
        print(self.sess.run(w1_info))
        print(self.sess.run(w1))

        #graph = tf.get_default_graph()
        #w1 = graph.get_tensor_by_name("w1:0")
        #w2 = graph.get_tensor_by_name("w2:0")
        #print(self.sess.run(w1))  # 输出读取到的一组参数，检验是否成功读取

        # ------------------ build target_net ------------------
        # self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        # with tf.variable_scope('target_net'):
        #     c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        #     self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)

    # def store_transition(self, s, a, r, s_):
    #     if self.prioritized:  # prioritized replay
    #         transition = np.hstack((s, [a, r], s_))
    #         self.memory.store(transition)  # have high priority for newly arrived transition
    #     else:  # random replay
    #         if not hasattr(self, 'memory_counter'):
    #             self.memory_counter = 0
    #         transition = np.hstack((s, [a, r], s_))
    #         index = self.memory_counter % self.memory_size
    #         self.memory[index, :] = transition
    #         self.memory_counter += 1
    #

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        #if np.random.uniform() < self.epsilon:


        # else:
        #     action = np.random.randint(0, self.n_actions)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        return action
