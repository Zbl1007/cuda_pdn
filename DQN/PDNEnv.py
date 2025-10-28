import numpy as np
import gym
from gym import spaces

class PDNEnv(gym.Env):
    """
    简化的PDN环境，用于解耦电容优化。
    - 状态(State): 一个2D网格，表示电容的布局。
    - 动作(Action): 在一个预定义的可用位置上放置一个电容。
    - 奖励(Reward): 基于放置电容后，满足目标阻抗的频点数量的提升。
    """

    def __init__(self, grid_size=(6, 4), num_available_locs=29, probing_port=(3, 1)):
        super(PDNEnv, self).__init__()

        self.grid_size = grid_size
        self.probing_port = probing_port

        # 定义动作空间：动作是选择第n个可用位置
        self.action_space = spaces.Discrete(num_available_locs)
        
        # 定义状态空间：一个2D网格，0表示没有电容，1表示有电容
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.float32)

        # 定义可用放置位置 (参考论文Test PDN #1的action space)
        # 这里为了简化，我们假设所有非探测端口的位置都可用
        self.available_locations = []
        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                # 简单处理：假设所有位置都可以放电容
                self.available_locations.append((r, c))
        # 确保动作空间大小与实际可用位置数量一致
        # 在真实复现中，这里的坐标列表应与论文严格对应
        self.action_space = spaces.Discrete(len(self.available_locations))


        # 简化物理模型参数
        self.freq_points = np.logspace(8, 10, 50)  # 100MHz to 10GHz, 50 points
        self.target_impedance = np.full(self.freq_points.shape, 0.5) # 目标阻抗: 0.5 Ohm
        
        # 初始阻抗模型：一个有很高峰值的阻抗曲线
        self.base_impedance = 10 / (1 + ((self.freq_points - 1e9) / 5e8)**2) + 1.0

        self.state = None
        self.current_impedance = None
        self.max_steps = num_available_locs # 最多把所有位置都放满
        self.current_step = 0

    def _calculate_impedance(self):
        """
        简化的阻抗计算模型。
        - 放置的每个电容都会降低阻抗。
        - 离探测端口越近，效果越好（模拟环路电感减小）。
        """
        total_reduction = np.zeros_like(self.freq_points)
        decap_positions = np.argwhere(self.state == 1)

        if len(decap_positions) == 0:
            return self.base_impedance

        for pos in decap_positions:
            distance = np.linalg.norm(np.array(pos) - np.array(self.probing_port))
            # 距离越近，效果越强；效果随频率变化
            effectiveness = 5.0 / (1 + distance)
            reduction_shape = 1 / (1 + ((self.freq_points - 5e9) / 4e9)**2) # 假设在中高频效果最好
            total_reduction += effectiveness * reduction_shape
        
        # 增加电容总量效果（对低频影响更大）
        num_decaps = len(decap_positions)
        capacitance_effect = 1 / (1 + 1e8/self.freq_points) * (num_decaps * 0.2)
        total_reduction += capacitance_effect

        return np.maximum(0.1, self.base_impedance - total_reduction)

    def _calculate_reward(self, old_impedance, new_impedance):
        """
        根据论文中的公式(5)简化奖励计算。
        Rt = (Nt+1 - Nt) / N + T
        """
        N = len(self.freq_points)
        
        Nt = np.sum(old_impedance <= self.target_impedance)
        Nt_plus_1 = np.sum(new_impedance <= self.target_impedance)

        # 检查是否完全满足目标 (T)
        T = 1.0 if Nt_plus_1 == N else 0.0
        
        reward = (Nt_plus_1 - Nt) / N + T
        return reward

    def step(self, action):
        """
        执行一个动作，更新环境状态。
        """
        self.current_step += 1
        
        # 检查动作是否有效（在该位置是否已经有电容）
        action_pos = self.available_locations[action]
        if self.state[action_pos] == 1:
            # 如果重复放置，给予一个小的负奖励，状态不变
            return self.state, -0.1, self.current_step >= self.max_steps, {}

        # 获取旧阻抗
        old_impedance = self.current_impedance

        # 在新位置放置电容
        self.state[action_pos] = 1

        # 计算新阻抗和奖励
        self.current_impedance = self._calculate_impedance()
        reward = self._calculate_reward(old_impedance, self.current_impedance)

        # 检查终止条件
        done = (np.all(self.current_impedance <= self.target_impedance)) or (self.current_step >= self.max_steps)
        
        # 如果完成目标，给予一个大的额外奖励
        if np.all(self.current_impedance <= self.target_impedance):
            reward += 10 # 大奖励鼓励尽快完成目标

        return self.state, reward, done, {}

    def reset(self):
        """
        重置环境到初始状态。
        """
        self.state = np.zeros(self.grid_size, dtype=np.float32)
        self.current_impedance = self._calculate_impedance()
        self.current_step = 0
        return self.state

    def render(self, mode='human'):
        """
        可视化当前电容布局。
        """
        print("+" + "----+" * self.grid_size[1])
        for r in range(self.grid_size[0]):
            row_str = "|"
            for c in range(self.grid_size[1]):
                if (r, c) == self.probing_port:
                    char = " P  " # Probing Port
                elif self.state[r, c] == 1:
                    char = " D  " # Decap
                else:
                    char = "    "
                row_str += char + "|"
            print(row_str)
            print("+" + "----+" * self.grid_size[1])
        print(f"Steps: {self.current_step}")