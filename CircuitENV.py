import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import yaml
import re

# 假设您已经有了这些辅助函数和类
from AcAdjoint import AcAdjointFunction
from AcSimulation import AcSimulationCuDSS, AcSimulationPardiso
from Circuit import Circuit, BranchType
from build_ckt import build_ac_ckt
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import torch.nn.functional as F


# class CustomPPOPolicy(ActorCriticPolicy):
#     def __init__(self, input_dim, output_dim, hidden_dim = 256):
#         super(CustomPPOPolicy, self).__init__()
#         # 您可以在这里添加自定义的神经网络层（如果需要）
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
        
#     def forward(self,x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         probs = F.softmax(self.fc3(x),dim=1)
#         return probs
        

#     def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
#         """
#         覆写父类的_predict方法来自定义动作选择。

#         :param observation: 当前的观测.
#         :param deterministic: 是否使用确定性策略.
#         :return: 选择的动作.
#         """
#         # --- 1. 首先，获取原始策略会选择的动作 ---
#         # Actor网络输出动作的概率分布
#         latent_pi, latent_vf, latent_sde = self._get_latent(observation)
#         distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        
#         if deterministic:
#             # 在确定性模式下，选择概率最高的动作
#             original_action = distribution.get_actions(deterministic=True)
#         else:
#             # 在随机模式下，根据概率分布采样
#             original_action = distribution.get_actions(deterministic=False)

#         # --- 2. 在这里加入您的自定义逻辑，修改 original_action ---
#         #
#         # **示例**: 假设我们有一个规则，如果总违规量（假设它是观测的最后一个特征）
#         # 超过某个阈值，就强制让模型在前50个动作中选择，以求稳定。
#         #
#         # 注意：这是一个高度简化的教学示例！
#         #
#         custom_action = original_action.clone() # 复制一份，避免原地修改
#         total_violation_feature_index = -len(self.observation_space.shape) # 假设违规量在观测的某个位置
        
#         # 假设我们能从 observation 中提取出 total_violation
#         # 实际操作中，您可能需要更复杂的逻辑来解析 observation
#         # 这里我们仅作示意
#         # if observation[0, total_violation_feature_index] > SOME_THRESHOLD:
#         #     if custom_action.item() >= 50:
#         #         # 强制选择一个前50的随机有效动作
#         #         # 这里的逻辑会很复杂，需要结合 action_mask
#         #         # 此处仅打印信息作为示例
#         #         print(f"--- 自定义策略触发：违规量过大，原动作 {custom_action.item()} 可能被否决 ---")

#         return custom_action


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, hidden_dim: int = 256):
        # features_dim 需要手动计算，等于所有部分特征拼接后的总维度
        # 我们让线性部分的输出维度和CNN部分的输出维度都为 cnn_output_dim
        features_dim = hidden_dim * 3 # impedance + target_z + decap_map
        super().__init__(observation_space, features_dim)

        # 提取每个输入的维度信息
        impedance_dim = observation_space["impedance"].shape[0]
        target_z_dim = observation_space["target_z"].shape[0]
        decap_map_shape = observation_space["decap_map"].shape # (width, height)
        decap_map_dim_flat = decap_map_shape[0] * decap_map_shape[1]
        # 我们将每个部分都映射到指定的 hidden_dim
        self.impedance_fc = nn.Linear(impedance_dim, hidden_dim)
        self.target_z_fc = nn.Linear(target_z_dim, hidden_dim)
        self.decap_map_fc = nn.Linear(decap_map_dim_flat, hidden_dim)
        
        # # --- 为1D向量数据（阻抗、目标）创建处理流水线 (全连接层) ---
        # self.vector_net = nn.Sequential(
        #     nn.Linear(impedance_dim + target_z_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, cnn_output_dim) # 输出一个256维的特征
        # )

        # # --- 为2D网格数据（电容地图）创建处理流水线 (卷积层) ---
        # # 这个CNN结构参考了通用图像处理网络，您可以根据需要调整
        # self.cnn = nn.Sequential(
        #     # 输入通道为1，因为我们的电容地图是单层的
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # stride=2会缩小图像尺寸
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(), # 将卷积后的二维特征图展平
        # )
        
        # # 计算CNN展平后的输出维度
        # # 我们需要用一个假的输入来动态计算这个维度
        # with torch.no_grad():
        #     dummy_input = torch.as_tensor(np.zeros((1, 1) + decap_map_shape))
        #     cnn_output_shape = self.cnn(dummy_input).shape[1]

        # # 再接一个线性层，将展平后的特征映射到最终的维度
        # self.cnn_linear = nn.Sequential(
        #     nn.Linear(cnn_output_shape, cnn_output_dim),
        #     nn.ReLU()
        # )
        
        # # (为了简化，我们将 impedance, target_z, 和 decap_map 分开处理)
        # # 论文中是将所有东西都处理成16x16再用卷积，这里我们做一个简化版，效果通常也很好
        # # 下面是一个更符合您之前思路的、更简单的全连接版本
        # self.impedance_fc = nn.Linear(impedance_dim, 128)
        # self.target_z_fc = nn.Linear(target_z_dim, 128)
        # self.decap_map_fc = nn.Linear(decap_map_shape[0] * decap_map_shape[1], 128) # 将二维地图展平后处理
        
        # # 重新计算总特征维度
        # self._features_dim = 128 + 128 + 128
        
    def forward(self, observations: dict) -> torch.Tensor:
        impedance = observations["impedance"]
        target_z = observations["target_z"]
        decap_map = observations["decap_map"]

        decap_map_flat = torch.flatten(decap_map, start_dim=1)

        impedance_features = F.relu(self.impedance_fc(impedance))
        target_z_features = F.relu(self.target_z_fc(target_z))
        decap_map_features = F.relu(self.decap_map_fc(decap_map_flat))
        
        concatenated_features = torch.cat([impedance_features, target_z_features, decap_map_features], dim=1)
        
        return concatenated_features


class CircuitEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self, ckt_file, result_file, render_mode=None):
        super(CircuitEnv, self).__init__()

        # ------------------ 1. 初始化环境参数 ------------------
        with open(ckt_file, "r") as f:
            design = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(result_file, "r") as f:
            self.initial_result = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.vdd = design["vdd"]
        self.target_impedance_value = 0.1 * self.vdd * self.vdd / design["chiplets"][0]["power"]
        
        self.base_ckt = build_ac_ckt(ckt_file, result_file)
        self.frequency_points = np.geomspace(0.1e9, 10e9, 100) # 100个频率点
        
        # 准备候选电容列表
        self.interposer_w = design["interposer"]["w"]
        self.interposer_h = design["interposer"]["h"]
        self.initial_candidate_branch = [
            f"cd_{x}_{y}" for x in range(self.interposer_w) for y in range(self.interposer_h)
            if (x, y) not in self.initial_result["tsvs"]
        ]
        
        self.num_initial_candidates = len(self.initial_candidate_branch)

        
        # ------------------ 2. 定义观测和动作空间 ------------------
        # 观测空间：100个频率点的阻抗值
        num_freq_points = len(self.frequency_points)
        obs_low = np.zeros(num_freq_points , dtype=np.float64)
        obs_high = np.full(num_freq_points , np.inf, dtype=np.float64)
        obs_high[-1] = num_freq_points  # 计数的最大值是总频点数
        
        # 观测空间：100个频率点的阻抗值 + 1个低于目标的频点计数值
        obs_shape = (num_freq_points + self.num_initial_candidates,)
        # self.observation_space = spaces.Box(low=0, high=np.inf, shape=obs_shape, dtype=np.float64)
        self.observation_space = spaces.Dict({
            "impedance": spaces.Box(low=0, high=np.inf, shape=(num_freq_points,), dtype=np.float64),
            "target_z": spaces.Box(low=0, high=np.inf, shape=(num_freq_points,), dtype=np.float64),
            "decap_map": spaces.Box(low=0, high=1, shape=(self.interposer_w, self.interposer_h), dtype=np.float64)
        })

        # self.observation_space = spaces.Box(low=obs_low, high=obs_high,  shape=obs_shape, dtype=np.float64)
        
        # self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(self.frequency_points),), dtype=np.float64)
        
        # 动作空间：从所有初始候选位置中选择一个。这是一个离散的动作。
        self.num_initial_candidates = len(self.initial_candidate_branch)
        self.action_space = spaces.Discrete(self.num_initial_candidates)
        print(self.action_space)

        # ------------------ 3. 准备仿真所需的静态参数 ------------------
        self._prepare_sim_tensors()
        self.count_below_target = 0
        
        self.render_mode = render_mode

    def _prepare_sim_tensors(self):
        # 这部分代码源自您的脚本，用于准备仿真参数
        observe_branch = ["id"]
        typ, u, v, val, index = self.base_ckt.prepare_sim("0", observe_branch + self.initial_candidate_branch)
        
        self.typ, self.u, self.v = typ, u, v
        self.g_index = torch.tensor([], dtype=torch.long)
        self.r_index = torch.tensor([], dtype=torch.long)
        self.c_index_map = index[len(observe_branch):] # 所有候选电容在val中的索引
        self.l_index = torch.tensor([], dtype=torch.long)
        self.i_index = index[:len(observe_branch)]
        self.v_index = torch.tensor([], dtype=torch.long)
        (self.all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
        
        self.g_value = torch.tensor([], dtype=torch.float64)
        self.r_value = torch.tensor([], dtype=torch.float64)
        self.initial_c_base_value = val[self.c_index_map]
        self.l_value = torch.tensor([], dtype=torch.float64)
        self.all_exc_value = val[self.all_exc_index]
        
        # 创建仿真器实例
        self.sims = [AcSimulationPardiso(typ, u, v, val, freq) for freq in self.frequency_points]

    def _get_impedances(self):
        # 辅助函数，只计算阻抗，不拼接额外信息
        zs = []
        c_value_tensor = torch.zeros_like(self.initial_c_base_value)
        if self.placed_capacitor_indices:
            placed_indices_tensor = torch.tensor(self.placed_capacitor_indices)
            c_value_tensor[placed_indices_tensor] = self.initial_c_base_value[placed_indices_tensor]

        c_index_tensor = self.c_index_map
        for freq, sim in zip(self.frequency_points, self.sims):
            # i_voltage, _ = AcAdjointFunction.apply(
            #     self.g_index, self.r_index, c_index_tensor, self.l_index,
            #     torch.tensor([],dtype=torch.long), torch.tensor([],dtype=torch.long),
            #     self.i_index, self.v_index, self.all_exc_index,
            #     self.g_value, self.r_value, c_value_tensor, self.l_value,
            #     torch.tensor([],dtype=torch.float64), torch.tensor([],dtype=torch.float64),
            #     self.all_exc_value, freq, sim
            # )
            # zs.append(i_voltage)
            sim.alter(c_index_tensor, c_value_tensor)
            sim.factorize()
            sim.solve()
            i_voltage = sim.branch_voltage(self.i_index)
            zs.append(i_voltage)
        
        return torch.cat(zs).abs().detach().numpy()
    def _get_obs(self):
        # 1. 获取当前的阻抗曲线
        impedances_np = self._get_impedances()
        
        # 2. 创建当前的电容地图 (decap map)
        # 这是一个长度等于总候选位置数的向量，初始为0
        # decap_map = np.zeros(self.num_initial_candidates, dtype=np.float32)
        decap_map = np.zeros((self.interposer_w, self.interposer_h), dtype=np.float64)

        # if self.placed_capacitor_indices:
        #     # 将已放置电容的位置索引处标记为1
        #     decap_map[self.placed_capacitor_indices] = 1.0
        
        # 3. 遍历所有已放置的电容
        if self.placed_capacitor_indices:
            for action_index in self.placed_capacitor_indices:
                # 从我们之前存储的信息中，查找该动作索引对应的 x, y 坐标
                info = self.initial_candidate_branch[action_index]
                # print(f"Placed capacitor at ({info})")
                # 使用正则表达式匹配数字
                match = re.search(r"cd_(\d+)_(\d+)", info)
                
                if match:
                    # 从匹配结果中提取 x 和 y，并转换为整数
                    x = int(match.group(1))
                    y = int(match.group(2))

                    # 在二维地图上将该位置标记为1
                    decap_map[x, y] = 1.0
            
        # 3. 将两者拼接成最终的观测向量
        # observation = np.concatenate([impedances_np, decap_map])
        
        # return observation.astype(np.float64)
        return {
            "impedance": impedances_np,
            "target_z": np.full_like(impedances_np, self.target_impedance_value),
            "decap_map": decap_map
        }

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.num_initial_candidates, dtype=bool) # 使用 bool 类型更标准
        if self.placed_capacitor_indices: 
            mask[self.placed_capacitor_indices] = False
        return mask

    def _get_info(self):
        """
        现在这个函数只负责返回信息字典。
        掩码的计算逻辑已经移到了 action_masks() 中。
        """
        return {"action_mask": self.action_masks()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置已放置电容列表
        self.placed_capacitor_indices = []
        self.current_step = 0
        
        # 获取初始观测
        observation = self._get_obs()
        # self.peak_impedance = np.max(observation) # 记录初始峰值阻抗
        # self.peak_impedance = np.max(observation[:-1]) 
        # self.count_below_target = observation[-1]
        impedances_part = observation["impedance"][:len(self.frequency_points)]
        impedance_violation = np.maximum(impedances_part - self.target_impedance_value, 0)
        self.total_violation = np.sum(impedance_violation)
        
        
        # 获取信息（包含初始的动作掩码）
        info = self._get_info()
        
        
        if self.render_mode == "console":
            self.render(observation, 0)

        return observation, info

    def step(self, action):
        # 检查动作是否有效 (通过掩码)
        current_mask = self._get_info()["action_mask"]
        old_obs = self._get_obs() # 需要在放置前获取一次旧状态
        # 计算违例的频率点数
        old_impedances = old_obs["impedance"][:len(self.frequency_points)]
        N_t = np.sum(old_impedances<=self.target_impedance_value)

        # 执行动作：将选择的电容位置记录下来
        # 获取旧的达标点数
        # old_obs = self._get_obs() # 需要在放置前获取一次旧状态
        # old_count_below_target = old_obs[-1]
        if action in self.placed_capacitor_indices:
            print(f"!!! 警告：位置 {action} 已被放置，请选择其他位置。 !!!")
            # 如果位置已经被放置，给予小惩罚并返回旧状态
            
        self.placed_capacitor_indices.append(action)
        self.current_step += 1

        # 获取新的观测值
        observation = self._get_obs()
        # impedances_part = observation[:-1]
        # new_count_below_target = observation[-1] # 新的达标点数
        impedances_part = observation["impedance"][:len(self.frequency_points)]
        N_t1= np.sum(impedances_part<=self.target_impedance_value)
        improvement_reward = (N_t1 - N_t) * 5.0 # 放大这个奖励
        terminated  = False
        truncated  = False
        if len(self.placed_capacitor_indices) == self.num_initial_candidates and np.sum(old_obs[:len(self.frequency_points)]>self.target_impedance_value)>0:
            T = -100
            truncated = True
        else:
            T = self.num_initial_candidates - self.current_step
            
        
        reward = improvement_reward/self.num_initial_candidates + T
        # impedance_violation = np.maximum(impedances_part - self.target_impedance_value, 0)
        # new_total_violation = np.sum(impedance_violation)

        # # new_peak_impedance = np.max(impedances_part)
        # # self.count_below_target = new_count_below_target
        

        # # --- 计算奖励 ---
        # violation_reduction = self.total_violation - new_total_violation
        # reward = violation_reduction * 100 # 放大这个奖励，让它成为主导
        # self.total_violation = new_total_violation
        
        # # 3. 惩罚电容数量：每放置一个电容，就有一个固定的负奖励
        # cap_count_penalty = -1.0 
        # reward += cap_count_penalty
        
        # # 奖励核心：峰值阻抗降低得越多，奖励越高
        # reward = self.peak_impedance - new_peak_impedance 
        # self.peak_impedance = new_peak_impedance # 更新峰值阻抗
        
        # reward += (new_count_below_target - old_count_below_target) * 0.5 # 每多一个点达标，就给0.5的额外奖励

        
        # # 每放置一个电容给一个小的负奖励，鼓励用更少的电容
        # reward -= 0.1

        # --- 判断回合是否结束 ---
        # 1. 成功：所有点的阻抗都低于目标
        # terminated = np.all(observation < self.target_impedance_value)
        # terminated = bool(np.all(observation <= self.target_impedance_value))
        # if terminated:
        #     reward += 500 # 成功给予巨大奖励
        #     print(f"放置成功!Placed dtc:{len(self.placed_capacitor_indices)}, peak_impedance:{self.peak_impedance}")
        # if self.total_violation < 1e-6: # 用一个很小的值判断是否为0
        #     terminated = True
        
        if N_t1==len(self.frequency_points):
            reward += 200 # 成功时给予巨大的额外奖励
            terminated = True
            print(f"放置成功! Placed dtc:{len(self.placed_capacitor_indices)}, peak_impedance:{self.target_impedance_value}")
            
            
        # 2. 截断：所有候选位置都已用完，或达到最大步数
        # truncated = (len(self.placed_capacitor_indices) == self.num_initial_candidates) or (self.current_step >= self.num_initial_candidates)

        info = self._get_info() # 获取更新后的动作掩码

        if self.render_mode == "console":
            self.render(observation, reward)
            
        return observation, reward, terminated, truncated, info

    def render(self, observation, reward, mode='console'):
        if mode == 'console':
            num_freq_points = len(self.frequency_points)
            impedances_part = observation["impedance"][:num_freq_points]

            # 2. 现在，在正确的阻抗部分上进行计算
            peak_impedance = np.max(impedances_part)
            points_below_target = np.sum(impedances_part <= self.target_impedance_value)

            # 3. 打印您需要的关键信息
            print(
                f"Step: {self.current_step}, "
                # f"Placed Caps: {len(self.placed_capacitor_indices)}, "
                # f"Placed Caps_info: {(self.placed_capacitor_indices)},"
                f"reward: {reward:.4f}, "
                f"Peak Impedance: {peak_impedance:.4f}, "
                f"Target Impedance: {self.target_impedance_value:.4f}, "
                f"Points Below Target: {points_below_target}/{num_freq_points}"
            )
