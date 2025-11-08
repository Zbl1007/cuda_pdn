import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import yaml
import re
from AcSimulation import AcSimulationCuDSS, AcSimulationPardiso
from build_ckt import build_ac_ckt
from Circuit import Circuit, BranchType

import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict, hidden_dim: int = 256):
#         # features_dim 需要手动计算，等于所有部分特征拼接后的总维度
#         # 我们让线性部分的输出维度和CNN部分的输出维度都为 cnn_output_dim
#         features_dim = hidden_dim * 3 # impedance + target_z + decap_map
#         super().__init__(observation_space, features_dim)

#         # 提取每个输入的维度信息
#         impedance_dim = observation_space["impedance"].shape[0]
#         target_z_dim = observation_space["target_z"].shape[0]
#         decap_map_shape = observation_space["decap_map"].shape # (width, height)
#         decap_map_dim_flat = decap_map_shape[0] * decap_map_shape[1]
#         # 我们将每个部分都映射到指定的 hidden_dim
#         self.impedance_fc = nn.Linear(impedance_dim, hidden_dim)
#         self.target_z_fc = nn.Linear(target_z_dim, hidden_dim)
#         self.decap_map_fc = nn.Linear(decap_map_dim_flat, hidden_dim)
        
#         # # --- 为1D向量数据（阻抗、目标）创建处理流水线 (全连接层) ---
#         # self.vector_net = nn.Sequential(
#         #     nn.Linear(impedance_dim + target_z_dim, 256),
#         #     nn.ReLU(),
#         #     nn.Linear(256, cnn_output_dim) # 输出一个256维的特征
#         # )

#         # # --- 为2D网格数据（电容地图）创建处理流水线 (卷积层) ---
#         # # 这个CNN结构参考了通用图像处理网络，您可以根据需要调整
#         # self.cnn = nn.Sequential(
#         #     # 输入通道为1，因为我们的电容地图是单层的
#         #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#         #     nn.ReLU(),
#         #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # stride=2会缩小图像尺寸
#         #     nn.ReLU(),
#         #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
#         #     nn.ReLU(),
#         #     nn.Flatten(), # 将卷积后的二维特征图展平
#         # )
        
#         # # 计算CNN展平后的输出维度
#         # # 我们需要用一个假的输入来动态计算这个维度
#         # with torch.no_grad():
#         #     dummy_input = torch.as_tensor(np.zeros((1, 1) + decap_map_shape))
#         #     cnn_output_shape = self.cnn(dummy_input).shape[1]

#         # # 再接一个线性层，将展平后的特征映射到最终的维度
#         # self.cnn_linear = nn.Sequential(
#         #     nn.Linear(cnn_output_shape, cnn_output_dim),
#         #     nn.ReLU()
#         # )
        
#         # # (为了简化，我们将 impedance, target_z, 和 decap_map 分开处理)
#         # # 论文中是将所有东西都处理成16x16再用卷积，这里我们做一个简化版，效果通常也很好
#         # # 下面是一个更符合您之前思路的、更简单的全连接版本
#         # self.impedance_fc = nn.Linear(impedance_dim, 128)
#         # self.target_z_fc = nn.Linear(target_z_dim, 128)
#         # self.decap_map_fc = nn.Linear(decap_map_shape[0] * decap_map_shape[1], 128) # 将二维地图展平后处理
        
#         # # 重新计算总特征维度
#         # self._features_dim = 128 + 128 + 128
        
#     def forward(self, observations: dict) -> torch.Tensor:
#         impedance = observations["impedance"]
#         target_z = observations["target_z"]
#         decap_map = observations["decap_map"]

#         decap_map_flat = torch.flatten(decap_map, start_dim=1)

#         impedance_features = F.relu(self.impedance_fc(impedance))
#         target_z_features = F.relu(self.target_z_fc(target_z))
#         decap_map_features = F.relu(self.decap_map_fc(decap_map_flat))
        
#         concatenated_features = torch.cat([impedance_features, target_z_features, decap_map_features], dim=1)
        
#         return concatenated_features

## 旧版本
# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, hidden_dim=256):
#         super().__init__(observation_space, self._features_dim)

#         # 1. 提取维度信息
#         impedance_dim = observation_space["impedance"].shape[0]
#         target_z_dim = observation_space["target_z"].shape[0]
#         decap_shape = observation_space["decap_map"].shape  # (W, H)
#         context_shape = observation_space["context_map"].shape  # (C, W, H)

#         self.hidden_dim = hidden_dim

#         # 2. 全连接：阻抗和目标阻抗
#         self.impedance_fc = nn.Sequential(
#             nn.Linear(impedance_dim, hidden_dim),
#             nn.ReLU()
#         )
#         self.target_z_fc = nn.Sequential(
#             nn.Linear(target_z_dim, hidden_dim),
#             nn.ReLU()
#         )

#         # 3. CNN: decap_map (1 channel)
#         self.decap_cnn = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # 4. CNN: context_map (多通道)
#         self.context_cnn = nn.Sequential(
#             nn.Conv2d(context_shape[0], 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # 5. 动态计算 Flatten 后维度
#         with torch.no_grad():
#             dummy_decap = torch.zeros(1, 1, *decap_shape)  # (B, 1, W, H)
#             decap_feat_dim = self.decap_cnn(dummy_decap).shape[1]

#             dummy_context = torch.zeros(1, *context_shape)  # (B, C, W, H)
#             context_feat_dim = self.context_cnn(dummy_context).shape[1]

#         # 6. 最终线性层
#         total_input_dim = hidden_dim * 2 + decap_feat_dim + context_feat_dim
#         self.linear = nn.Sequential(
#             nn.Linear(total_input_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU()
#         )

#         self._features_dim = hidden_dim  # 重要：供 SB3 使用

#     def forward(self, obs):
#         # obs 是一个字典
#         impedance = obs["impedance"]
#         target_z = obs["target_z"]
#         decap_map = obs["decap_map"].unsqueeze(1)  # (B, 1, H, W)
#         context_map = obs["context_map"]  # (B, C, H, W)

#         # 1. 各部分提取特征
#         imp_feat = self.impedance_fc(impedance)
#         targ_feat = self.target_z_fc(target_z)
#         decap_feat = self.decap_cnn(decap_map)
#         context_feat = self.context_cnn(context_map)

#         # 2. 拼接所有
#         concat = torch.cat([imp_feat, targ_feat, decap_feat, context_feat], dim=1)
#         return self.linear(concat)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_dim=256):
        # 临时设置占位值，稍后会正确赋值 _features_dim
        super().__init__(observation_space, features_dim=1)

        self.hidden_dim = hidden_dim

        # 获取各输入维度
        impedance_dim = observation_space["impedance"].shape[0]
        target_z_dim = observation_space["target_z"].shape[0]
        decap_shape = observation_space["decap_map"].shape  # (H, W)
        context_shape = observation_space["context_map"].shape  # (C, H, W)
        global_shape = observation_space["global_map"].shape    # (C_g, 64, 64)
        local_mask_shape = observation_space["local_mask"].shape  # (1, 64, 64)

        # --- 1. 全连接：impedance / target_z ---
        self.imp_fc = nn.Sequential(
            nn.Linear(impedance_dim, hidden_dim),
            nn.ReLU()
        )
        self.targ_fc = nn.Sequential(
            nn.Linear(target_z_dim, hidden_dim),
            nn.ReLU()
        )

        # --- 2. CNN 分支：decap_map (B, 1, H, W) ---
        self.decap_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # --- 3. CNN 分支：context_map (B, C, H, W) ---
        self.context_cnn = nn.Sequential(
            nn.Conv2d(context_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # --- 4. CNN 分支：global_map (B, C, 64, 64) ---
        self.global_cnn = nn.Sequential(
            nn.Conv2d(global_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # --- 5. CNN 分支：local_mask (B, 1, 64, 64) ---
        self.mask_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # --- 6. 计算 flatten 后维度 ---
        with torch.no_grad():
            dummy_decap = torch.zeros(1, 1, *decap_shape)
            dummy_context = torch.zeros(1, *context_shape)
            dummy_global = torch.zeros(1, *global_shape)
            dummy_mask = torch.zeros(1, *local_mask_shape)

            decap_feat_dim = self.decap_cnn(dummy_decap).shape[1]
            context_feat_dim = self.context_cnn(dummy_context).shape[1]
            global_feat_dim = self.global_cnn(dummy_global).shape[1]
            mask_feat_dim = self.mask_cnn(dummy_mask).shape[1]

        # --- 7. 整合所有特征 ---
        total_dim = hidden_dim * 2 + decap_feat_dim + context_feat_dim + global_feat_dim + mask_feat_dim
        self.linear = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        self._features_dim = hidden_dim  # 更新为 SB3 所需的特征维度

    def forward(self, obs):
        # 1. 全连接输入
        imp_feat = self.imp_fc(obs["impedance"])
        targ_feat = self.targ_fc(obs["target_z"])

        # 2. 卷积输入
        decap_feat = self.decap_cnn(obs["decap_map"].unsqueeze(1))      # (B, 1, H, W)
        context_feat = self.context_cnn(obs["context_map"])             # (B, C, H, W)
        global_feat = self.global_cnn(obs["global_map"])                # (B, C_g, 64, 64)
        mask_feat = self.mask_cnn(obs["local_mask"])                    # (B, 1, 64, 64)

        # 3. 拼接全部特征
        concat = torch.cat([imp_feat, targ_feat, decap_feat, context_feat, global_feat, mask_feat], dim=1)
        return self.linear(concat)


class CircuitEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self, ckt_file, result_file, render_mode=None):
        super(CircuitEnv, self).__init__()

        # ------------------ 1. 初始化环境参数 (与您原版相同) ------------------
        with open(ckt_file, "r") as f:
            self.design = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(result_file, "r") as f:
            self.initial_result = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.vdd = self.design["vdd"]
        self.target_impedance_value = 0.1 * self.vdd * self.vdd / self.design["chiplets"][0]["power"]
        
        self.base_ckt = build_ac_ckt(ckt_file, result_file)
        self.frequency_points = np.geomspace(0.1e9, 10e9, 100) # 100个频率点
        
        self.interposer_w = self.design["interposer"]["w"]
        self.interposer_h = self.design["interposer"]["h"]
        self.initial_candidate_branch = [
            f"cd_{x}_{y}" for x in range(self.interposer_w) for y in range(self.interposer_h)
            if (x, y) not in self.initial_result["tsvs"]
        ]
        self.num_initial_candidates = len(self.initial_candidate_branch)

        # ------------------ 2. 定义观测和动作空间 (与您原版相同) ------------------
        num_freq_points = len(self.frequency_points)
        # self.observation_space = spaces.Dict({
        #     "impedance": spaces.Box(low=0, high=1, shape=(num_freq_points,), dtype=np.float64),
        #     "target_z": spaces.Box(low=0, high=1, shape=(num_freq_points,), dtype=np.float64),
        #     "decap_map": spaces.Box(low=0, high=1, shape=(self.interposer_w, self.interposer_h), dtype=np.float64)
        # })

        self.observation_space = spaces.Dict({
            "impedance": spaces.Box(low=0, high=1, shape=(100,), dtype=np.float32),
            "target_z": spaces.Box(low=0, high=1, shape=(100,), dtype=np.float32),
            "decap_map": spaces.Box(low=0, high=1, shape=(self.interposer_h, self.interposer_w), dtype=np.float32),
            "context_map": spaces.Box(low=0, high=1, shape=(3, self.interposer_h, self.interposer_w), dtype=np.float32),
            "global_map": spaces.Box(low=0, high=1, shape=(3, 64, 64), dtype=np.float32),
            "local_mask": spaces.Box(low=0, high=1, shape=(1, 64, 64), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(self.num_initial_candidates)
        
        # 扩大阻抗值，便于归一化操作
        self.Z_MIN = 1e-4 
        self.Z_MAX = 0.3 # 或者可以稍微放宽一点，比如 0.1，以应对可能的异常值
        self.Z_RANGE = self.Z_MAX - self.Z_MIN

        # ------------------ 3. 准备仿真所需的静态参数 (与您原版相同) ------------------
        self._prepare_sim_tensors()
        self.render_mode = render_mode

        # ------------------ 4. 初始化奖励函数所需的状态变量 ------------------
        # 在 reset() 中会被正确赋值
        self.total_violation = 0.0

    # _prepare_sim_tensors, _get_impedances, _get_obs, action_masks, _get_info
    # 这些辅助函数与您的版本完全相同，这里为了简洁省略，直接使用您原有的即可
    # ... (此处省略您已有的 _prepare_sim_tensors, _get_impedances, _get_obs, action_masks, _get_info 函数)
    def _prepare_sim_tensors(self):
        observe_branch = ["id"]
        typ, u, v, val, index = self.base_ckt.prepare_sim("0", observe_branch + self.initial_candidate_branch)
        self.typ, self.u, self.v = typ, u, v
        self.g_index, self.r_index, self.l_index, self.v_index = [torch.tensor([], dtype=torch.long)] * 4
        self.c_index_map = index[len(observe_branch):]
        self.i_index = index[:len(observe_branch)]
        (self.all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
        self.g_value, self.r_value, self.l_value = [torch.tensor([], dtype=torch.float64)] * 3
        self.initial_c_base_value = val[self.c_index_map]
        self.all_exc_value = val[self.all_exc_index]
        self.sims = [AcSimulationCuDSS(typ, u, v, val, freq) for freq in self.frequency_points]

    def _get_impedances(self):
        zs = []
        c_value_tensor = torch.zeros_like(self.initial_c_base_value)
        if self.placed_capacitor_indices:
            placed_indices_tensor = torch.tensor(self.placed_capacitor_indices)
            c_value_tensor[placed_indices_tensor] = self.initial_c_base_value[placed_indices_tensor]
        c_index_tensor = self.c_index_map
        for freq, sim in zip(self.frequency_points, self.sims):
            sim.alter(c_index_tensor, c_value_tensor)
            sim.factorize()
            sim.solve()
            i_voltage = sim.branch_voltage(self.i_index)
            zs.append(i_voltage)
        return torch.cat(zs).abs().detach().numpy()

    # def _get_obs(self):
    #     impedances_np = self._get_impedances()
    #     decap_map = np.zeros((self.interposer_w, self.interposer_h), dtype=np.float64)
    #     if self.placed_capacitor_indices:
    #         for action_index in self.placed_capacitor_indices:
    #             info = self.initial_candidate_branch[action_index]
    #             match = re.search(r"cd_(\d+)_(\d+)", info)
    #             if match:
    #                 x, y = int(match.group(1)), int(match.group(2))
    #                 decap_map[x, y] = 1.0
    #     return {"impedance": impedances_np, "target_z": np.full_like(impedances_np, self.target_impedance_value), "decap_map": decap_map}
        
    #     # ## 归一化操作
    #     # # 1. 获取原始的、未归一化的阻抗值
    #     # impedances_np_raw = self._get_impedances()
        
    #     # # 2. 【新的归一化方法】 Min-Max Scaling
    #     # # 公式: (X - min) / (max - min)
    #     # impedances_scaled = (impedances_np_raw - self.Z_MIN) / self.Z_RANGE
        
    #     # # 对目标值也做同样的处理
    #     # target_z_raw = np.full_like(impedances_np_raw, self.target_impedance_value)
    #     # target_z_scaled = (target_z_raw - self.Z_MIN) / self.Z_RANGE

    #     # # 3. 裁剪以确保在 [0, 1] 范围内
    #     # # 这一步仍然非常重要，可以处理掉任何超出你预设范围的意外值
    #     # impedances_normalized = np.clip(impedances_scaled, 0, 1)
    #     # target_z_normalized = np.clip(target_z_scaled, 0, 1)

    #     # # ... (decap_map 部分不变) ...
    #     # decap_map = np.zeros((self.interposer_w, self.interposer_h), dtype=np.float64)
    #     # if self.placed_capacitor_indices:
    #     #     for action_index in self.placed_capacitor_indices:
    #     #         info = self.initial_candidate_branch[action_index]
    #     #         match = re.search(r"cd_(\d+)_(\d+)", info)
    #     #         if match:
    #     #             x, y = int(match.group(1)), int(match.group(2))
    #     #             decap_map[x, y] = 1.0
    #     # # 5. 返回归一化后的观测字典
    #     # return {
    #     #     "impedance": impedances_normalized, 
    #     #     "target_z": target_z_normalized, 
    #     #     "decap_map": decap_map
    #     # }

    def _get_obs(self):
        # 1. 获取阻抗（长度 = freq 点数）
        impedances_np = self._get_impedances()
        
        # 2. 构造 decap_map
        decap_map = np.zeros((self.interposer_h, self.interposer_w), dtype=np.float32)
        if self.placed_capacitor_indices:
            for action_index in self.placed_capacitor_indices:
                info = self.initial_candidate_branch[action_index]
                match = re.search(r"cd_(\d+)_(\d+)", info)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    decap_map[y, x] = 1.0

        # 3. 构造 context_map（形状 [3, W, H]）
        context_map = np.zeros((3, self.interposer_h, self.interposer_w), dtype=np.float32)
        for chiplet in self.design["chiplets"]:
            chip_power = chiplet["power"] / 3.0         # 归一化 power（最大 3）
            chip_freq = chiplet["freq"] / 10e9          # 归一化 freq（最大 10GHz）

            for x in range(chiplet["x"], chiplet["x"] + chiplet["w"]):
                for y in range(chiplet["y"], chiplet["y"] + chiplet["h"]):
                    if 0 <= x < self.interposer_w and 0 <= y < self.interposer_h:
                        context_map[0, y, x] = 1.0          # chiplet 标志
                        context_map[1, y, x] = chip_power
                        context_map[2, y, x] = chip_freq

        # 4. 构造目标阻抗向量
        target_z = np.full_like(impedances_np, self.target_impedance_value, dtype=np.float32)
        
        global_map = np.zeros((3, 64, 64), dtype=np.float32)
        local_mask = np.zeros((1, 64, 64), dtype=np.float32)

        # 将 decap_map / context_map 映射到 global_map 中（左上角对齐）
        h, w = self.interposer_h, self.interposer_w
        global_map[:, :h, :w] = context_map

        # 同理构建 local_mask（例如将当前芯粒区域 mask 掉）
        for chiplet in self.design["chiplets"]:
            for x in range(chiplet["x"], chiplet["x"] + chiplet["w"]):
                for y in range(chiplet["y"], chiplet["y"] + chiplet["h"]):
                    if x < 64 and y < 64:
                        local_mask[0, y, x] = 1.0

        # 5. 返回观察字典（注意全部为 float32）
        return {
            "impedance": impedances_np.astype(np.float32),
            "target_z": target_z,
            "decap_map": decap_map,
            "context_map": context_map,
            "global_map": global_map,
            "local_mask": local_mask, 
        }

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.num_initial_candidates, dtype=bool)
        if self.placed_capacitor_indices:
            mask[self.placed_capacitor_indices] = False
        return mask

    def _get_info(self):
        return {"action_mask": self.action_masks()}


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置状态
        self.placed_capacitor_indices = []
        self.current_step = 0
        
        # 获取初始观测
        observation = self._get_obs()
        
        # 【修改】计算初始的总违规量
        impedances = observation["impedance"]
        violation_array = np.maximum(impedances - self.target_impedance_value, 0)
        self.total_violation = np.sum(violation_array)
        
        info = self._get_info()
        
        if self.render_mode == "console":
            self.render(observation, 0) # 初始奖励为0

        return observation, info
    
    def step(self, action):
    # 1. 【核心修改】在执行动作前，记录旧的达标点数
        old_obs = self._get_obs()
        old_impedances = old_obs["impedance"]
        old_points_below_target = np.sum(old_impedances <= self.target_impedance_value)
        
        # 2. 执行动作
        self.placed_capacitor_indices.append(action)
        self.current_step += 1

        # 3. 获取新的观测值
        observation = self._get_obs()
        new_impedances = observation["impedance"]
        new_points_below_target = np.sum(new_impedances <= self.target_impedance_value)

        # 4. 【核心修改】计算新的奖励
        # 4.1 主要奖励：基于达标点数的增加量
        improvement_reward = (new_points_below_target - old_points_below_target)
        
        # 放大这个信号，使其比行动成本更重要
        # 例如，每多一个点达标，就奖励 5 分
        reward = improvement_reward * 5.0
        
        # 4.2 行动成本：每一步都给予一个小的负奖励
        action_cost = -2 # 这个值是超参数，可以调整
        reward += action_cost

        # 5. 判断回合是否结束
        # 5.1 成功终止 (Terminated): 所有频率点都达标
        terminated = bool(new_points_below_target == len(self.frequency_points))
        
        # 5.2 失败截断 (Truncated): 所有可用位置都已放置电容
        truncated = (len(self.placed_capacitor_indices) == self.num_initial_candidates)
        
        # 6. 【核心修改】为回合结束添加最终奖励或惩罚
        if terminated:
            final_reward = 100.0
            efficiency_bonus = (self.num_initial_candidates - len(self.placed_capacitor_indices)) * 0.5
            final_reward += efficiency_bonus
            
            reward += final_reward
            print(f"🎉🎉🎉 放置成功! 使用电容数量: {len(self.placed_capacitor_indices)}, 获得效率奖励: {efficiency_bonus:.2f}")
            # reward += 200.0
            # print(f"🎉🎉🎉 放置成功! 使用电容数量: {len(self.placed_capacitor_indices)}")

        elif truncated and not terminated:
            reward -= 100 # 用尽所有电容仍未成功，给予巨大的惩罚
            print(f"🔥🔥🔥 放置失败! 所有可用位置已用尽。")

        # 7. 获取信息（包含更新后的动作掩码）
        info = self._get_info()

        if self.render_mode == "console":
            self.render(observation, reward)
            
        return observation, reward, terminated, truncated, info


    # def step(self, action):
    #     # 1. 执行动作: 将选择的电容位置记录下来
    #     #    注意：我们不再需要检查 action 是否已被放置，因为 MaskablePPO 会处理这个问题。
    #     self.placed_capacitor_indices.append(action)
    #     self.current_step += 1

    #     # 2. 获取新的观测值
    #     observation = self._get_obs()
    #     new_impedances = observation["impedance"]

    #     # 3. 【核心修改】计算新的奖励
    #     # 3.1 计算新的总违规量
    #     new_violation_array = np.maximum(new_impedances - self.target_impedance_value, 0)
    #     new_total_violation = np.sum(new_violation_array)

    #     # 3.2 计算奖励 = 违规量的减少量 - 行动成本
    #     # 违规量减少得越多，这个值就越大，奖励也越高
    #     violation_reduction = self.total_violation - new_total_violation
        
    #     # 将减少量放大，使其成为一个更强的学习信号
    #     reward = violation_reduction * 10.0
        
    #     # 增加一个小的负奖励作为放置电容的“成本”，鼓励智能体使用更少的电容
    #     reward -= 0.5  # 这个值是一个超参数，可以调整

    #     # 3.3 更新环境状态，为下一步做准备
    #     self.total_violation = new_total_violation

    #     # 4. 判断回合是否结束
    #     # 4.1 成功终止 (Terminated): 当总违规量小于一个极小值时，认为任务成功
    #     terminated = bool(self.total_violation < 1e-6)

    #     # 4.2 失败截断 (Truncated): 当所有可用位置都已放置电容
    #     truncated = (len(self.placed_capacitor_indices) == self.num_initial_candidates)

    #     # 5. 【核心修改】为回合结束添加最终奖励或惩罚
    #     if terminated:
    #         reward += 200  # 成功完成任务，给予巨大的额外奖励
    #         print(f"🎉🎉🎉 放置成功! 使用电容数量: {len(self.placed_capacitor_indices)}")
            
    #     elif truncated and not terminated:
    #         reward -= 100 # 用尽所有电容仍未成功，给予巨大的惩罚
    #         print(f"🔥🔥🔥 放置失败! 所有可用位置已用尽。")

    #     # 6. 获取信息（包含更新后的动作掩码）
    #     info = self._get_info()

    #     if self.render_mode == "console":
    #         self.render(observation, reward)
            
    #     return observation, reward, terminated, truncated, info

    def render(self, observation, reward, mode='console'):
        # (与您原版相同)
        if mode == 'console':
            impedances_part = observation["impedance"]
            peak_impedance = np.max(impedances_part)
            points_below_target = np.sum(impedances_part <= self.target_impedance_value)
            self.total_violation = np.sum(np.maximum(impedances_part - self.target_impedance_value, 0))


            print(
                f"Step: {self.current_step}, "
                f"Placed Caps: {len(self.placed_capacitor_indices)}, "
                f"Reward: {reward:+.4f}, "
                f"Total Violation: {self.total_violation:.4f}, "
                f"Peak Impedance: {peak_impedance:.4f}, "
                f"Points Below Target: {points_below_target}/{len(self.frequency_points)}"
            )
            