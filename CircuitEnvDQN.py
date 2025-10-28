import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
import re
import yaml
# 假设您已经有了这些辅助函数和类
from AcAdjoint import AcAdjointFunction
from AcSimulation import AcSimulationCuDSS, AcSimulationPardiso
from Circuit import Circuit, BranchType
from build_ckt import build_ac_ckt



class CustomCombinedExtractorForDQN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, hidden_dim: int = 256):
        
        # --- 步骤 1: 预计算特征维度 (在调用 super 之前) ---
        # 我们可以先定义网络结构，但不把它赋给 self
        
        canvas_size = 256
        
        # 先临时创建一个 CNN 来计算输出维度
        temp_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_canvas = torch.zeros(1, 1, canvas_size, canvas_size)
            cnn_output_dim = temp_cnn(dummy_canvas).shape[1]
        
        # 计算总的特征维度
        features_dim = cnn_output_dim + hidden_dim + hidden_dim
        
        # --- 步骤 2: 【核心修正】立即调用父类的 __init__ 方法 ---
        # 现在我们已经有了 features_dim，可以安全地调用它了
        super().__init__(observation_space, features_dim)

        # --- 步骤 3: 现在可以安全地将模块赋值给 self 了 ---
        self.canvas_size = canvas_size
        
        # 将我们之前临时创建的 cnn 赋给 self.cnn
        # 或者重新创建一遍，效果一样
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        impedance_dim = observation_space["impedance"].shape[0]
        target_z_dim = observation_space["target_z"].shape[0]

        self.impedance_fc = nn.Linear(impedance_dim, hidden_dim)
        self.target_z_fc = nn.Linear(target_z_dim, hidden_dim)
        
        print("--- CustomCombinedExtractorForDQN Initialized ---")
        # ... (其余打印信息)
        
    def forward(self, observations: dict) -> torch.Tensor:
        # forward 方法保持不变
        # ...
        # 提取各个部分的观测
        impedance = observations["impedance"]
        target_z = observations["target_z"]
        decap_map = observations["decap_map"] # 原始尺寸的地图, e.g., (N, 20, 15)

        # --- 核心逻辑：将不同尺寸的 decap_map 绘制到固定大小的画布上 ---
        batch_size = decap_map.shape[0]
        device = decap_map.device
        
        canvas = torch.zeros(batch_size, 1, self.canvas_size, self.canvas_size, device=device)
        map_h, map_w = decap_map.shape[1], decap_map.shape[2]
        
        # 将原始地图内容复制到画布的左上角
        canvas[:, 0, :map_h, :map_w] = decap_map
        
        # --- 特征提取 ---
        canvas_features = self.cnn(canvas)
        impedance_features = F.relu(self.impedance_fc(impedance))
        target_z_features = F.relu(self.target_z_fc(target_z))
        
        # --- 拼接所有特征 ---
        concatenated_features = torch.cat([canvas_features, impedance_features, target_z_features], dim=1)
        
        return concatenated_features

# maskable_dqn.py


from typing import Any, Dict, Optional, Tuple
from stable_baselines3 import DQN
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported


class MaskableDQN(DQN):
    """
    一个支持动作掩码的、完整的自定义DQN算法。
    它借鉴了 sb3-contrib 中 MaskablePPO 的思想，
    同时在 predict (决策) 和 train (学习) 阶段应用掩码。
    """

    def __init__(self, *args, **kwargs):
        # 在创建时，确保 replay_buffer_kwargs 包含 handle_timeout_termination=True
        # 这会隐式地让 ReplayBuffer 开始存储 info 字典
        if "replay_buffer_kwargs" not in kwargs:
            kwargs["replay_buffer_kwargs"] = {}
        kwargs["replay_buffer_kwargs"]["handle_timeout_termination"] = True
        
        super().__init__(*args, **kwargs)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        在选择动作前应用动作掩码。
        (沿用我们之前稳定、简洁的版本)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            # 探索：从有效动作中随机选择
            action_masks = get_action_masks(self.env)
            actions = []
            for mask in action_masks:
                valid_actions = np.where(mask)[0]
                if len(valid_actions) > 0:
                    actions.append(np.random.choice(valid_actions))
                else:
                    actions.append(self.action_space.sample()) # Fallback
            return np.array(actions), state
        else:
            # 利用：从有效动作中选择Q值最高的
            obs_tensor, _ = self.policy.obs_to_tensor(observation)
            with th.no_grad():
                q_values = self.policy.q_net(obs_tensor)

            action_masks = get_action_masks(self.env)
            mask_tensor = th.tensor(action_masks, device=self.device, dtype=th.float32)
            masked_q_values = q_values + (1.0 - mask_tensor) * -1e9
            
            action = th.argmax(masked_q_values, dim=1).cpu().numpy()
            return action, state

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # 切换到训练模式
        self.policy.set_training_mode(True)
        # 更新学习率
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # 从经验回放池中采样
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # --- 【核心修改】计算目标Q值时应用掩码 ---
                
                # 1. 使用在线网络为 Double-DQN 选择下一个动作
                next_q_values_online = self.policy.q_net(replay_data.next_observations)
                
                # 2. 从 info 字典中获取 next_observations 对应的动作掩码
                # 我们需要一个'兜底'的掩码，以防某些样本没有这个信息
                default_mask = np.ones((1, self.action_space.n), dtype=np.float32)
                
                next_action_masks = np.array([
                    info.get("action_mask", default_mask) for info in replay_data.infos
                ]).squeeze() # .squeeze() to remove extra dimensions if any
                
                next_mask_tensor = th.tensor(next_action_masks, device=self.device, dtype=th.float32)

                # 3. 屏蔽无效动作的Q值
                masked_next_q_values = next_q_values_online + (1.0 - next_mask_tensor) * -1e9

                # 4. 从被屏蔽的Q值中选择最优动作的索引
                next_actions_indices = th.argmax(masked_next_q_values, dim=1).unsqueeze(-1)

                # 5. 使用目标网络获取这些最优动作的Q值
                next_q_values_target = self.policy.q_net_target(replay_data.next_observations)
                target_q_from_next_state = next_q_values_target.gather(1, next_actions_indices)

                # 6. 计算最终的目标Q值
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q_from_next_state

            # 获取当前状态-动作对的Q值
            current_q_values = self.policy.q_net(replay_data.observations)
            current_q = current_q_values.gather(1, replay_data.actions)

            # 计算损失
            loss = F.smooth_l1_loss(current_q, target_q)
            losses.append(loss.item())

            # 优化步骤
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # 更新目标网络
        self._n_updates += gradient_steps
        if self._n_updates % self.target_update_interval == 0:
            self.policy.q_net_target.load_state_dict(self.policy.q_net.state_dict())

        # 记录日志
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
class CircuitEnv(gym.Env):
    metadata = {'render_modes': ['console']}

    def __init__(self, ckt_file, result_file, render_mode=None):
        super(CircuitEnv, self).__init__()

        # ------------------ 1. 初始化环境参数 (与您原版相同) ------------------
        with open(ckt_file, "r") as f:
            design = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(result_file, "r") as f:
            self.initial_result = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.vdd = design["vdd"]
        self.target_impedance_value = 0.1 * self.vdd * self.vdd / design["chiplets"][0]["power"]
        
        self.base_ckt = build_ac_ckt(ckt_file, result_file)
        self.frequency_points = np.geomspace(0.1e9, 10e9, 100) # 100个频率点
        
        self.interposer_w = design["interposer"]["w"]
        self.interposer_h = design["interposer"]["h"]
        self.initial_candidate_branch = [
            f"cd_{x}_{y}" for x in range(self.interposer_w) for y in range(self.interposer_h)
            if (x, y) not in self.initial_result["tsvs"]
        ]
        self.num_initial_candidates = len(self.initial_candidate_branch)

        # ------------------ 2. 定义观测和动作空间 (与您原版相同) ------------------
        num_freq_points = len(self.frequency_points)
        self.observation_space = spaces.Dict({
            "impedance": spaces.Box(low=0, high=1, shape=(num_freq_points,), dtype=np.float64),
            "target_z": spaces.Box(low=0, high=1, shape=(num_freq_points,), dtype=np.float64),
            "decap_map": spaces.Box(low=0, high=1, shape=(self.interposer_w, self.interposer_h), dtype=np.float64)
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
        self.sims = [AcSimulationPardiso(typ, u, v, val, freq) for freq in self.frequency_points]

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

    def _get_obs(self):
        impedances_np = self._get_impedances()
        decap_map = np.zeros((self.interposer_w, self.interposer_h), dtype=np.float64)
        if self.placed_capacitor_indices:
            for action_index in self.placed_capacitor_indices:
                info = self.initial_candidate_branch[action_index]
                match = re.search(r"cd_(\d+)_(\d+)", info)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    decap_map[x, y] = 1.0
        return {"impedance": impedances_np, "target_z": np.full_like(impedances_np, self.target_impedance_value), "decap_map": decap_map}
        
        # ## 归一化操作
        # # 1. 获取原始的、未归一化的阻抗值
        # impedances_np_raw = self._get_impedances()
        
        # # 2. 【新的归一化方法】 Min-Max Scaling
        # # 公式: (X - min) / (max - min)
        # impedances_scaled = (impedances_np_raw - self.Z_MIN) / self.Z_RANGE
        
        # # 对目标值也做同样的处理
        # target_z_raw = np.full_like(impedances_np_raw, self.target_impedance_value)
        # target_z_scaled = (target_z_raw - self.Z_MIN) / self.Z_RANGE

        # # 3. 裁剪以确保在 [0, 1] 范围内
        # # 这一步仍然非常重要，可以处理掉任何超出你预设范围的意外值
        # impedances_normalized = np.clip(impedances_scaled, 0, 1)
        # target_z_normalized = np.clip(target_z_scaled, 0, 1)

        # # ... (decap_map 部分不变) ...
        # decap_map = np.zeros((self.interposer_w, self.interposer_h), dtype=np.float64)
        # if self.placed_capacitor_indices:
        #     for action_index in self.placed_capacitor_indices:
        #         info = self.initial_candidate_branch[action_index]
        #         match = re.search(r"cd_(\d+)_(\d+)", info)
        #         if match:
        #             x, y = int(match.group(1)), int(match.group(2))
        #             decap_map[x, y] = 1.0
        # # 5. 返回归一化后的观测字典
        # return {
        #     "impedance": impedances_normalized, 
        #     "target_z": target_z_normalized, 
        #     "decap_map": decap_map
        # }

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
        if action in self.placed_capacitor_indices:
            print(f"🔥🔥🔥 放置失败! 电容位置占用。")
            return self._get_obs(), 0, True, True, self._get_info()
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
        final_info = {}
        for i, done in enumerate(np.atleast_1d(terminated or truncated)):
            if done:
                # 当回合结束时，"final_info" 中应该包含这个回合的信息
                final_info[i] = info

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
                # f"Placed Caps: {len(self.placed_capacitor_indices)}, "
                f"Placed Caps: {self.placed_capacitor_indices}, "
                f"Reward: {reward:+.4f}, "
                f"Total Violation: {self.total_violation:.4f}, "
                f"Peak Impedance: {peak_impedance:.4f}, "
                f"Points Below Target: {points_below_target}/{len(self.frequency_points)}"
            )
 