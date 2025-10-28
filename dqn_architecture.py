# dqn_architecture.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.policies import BasePolicy

# -------------------------------------------------------------------
# 1. 特征提取器 (保持不变)
# -------------------------------------------------------------------
class CustomCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.as_tensor(np.zeros(observation_space.shape, dtype=np.float32)).unsqueeze(0)
            cnn_output_dim = self.cnn(dummy_input).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# -------------------------------------------------------------------
# 2. 自定义的、能处理掩码的Q网络 (核心)
# -------------------------------------------------------------------
class MaskableQNetwork(QNetwork):
    """
    一个特殊的Q网络，它在其forward方法中接收并应用动作掩码。
    """
    def forward(self, features: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        """
        这个 forward 方法与标准的 QNetwork 不同，它额外接收一个 action_masks 参数。
        """
        # self.q_net 是在父类 QNetwork 中定义的标准MLP
        q_values_raw = self.q_net(features)
        
        # 应用掩码：将无效动作的Q值设置为负无穷
        masked_q_values = torch.where(
            action_masks.bool(),
            q_values_raw,
            torch.tensor(-float('inf'), device=q_values_raw.device)
        )
        
        return masked_q_values

# -------------------------------------------------------------------
# 3. 自定义的DQN策略 (将所有东西连接在一起)
# -------------------------------------------------------------------
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
# 注意：我们要继承自 DQNPolicy，而不是 ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy

class MaskableDQNPolicy(DQNPolicy):
    """
    一个自定义的DQN策略，它使用我们的 CustomCnnExtractor 和 MaskableQNetwork。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 这里的 make_features_extractor 会由父类自动调用
        # 我们需要重写 make_q_net
    
    def make_q_net(self) -> MaskableQNetwork:
        # 重写此方法，以确保创建的是我们自定义的 MaskableQNetwork
        # net_args 包含了 SB3 自动处理好的输入和输出维度等信息
        net_args = self._update_features_extractor(self.net_args, features_extractor=self.features_extractor)
        return MaskableQNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # DQN 的 policy forward 方法不用于决策，而是用于兼容 On-Policy 算法的接口
        # 真正的决策逻辑在 predict 方法中，而 Q 值计算在 q_net.forward 中
        # 我们需要重写 predict
        pass # 实际上我们不需要重写 forward

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        # 重写 predict 方法，以确保在调用 q_net 时传入掩码
        
        # 1. 提取状态特征 (父类已经帮我们做好了)
        # features = self.extract_features(obs)
        
        # 啊哈！这里的 `obs` 是一个已经处理过的 Tensor，不包含原始的 dict
        # 这意味着我们无法在这里轻易拿到掩码。
        
        # 让我们再次改变策略！有一个更聪明的 SB3 特性可以利用！
        # 如果特征提取器返回一个元组，这个元组会被解包并传递给后续网络！
        pass # 放弃这个复杂的方案，回到一个更简单的特征提取器方案