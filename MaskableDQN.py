import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

# 1. 改写Policy，支持传入动作掩码并屏蔽无效动作
class MaskableDQNPolicy(DQNPolicy):
    def forward(self, obs, action_mask=None, deterministic=False):
        q_values = self.q_net(obs)

        if action_mask is not None:
            # 屏蔽无效动作，置为很小的值，防止选中
            inf_mask = (~action_mask).bool()
            q_values = q_values.masked_fill(inf_mask, float('-1e8'))

        if deterministic:
            actions = q_values.argmax(dim=1)
        else:
            actions = q_values.argmax(dim=1)

        return actions, q_values

    def _predict(self, observation, deterministic=False):
        action_mask = None
        if isinstance(observation, dict) and "action_mask" in observation:
            action_mask = observation["action_mask"]
            # 去掉掩码，只保留环境obs输入
            observation = {k: v for k, v in observation.items() if k != "action_mask"}
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask).to(self.device)
        actions, _ = self.forward(observation, action_mask=action_mask, deterministic=deterministic)
        return actions.cpu().numpy()

# 2. 支持动作掩码的ReplayBuffer
class MaskableReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_masks = None

    def add(self, obs, next_obs, action, reward, done, infos):
        # 初始化动作掩码数组
        if self.action_masks is None:
            self.action_masks = np.zeros((self.buffer_size,) + obs["action_mask"].shape, dtype=np.bool_)

        self.action_masks[self.pos] = obs["action_mask"]
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        data = super().sample(batch_size, env)

        data.action_mask = torch.as_tensor(self.action_masks[data.indices]).to(data.actions.device)
        return data

# 3. MaskableDQN主类
class MaskableDQN(DQN):
    def _setup_model(self):
        super()._setup_model()
        self.replay_buffer = MaskableReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            optimize_memory_usage=self.optimize_memory_usage,
            handle_timeout_termination=self.handle_timeout_termination,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """
        采样时带上动作掩码传给策略网络。
        """
        # 这部分大部分代码参考stable-baselines3源码，做必要修改传掩码
        # 为简化示意，核心改动是把obs与action_mask一并传递给policy

        # 详细实现请参考 stable-baselines3 版本源码，这里做简化

        # (1) 初始化变量
        # 例如：
        #  obs = env.reset()
        #  for step in range(n_rollout_steps):
        #      action, _ = self.policy.predict(obs, deterministic=False)
        #      new_obs, reward, done, info = env.step(action)
        #      rollout_buffer.add(obs, action, reward, new_obs, done)
        #      obs = new_obs

        # 这里重点在于 obs 里带 action_mask，且传递给 policy.predict()

        # 以下伪代码示意：
        obs = env.reset()
        rollout_buffer.reset()
        for step in range(n_rollout_steps):
            # policy.predict 支持 obs 中含 action_mask
            actions, _states = self.policy.forward(obs)
            new_obs, rewards, dones, infos = env.step(actions)
            # 采集数据加入buffer
            rollout_buffer.add(obs, actions, rewards, new_obs, dones)

            obs = new_obs

        return True  # 采样成功

    def train(self, gradient_steps, batch_size=100):
        """
        训练时从 replay_buffer 采样，同时使用动作掩码屏蔽无效动作的Q值。
        """
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # 得到动作掩码
            action_mask = replay_data.action_mask

            # Q网络前向
            q_values = self.policy.q_net(replay_data.observations)

            # 屏蔽无效动作
            inf_mask = (~action_mask).bool()
            q_values = q_values.masked_fill(inf_mask, float('-1e8'))

            # 计算当前动作的Q值
            action_q_values = q_values.gather(1, replay_data.actions.unsqueeze(-1)).squeeze(-1)

            # 计算目标Q值等，后续不变
            with torch.no_grad():
                next_q_values = self.policy.q_net(replay_data.next_observations)
                next_q_values = next_q_values.masked_fill((~replay_data.next_action_mask).bool(), float('-1e8'))
                next_q_values_max, _ = next_q_values.max(dim=1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones.float()) * self.gamma * next_q_values_max

            loss = nn.functional.mse_loss(action_q_values, target_q_values)

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self._n_updates += 1
