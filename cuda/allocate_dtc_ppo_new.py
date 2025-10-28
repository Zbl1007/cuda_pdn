# gemini 2.5 pro 生成
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import os
import sys  

# 将上级目录加入系统路径，以便找到自定义模块
# 请确保此路径是正确的
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from AcSimulation import AcSimulationCuDSS

# 假设这些模块在当前路径或已安装
from AcAdjoint import AcAdjointFunction
from Circuit import Circuit, BranchType
from build_ckt import build_ac_ckt

# 如果Pardiso仿真器有无害的警告，则抑制它们
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# ======================================================
# ========== 新增：模式开关和模型路径 ==========
# ======================================================
# 设置为 True 进行训练并保存模型；设置为 False 加载模型并直接推理
TRAIN_MODE = True 


# --- 配置参数 ---
CASE_NAME = "case1"
# --- !!重要!! ---
# --- 请确保您的数据文件路径是正确的 ---
CONFIG_FILE = f"../../data/{CASE_NAME}.yaml"
RESULT_TSV_FILE = f"../../data/{CASE_NAME}_result_tsv.yaml"
OUTPUT_DTC_FILE = f"    ../../data/{CASE_NAME}_result_dtc_ppo_simplified.yaml"
OUTPUT_IMP_PLOT = f"./{CASE_NAME}_pdn_impedance_comparison.png" # 增加图片输出路径
OUTPUT_IMP_CSV = f"./{CASE_NAME}_final_pdn_impedance.csv"      # 增加CSV输出路径

MODEL_PATH = f"./result/{CASE_NAME}_ppo_dtc_agent.pth" # 模型保存/加载的路径

# --- 超参数 ---
LEARNING_RATE = 0.0001
GAMMA = 0.99                # 奖励折扣因子
GAE_LAMBDA = 0.99      # GAE平滑参数 
PPO_CLIP_EPSILON = 0.2      # PPO裁剪参数
TRAIN_EPOCHS = 200          # 总训练轮次
NUM_ENVS_PER_EPOCH = 8      # 每轮次用于探索的环境数量
K_EPOCHS_PER_UPDATE = 5     # 每次更新PPO时的优化周期数
ENTROPY_COEFF = 0.01        # 熵奖励系数，鼓励探索
VALUE_LOSS_COEFF = 0.5      # 价值损失系数

# --- 环境设置 (简化版) ---
class PDNEnvironmentSimplified:
    """
    将PDN仿真封装成一个强化学习环境。
    这个简化版只关注DTC的位置，不考虑类型或2D布局。
    """
    def __init__(self, config_file, result_tsv_file):
        print("正在初始化简化版PDN环境...")
        with open(config_file, "r") as f:
            design = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(result_tsv_file, "r") as f:
            result = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.interposer_w = design["interposer"]["w"]
        self.interposer_h = design["interposer"]["h"]
        self.vdd = design["vdd"]
        power = design["chiplets"][0]["power"]
        # if power > 0:
        self.target_impedance = 0.1 * self.vdd * self.vdd / power
        # else:
        #     self.target_impedance = 0.01 # 默认目标阻抗
            
        self.ckt = build_ac_ckt(config_file, result_tsv_file)
        self.frequency_points = np.geomspace(0.1e9, 10e9, 100)
        
        # 定义DTC的候选位置坐标
        self.candidate_dtc_coords = [
            (x, y) for x in range(self.interposer_w) for y in range(self.interposer_h)
            if (x, y) not in result["tsvs"]
        ]
        self.num_available_locations = len(self.candidate_dtc_coords)
        
        # 准备仿真所需的张量
        self._prepare_simulation_tensors()
        self.reset()
        print(f"环境初始化完成。目标阻抗: {self.target_impedance:.4f} Ohm。可用DTC位置数: {self.num_available_locations}")

    def _prepare_simulation_tensors(self):
        # 此设置基于用户提供的原始代码
        observe_branch = ["id"]
        candidate_branch = [f"cd_{x}_{y}" for x,y in self.candidate_dtc_coords]
        
        typ, u, v, val, index = self.ckt.prepare_sim("0", observe_branch + candidate_branch)
        
        self.g_index = torch.tensor([], dtype=torch.long)
        self.r_index = torch.tensor([], dtype=torch.long)
        self.c_index = index[len(observe_branch):]
        self.l_index = torch.tensor([], dtype=torch.long)
        self.xc_index = torch.tensor([], dtype=torch.long)
        self.xl_index = torch.tensor([], dtype=torch.long)
        self.i_index = index[:len(observe_branch)]
        self.v_index = torch.tensor([], dtype=torch.long)
        (self.all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))

        self.g_value = torch.tensor([], dtype=torch.float64)
        self.r_value = torch.tensor([], dtype=torch.float64)
        self.base_c_value = val[self.c_index]
        self.l_value = torch.tensor([], dtype=torch.float64)
        self.xc_value = torch.tensor([], dtype=torch.float64)
        self.xl_value = torch.tensor([], dtype=torch.float64)
        self.all_exc_value = val[self.all_exc_index]
        
        
        self.sims = [AcSimulationCuDSS(typ, u, v, val, freq) for freq in self.frequency_points]

    def reset(self):
        """为新回合重置环境。"""
        self.placed_decaps_indices = set() # 使用集合存储已放置DTC的位置索引
        self.current_impedance = self._get_impedance()
        return self._get_state()

    def _get_impedance(self, placement_indices = None):
        """为当前DTC配置计算阻抗曲线。"""
        c_value = torch.zeros_like(self.base_c_value)
        # 将已选择位置的电容值设为其基础值
        if placement_indices:
            indices_to_update = list(placement_indices)
            c_value[indices_to_update] = self.base_c_value[indices_to_update]
        elif self.placed_decaps_indices:
            indices_to_update = list(self.placed_decaps_indices)
            c_value[indices_to_update] = self.base_c_value[indices_to_update]
        
        zs = []
        with torch.no_grad():
            for freq, sim in zip(self.frequency_points, self.sims):
                i_voltage, _ = AcAdjointFunction.apply(
                    self.g_index, self.r_index, self.c_index, self.l_index,
                    self.xc_index, self.xl_index, self.i_index, self.v_index,
                    self.all_exc_index, self.g_value, self.r_value, c_value,
                    self.l_value, self.xc_value, self.xl_value, self.all_exc_value,
                    freq, sim
                )
                zs.append(i_voltage.abs())
        return torch.tensor(zs, dtype=torch.float32)
    
    def _get_state(self):
        """
        构建简化的1D状态向量。
        状态 = [DTC放置情况(二进制向量), 当前阻抗曲线, 目标阻抗曲线]
        """
        # 1. DTC放置情况的二进制向量
        placement_vector = torch.zeros(self.num_available_locations)
        if self.placed_decaps_indices:
            indices = list(self.placed_decaps_indices)
            placement_vector[indices] = 1.0
        
        # 2. 当前和目标阻抗曲线
        impedance_state = self.current_impedance
        target_impedance_state = torch.full_like(self.current_impedance, self.target_impedance)
        
        # 3. 拼接成最终的状态向量
        state = torch.cat([placement_vector, impedance_state, target_impedance_state])
        return state

    def step(self, action_loc_idx):
        """
        在环境中执行一步。
        action_loc_idx: 代表选择放置DTC的位置索引的单个整数。
        """
        # 检查无效动作（在已占用的位置上重复放置）
        if action_loc_idx in self.placed_decaps_indices:
            # 施加重罚并终止回合
            return self._get_state(), -10.0, True

        # 获取动作前满足目标的频点数
        n_t = torch.sum(self.current_impedance <= self.target_impedance).item()

        # 应用动作：将新位置加入已放置集合
        self.placed_decaps_indices.add(action_loc_idx)
        
        # 获取新阻抗和满足的频点数
        self.current_impedance = self._get_impedance()
        n_t1 = torch.sum(self.current_impedance <= self.target_impedance).item()
        
        # 计算奖励
        done = False
        terminal_bonus = 0
        total_freq_points = len(self.frequency_points)
        num_placed = len(self.placed_decaps_indices)

        if n_t1 == total_freq_points:
            # 目标达成！根据使用的DTC数量给予奖励
            terminal_bonus = 1.0 + (self.num_available_locations - num_placed)
            done = True
        elif num_placed == self.num_available_locations:
            # 板子已满，但目标未达成，施加惩罚
            violation_ratio = torch.max(self.current_impedance / self.target_impedance - 1.0).item()
            terminal_bonus = -max(0, violation_ratio)
            done = True
            
        reward = (n_t1 - n_t) / total_freq_points + terminal_bonus

        return self._get_state(), reward, done

# --- PPO 代理实现 ---

class ActorCritic(nn.Module):
    """简化的Actor-Critic网络，使用MLP。"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor网络，输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, action_dim), nn.Softmax(dim=-1)
        )
        # Critic网络，输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

    def act(self, state, available_actions_mask):
        """训练时使用：带探索（采样）地选择动作"""
        action_probs = self.actor(state)
        # 屏蔽掉已经选择过的动作，避免无效探索
        action_probs = action_probs * available_actions_mask
        if torch.sum(action_probs) > 0: action_probs = action_probs / action_probs.sum()
        else: action_probs = available_actions_mask / available_actions_mask.sum()
        
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def act_greedy(self, state, available_actions_mask):
        """推理时使用：选择概率最高的动作（贪婪）"""
        action_probs = self.actor(state)
        # 屏蔽掉不可用动作
        action_probs[available_actions_mask == 0] = -1e8 
        action = torch.argmax(action_probs)
        return action.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

class RolloutBuffer:
    """用于存储轨迹（经验）的缓冲区。"""
    def __init__(self):
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = [], [], [], [], []

    def clear(self):
        del self.actions[:]; del self.states[:]; del self.logprobs[:]; del self.rewards[:]; del self.is_terminals[:]

    def __len__(self): return len(self.states)

class PPO:
    """PPO代理，包含Actor-Critic网络和更新逻辑。"""
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        # self.memory = RolloutBuffer()

    def select_action(self, state, available_actions_mask, memory):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.act(state, available_actions_mask)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.item()
    
    def select_greedy_action(self, state, available_actions_mask):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = self.policy_old.act_greedy(state, available_actions_mask)
        return action.item()

    def update(self, memory_A, memory_B):
        # 论文核心创新：合并近期经验(A)和历史最佳经验(B)进行训练 
        combined_rewards = memory_A.rewards + memory_B.rewards
        combined_terminals = memory_A.is_terminals + memory_B.is_terminals
        # --- 使用GAE(广义优势估计)计算优势，与论文对齐  ---
        advantages = []
        last_advantage = 0
        # 需要从后往前计算
        for i in reversed(range(len(combined_rewards))):
            if combined_terminals[i]:
                # 如果是终局，V(s_t+1)为0
                state_value = self.policy.critic(memory_A.states[i] if i < len(memory_A) else memory_B.states[i-len(memory_A)])
                delta = combined_rewards[i] - state_value
                last_advantage = delta
            else:
                state_value = self.policy.critic(memory_A.states[i] if i < len(memory_A) else memory_B.states[i-len(memory_A)])
                next_state_value = self.policy.critic(memory_A.states[i+1] if i+1 < len(memory_A) else memory_B.states[i+1-len(memory_A)])
                delta = combined_rewards[i] + GAMMA * next_state_value - state_value
                last_advantage = delta + GAMMA * GAE_LAMBDA * last_advantage
            advantages.insert(0, last_advantage)
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 构造训练数据
        old_states = torch.squeeze(torch.stack(memory_A.states + memory_B.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory_A.actions + memory_B.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory_A.logprobs + memory_B.logprobs, dim=0)).detach()
        
        # 蒙特卡洛回报估计
        rewards, discounted_reward = [], 0
        for reward, is_terminal in zip(reversed(combined_rewards), reversed(combined_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        

        # 对策略进行K个周期的优化
        for _ in range(K_EPOCHS_PER_UPDATE):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - PPO_CLIP_EPSILON, 1 + PPO_CLIP_EPSILON) * advantages
            loss = -torch.min(surr1, surr2) + VALUE_LOSS_COEFF * self.MseLoss(state_values, rewards) - ENTROPY_COEFF * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

# --- 主训练循环 ---
if __name__ == '__main__':
    env = PDNEnvironmentSimplified(CONFIG_FILE, RESULT_TSV_FILE)
    state_dim = env._get_state().shape[0]
    action_dim = env.num_available_locations
    ppo_agent = PPO(state_dim, action_dim)

    if TRAIN_MODE:

        # 日志记录变量
        start_time = time.time()
        best_solution_decap_count = env.num_available_locations + 1
        best_solution_placements = set()

        # 创建论文中描述的双经验池A和B 
        memory_A = RolloutBuffer() # 存储近期经验
        memory_B = RolloutBuffer() # 存储历史最佳经验

        print("\n--- 开始训练阶段 ---")
        start_time = time.time()
        best_solution_decap_count = env.num_available_locations + 1

        # 训练循环
        for epoch in range(1, TRAIN_EPOCHS + 1):
            # 清空近期经验池A，准备收集新数据 
            memory_A.clear()
            # 在一轮中，并行运行多个环境以收集数据
            for _ in range(NUM_ENVS_PER_EPOCH):
                state = env.reset()
                available_actions_mask = torch.ones(action_dim) # 1表示可用，0表示不可用

                # 临时存储当前回合的经验
                current_episode_memory = RolloutBuffer()
                
                # 一个回合最多放置所有可用数量的DTC
                for t in range(env.num_available_locations):
                    action = ppo_agent.select_action(state, available_actions_mask, current_episode_memory)
                    
                    # 更新掩码，将已选动作设为不可用
                    available_actions_mask[action] = 0
                    
                    state, reward, done = env.step(action)
                    current_episode_memory.rewards.append(reward)
                    current_episode_memory.is_terminals.append(done)

                    # ppo_agent.memory.rewards.append(reward)
                    # ppo_agent.memory.is_terminals.append(done)
                    
                    if done:
                        # 如果找到了一个有效解
                        if reward > 0 and len(env.placed_decaps_indices) < best_solution_decap_count:
                            best_solution_decap_count = len(env.placed_decaps_indices)
                            # 清空精英池B，并存入新的最佳经验 
                            memory_B.clear()
                            memory_B.states = list(current_episode_memory.states)
                            memory_B.actions = list(current_episode_memory.actions)
                            memory_B.logprobs = list(current_episode_memory.logprobs)
                            memory_B.rewards = list(current_episode_memory.rewards)
                            memory_B.is_terminals = list(current_episode_memory.is_terminals)
                            print(f"--- 新发现更优解！DTC数量: {best_solution_decap_count} ---")
                        break
                
                # 将当前回合的经验存入近期经验池A
                memory_A.states.extend(current_episode_memory.states)
                memory_A.actions.extend(current_episode_memory.actions)
                memory_A.logprobs.extend(current_episode_memory.logprobs)
                memory_A.rewards.extend(current_episode_memory.rewards)
                memory_A.is_terminals.extend(current_episode_memory.is_terminals)
            
            # 使用收集到的数据更新PPO代理
            # 使用A和B池中的数据更新PPO代理 
            if len(memory_A) > 0:
                ppo_agent.update(memory_A, memory_B)

            # if epoch % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"轮次: {epoch}/{TRAIN_EPOCHS}, 最佳解DTC数量: {best_solution_decap_count}, 已用时: {elapsed_time:.2f}s")

        print("\n--- 训练完成 ---")

        print(f"训练阶段找到的最优DTC数量: {best_solution_decap_count}")
        print(f"训练阶段布局（位置索引）: {best_solution_placements}")

        # 训练结束后，保存模型
        print(f"--- 正在保存已训练的模型至 {MODEL_PATH} ---")
        torch.save(ppo_agent.policy.state_dict(), MODEL_PATH)
        print("--- 模型保存成功 ---")

    else:
        # ======================================================
        # ========== 加载模型阶段 ==========
        # ======================================================
        print(f"\n--- 当前模式: 推理 ---")
        print(f"--- 正在从 {MODEL_PATH} 加载预训练模型 ---")
        try:
            ppo_agent.policy.load_state_dict(torch.load(MODEL_PATH))
            ppo_agent.policy_old.load_state_dict(torch.load(MODEL_PATH))
            print("--- 模型加载成功 ---")
        except FileNotFoundError:
            print(f"错误: 找不到模型文件 {MODEL_PATH}。请先将 TRAIN_MODE 设为 True 进行训练。")
            sys.exit(1)

    # ======================================================
    # ========== 推理/使用阶段 ==========
    # (无论训练或加载，最终都会执行这一步)
    # ======================================================
    print("\n--- 开始使用模型进行最终推理 ---")
    ppo_agent.policy.eval() # 切换到评估模式
    state = env.reset()
    available_actions_mask = torch.ones(action_dim)
    done = False
    
    # 使用训练好的智能体，通过贪婪策略来找到最终解
    while not done and len(env.placed_decaps_indices) < env.num_available_locations:
        action = ppo_agent.select_greedy_action(state, available_actions_mask)
        available_actions_mask[action] = 0
        state, reward, done = env.step(action)

    final_placements = env.placed_decaps_indices
    final_decap_count = len(final_placements)

    print("\n--- 推理/使用阶段完成 ---")

    if done and reward > 0:
        print(f"找到的最优DTC数量: {final_decap_count}")
        print(f"布局（位置索引）: {final_placements}")
        
        # 将最终结果保存到yaml文件
        result_dtc_data = {"dtcs": []}
        for loc_idx in final_placements:
            x, y = env.candidate_dtc_coords[loc_idx]
            result_dtc_data["dtcs"].append((int(x), int(y))) # 确保是标准int

        with open(OUTPUT_DTC_FILE, "w") as f:
            yaml.dump(result_dtc_data, f)
        print(f"\n已将最佳DTC布局保存至 {OUTPUT_DTC_FILE}")

        # ======================================================
        # ========== 新增的数据可视化部分 ==========
        # ======================================================
        print("\n正在生成结果对比图...")
        
        # 1. 计算基准阻抗（仅TSV，无DTC）
        impedance_baseline = env._get_impedance(set())
        
        # 2. 计算优化后的最终阻抗
        impedance_optimized = env._get_impedance(final_placements)
        
        # 3. 绘图
        plt.figure(figsize=(10, 6))
        plt.loglog(env.frequency_points, impedance_baseline.numpy(), label='仅有 TSV (基准)', color='blue')
        plt.loglog(env.frequency_points, impedance_optimized.numpy(), label=f'TSV + {best_solution_decap_count}个优化后DTC', color='green')
        plt.axhline(y=env.target_impedance, color='red', linestyle='--', label=f'目标阻抗 ({env.target_impedance:.4f} Ohm)')
        
        # 4. 美化图表
        plt.title('PDN阻抗优化前后对比')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('阻抗 (Ohm)')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        
        # 5. 保存图像
        plt.savefig(OUTPUT_IMP_PLOT, dpi=300)
        print(f"结果对比图已保存至 '{OUTPUT_IMP_PLOT}'")
        plt.show() # 如果希望在运行时直接显示图像，可以取消此行注释

    else:
        print("在训练中未能找到满足条件的解。")