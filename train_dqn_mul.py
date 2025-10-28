# # # train_dqn.py

# # import os
# # import torch
# # import torch.multiprocessing as mp
# # from stable_baselines3.common.env_util import make_vec_env
# # from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# # # 导入你的环境（假设在 new_ENV.py 中）
# # from CircuitEnvDQN import CircuitEnv, CustomCombinedExtractorForDQN, MaskableDQN
# # # 【重要】从PPO代码中复制过来的你的环境代码中，可能仍然有一些
# # # 依赖于MaskablePPO的特性，例如 `action_masks()`。
# # # 标准的DQN不会使用这个，但保留它也没问题。

# # if __name__ == '__main__':
# #     # 设置多进程启动方法，以保证跨平台兼容性
# #     try:
# #         mp.set_start_method('spawn', force=True)
# #     except RuntimeError:
# #         pass

# #     # --- 训练参数 ---
# #     case = "case1"
# #     CKT_FILE = f'data/{case}.yaml'
# #     RESULT_FILE = f'data/{case}_result_tsv.yaml'
# #     N_ENVS = 1  # 对于DQN，可以从较少的并行环境开始，甚至从1开始
# #     TOTAL_TIMESTEPS = 10_0000
# #     MODEL_SAVE_PATH = f"dqn_circuit_model_{case}"
# #     TENSORBOARD_LOG_PATH = f"./dqn_circuit_tensorboard_{case}/"
    
# #     # 【修复建议】在你的 CircuitEnv 中实现观测归一化
# #     # 在此之前，请确保你已经在 CircuitEnv 的 _get_obs 方法中
# #     # 实现了对 "impedance" 和 "target_z" 的归一化。
# #     # 这是成功训练的关键。

# #     # --- 创建并行环境 ---
# #     # 建议先用 DummyVecEnv (n_envs=1) 跑通，再换成 SubprocVecEnv
# #     print(f"正在创建 {N_ENVS} 个并行环境...")
# #     env = make_vec_env(
# #         CircuitEnv, 
# #         n_envs=N_ENVS, 
# #         vec_env_cls=DummyVecEnv,
# #         env_kwargs=dict(
# #             ckt_file=CKT_FILE, 
# #             result_file=RESULT_FILE, 
# #             render_mode='console'
# #         )
# #     )
# #     print("并行环境创建完成！")

# #     # --- 定义DQN模型和策略网络参数 ---
# #     policy_kwargs = dict(
# #         features_extractor_class=CustomCombinedExtractorForDQN,
# #         features_extractor_kwargs=dict(hidden_dim=256),
# #     )

# #     # --- 创建并配置 DQN 模型 ---
# #     # DQN的超参数与PPO非常不同，需要仔细调整
# #     model = MaskableDQN(
# #         "MultiInputPolicy",  # 适用于Dict观测空间
# #         env,
# #         policy_kwargs=policy_kwargs,
# #         learning_rate=5e-5,          # DQN的学习率通常比PPO小
# #         buffer_size=200_000,         # 经验回放池大小
# #         learning_starts=50_000,      # 探索这么多步后才开始学习
# #         batch_size=64,               # 每次从池中采样的批次大小
# #         tau=1.0,                     # 使用硬更新
# #         gamma=0.99,
# #         train_freq=(4, "step"),      # 每4个环境步骤训练一次
# #         gradient_steps=1,
# #         target_update_interval=10_000, # 每10000步更新一次目标网络
# #         exploration_fraction=0.3,    # 用30%的总步数完成探索率衰减
# #         exploration_initial_eps=1.0, # 初始探索率
# #         exploration_final_eps=0.05,  # 最终探索率
# #         verbose=1,
# #         tensorboard_log=TENSORBOARD_LOG_PATH,
# #         device='cuda'
# #     )

# #     # --- 开始训练 ---
# #     print("--- Starting DQN Training ---")
# #     print(f"Model Architecture:\n{model.q_net}")
# #     model.learn(total_timesteps=TOTAL_TIMESTEPS)#, progress_bar=True)

# #     # --- 保存模型 ---
# #     model.save(MODEL_SAVE_PATH)
# #     print(f"DQN model saved to {MODEL_SAVE_PATH}")

# #     # 关闭环境
# #     env.close()



# # train_dqn_from_paper.py
# # train_dqn_from_paper.py (完整版)

# import os
# import torch
# import torch.multiprocessing as mp
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# # 导入我们新设计的、严格遵循论文思想的环境
# from new_EnvForPaper import CircuitEnvPaper, CustomCnnExtractor

# if __name__ == '__main__':
#     # 设置多进程启动方法，以保证跨平台兼容性
#     try:
#         mp.set_start_method('spawn', force=True)
#     except RuntimeError:
#         pass

#     # --- 训练参数 ---
#     case = "case1"
#     CKT_FILE = f'data/{case}.yaml'
#     RESULT_FILE = f'data/{case}_result_tsv.yaml'
    
#     # 建议从单环境开始，以确保逻辑正确性
#     N_ENVS = 1
    
#     # 总训练步数
#     TOTAL_TIMESTEPS = 500_000 # 可以根据需要调整
    
#     MODEL_SAVE_PATH = f"dqn_paper_{case}"
#     TENSORBOARD_LOG_PATH = f"./dqn_paper_tensorboard_{case}/"
    
#     # --- 创建环境 ---
#     # 使用 DummyVecEnv 进行调试和单线程训练
#     print(f"正在创建 {N_ENVS} 个单线程环境...")
#     env = make_vec_env(
#         CircuitEnvPaper, 
#         n_envs=N_ENVS, 
#         vec_env_cls=DummyVecEnv,
#         env_kwargs=dict(
#             ckt_file=CKT_FILE, 
#             result_file=RESULT_FILE, 
#             render_mode='console' # 在训练时通常不渲染以加快速度
#         )
#     )
#     print("环境创建完成！")
    
#     policy_kwargs = dict(
#         features_extractor_class=CustomCnnExtractor,
#         features_extractor_kwargs=dict(features_dim=256), # CNN最终输出的特征维度
#     )

#     # --- 创建并配置 DQN 模型 ---
#     # 使用你提供的、经过验证的超参数
#     model = DQN(
#         "CnnPolicy",                 # <-- 使用标准的CNN策略，因为它能很好地处理图像输入
#         env,
#         policy_kwargs=policy_kwargs, # <-- 在这里传入我们的自定义CNN
#         learning_rate=5e-5,          # DQN的学习率通常比PPO小
#         buffer_size=200_000,         # 经验回放池大小
#         learning_starts=50_000,      # 探索这么多步后才开始学习
#         batch_size=64,               # 每次从池中采样的批次大小
#         tau=1.0,                     # 使用硬更新 (tau=1.0)
#         gamma=0.99,                  # 折扣因子
#         train_freq=(4, "step"),      # 每4个环境步骤训练一次
#         gradient_steps=1,            # 每次训练执行一步梯度下降
#         target_update_interval=10_000, # 每10000步更新一次目标网络
#         exploration_fraction=0.3,    # 用30%的总步数完成探索率衰减
#         exploration_initial_eps=1.0, # 初始探索率，从完全随机开始
#         exploration_final_eps=0.05,  # 最终探索率，保留5%的随机性
#         verbose=1,                   # 打印训练信息
#         tensorboard_log=TENSORBOARD_LOG_PATH,
#         device='cuda'                 # 在GPU上训练
#     )

#     # --- 开始训练 ---
#     print("--- Starting DQN Training (Paper's Method) ---")
#     print(f"Model Architecture:\n{model.q_net}") # 打印出Q网络结构，确认是CNN
    
#     # 开始学习，progress_bar=True 可以显示一个进度条
#     model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

#     # --- 保存模型 ---
#     model.save(MODEL_SAVE_PATH)
#     print(f"DQN model saved to {MODEL_SAVE_PATH}")

#     # 关闭环境
#     env.close()



# train_parallel_dqn.py

import os
import ray
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from new_EnvForPaper import CircuitEnvPaper
from my_dqn import ValueDQN,ValueDQNMul # 假设你的buffer也在这个文件里
from my_dqn import ReplayBuffer # 需要单独导入ReplayBuffer

# ==================== Ray Actor 定义 ====================

@ray.remote
class GlobalReplayBuffer:
    """ 一个可以被所有进程共享的经验回放池 Actor """
    def __init__(self, capacity):
        self.buffer = ReplayBuffer(capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def size(self):
        return len(self.buffer)

@ray.remote(num_cpus=1) # 每个Worker分配一个CPU核心
class RolloutWorker:
    """ 负责与环境交互并收集数据的 Actor """
    def __init__(self, ckt_file, result_file, dqn_params, rank):
        self.env = CircuitEnvPaper(ckt_file, result_file)
        # Worker也需要一个agent来选择动作，但它不训练，也没有自己的buffer
        worker_params = dqn_params.copy()
        worker_params['buffer_capacity'] = 1 
        self.agent = ValueDQNMul(self.env, **worker_params)
        self.agent.online_net.eval() # Worker的网络只用于推理
        self.rank = rank
        
    def set_weights(self, weights):
        self.agent.set_weights(weights)

    def rollout(self, global_buffer_handle):
        """ 运行一个完整的episode，并将数据推送到全局buffer """
        state, _ = self.env.reset(seed=int(time.time()) + self.rank) # 保证随机性
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # 异步地将数据推送到全局buffer
            global_buffer_handle.push.remote(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
        return episode_reward, episode_length

# ==================== 主训练逻辑 ====================

if __name__ == '__main__':
    # --- 1. 参数配置 ---
    ray.init() # num_cpus=... 可以指定核心数

    case = "case1"
    run_name = f"MyDQN_Parallel_{case}"
    SEED = 12
    CKT_FILE = f'data/{case}.yaml'
    RESULT_FILE = f'data/{case}_result_tsv.yaml'
    N_WORKERS = 4 # 并行收集数据的Worker数量
    
    # ... (其他文件路径和超参数配置，与你的原文件类似) ...
    TOTAL_TIMESTEPS = 300_000
    dqn_params = {
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.001,
        'epsilon_decay': 0.9999, # 衰减可以慢一些，因为步数更多
        'learning_rate': 0.0001,
        'buffer_capacity': 50_000, # buffer可以适当调小，加快新经验的学习
        'batch_size': 64,
        'target_update_freq': 1000
    }
    
    # --- 2. 初始化 Actors 和 Learner ---
    print("--- Initializing Actors and Learner ---")
    
    # 创建全局共享的 Replay Buffer
    global_buffer = GlobalReplayBuffer.remote(dqn_params['buffer_capacity'])
    
    # 创建 Learner (在主进程中)
    # Learner需要一个环境来获取obs/action space，但它不运行env.step
    learner_env = CircuitEnvPaper(CKT_FILE, RESULT_FILE)
    learner_agent = ValueDQNMul(learner_env, **dqn_params)
    learner_env.close()

    # 创建 N 个 Rollout Workers
    workers = [RolloutWorker.remote(CKT_FILE, RESULT_FILE, dqn_params, i) for i in range(N_WORKERS)]
    
    writer = SummaryWriter(log_dir=f"tensorboard_logs/{run_name}/")

    # --- 3. 训练循环 (异步) ---
    print("\n--- Starting Asynchronous Training ---")

    # 首次同步模型权重
    weights = ray.put(learner_agent.get_weights())
    for w in workers:
        w.set_weights.remote(weights)

    # 启动第一批 rollout 任务
    rollout_futures = [w.rollout.remote(global_buffer) for w in workers]
    
    global_step = 0
    with tqdm(total=TOTAL_TIMESTEPS, desc="Training Timesteps") as pbar:
        while global_step < TOTAL_TIMESTEPS:
            # 1. Learner 进行模型更新
            # 检查buffer大小是否足够
            buffer_size = ray.get(global_buffer.size.remote())
            if buffer_size > dqn_params['batch_size']:
                # 从全局buffer异步采样
                batch_future = global_buffer.sample.remote(dqn_params['batch_size'])
                batch = ray.get(batch_future)
                
                # Learner 更新网络
                loss = learner_agent.update(batch)
                
                if loss is not None:
                    writer.add_scalar('Train/Step_Loss', loss, global_step)
            
            # 2. 检查是否有Worker完成了rollout
            ready_futures, remaining_futures = ray.wait(rollout_futures, num_returns=1, timeout=0)
            
            if ready_futures:
                # 处理完成的任务
                for future in ready_futures:
                    episode_reward, episode_length = ray.get(future)
                    
                    # 记录数据
                    writer.add_scalar('Rollout/Episode_Reward', episode_reward, global_step)
                    writer.add_scalar('Rollout/Episode_Length', episode_length, global_step)
                    
                    # 更新总步数和进度条
                    global_step += episode_length
                    pbar.update(episode_length)
                    pbar.set_postfix({"eps": f"{learner_agent.epsilon:.3f}"})
                
                # 重新启动完成了的Worker
                new_futures = [w.rollout.remote(global_buffer) for w, f in zip(workers, rollout_futures) if f in ready_futures]
                rollout_futures = remaining_futures + new_futures
            
            # 3. 定期更新Worker的模型权重
            if global_step > 0 and global_step % 1000 == 0: # 例如每1000步
                weights = ray.put(learner_agent.get_weights())
                for w in workers:
                    w.set_weights.remote(weights)

    # --- 4. 训练结束 ---
    print("\n--- Training Finished ---")
    torch.save(learner_agent.online_net.state_dict(), f"models/{run_name}.pth")
    print("Model saved.")
    writer.close()
    ray.shutdown()