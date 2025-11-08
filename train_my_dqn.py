# # train_my_dqn.py
# from new_EnvForPaper import CircuitEnvPaper
# from my_dqn import ValueDQN

# # --- 参数 ---
# CKT_FILE = 'data/case1.yaml'
# RESULT_FILE = 'data/case1_result_tsv.yaml'
# TOTAL_EPISODES = 20000

# # --- 初始化 ---
# env = CircuitEnvPaper(ckt_file=CKT_FILE, result_file=RESULT_FILE)
# agent = ValueDQN(env)

# # --- 训练循环 ---
# for episode in range(TOTAL_EPISODES):
#     state, _ = env.reset()
#     episode_reward = 0
#     done = False
    
#     while not done:
#         action = agent.choose_action(state)
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
        
#         agent.replay_buffer.push(state, action, reward, next_state, done)
#         loss = agent.update()
        
#         state = next_state
#         episode_reward += reward
        
#     print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {loss or 0:.4f}")



# train_my_dqn.py (集成 TensorBoard 和 Render 的最终版)

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm # 引入进度条，让体验更好

# 导入我们自己实现的环境和DQN算法
from new_EnvForPaper import CircuitEnvPaper
from my_dqn import ValueDQN


if __name__ == '__main__':
    # --- 1. 参数配置 ---
    case = "multigpu"
    run_name = f"2025_8_25_MyDQN_ValueBased_{case}"
    
    CKT_FILE = f'data/{case}.yaml'
    RESULT_FILE = f'data/2025_11/{case}_result_tsv_tsvoneshot_50.yaml'
    MODEL_SAVE_PATH = f"models/{run_name}.pth"
    TENSORBOARD_LOG_PATH = f"tensorboard_logs/{run_name}"
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)
    
    # 【核心修改】训练现在基于总步数，而不是总回合数
    TOTAL_TIMESTEPS = 300_000
    LOG_INTERVAL = 1000 # 每1000步记录一次回合的平均数据

    # new630 
    # dqn_params = {
    #     'gamma': 0.5,
    #     'epsilon': 1.0,
    #     'epsilon_min': 0.001,
    #     'epsilon_decay': 0.9999, # 衰减可以慢一些，因为步数更多
    #     'learning_rate': 0.001,
    #     'buffer_capacity': 1000, # buffer可以适当调小，加快新经验的学习
    #     'batch_size': 200,
    #     'target_update_freq': 100
    # }
    # 630
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

    # --- 2. 初始化 ---
    print("--- Initializing ---")
    env = CircuitEnvPaper(
        ckt_file=CKT_FILE, 
        result_file=RESULT_FILE, 
        # render_mode='console' # 设为None可以大幅加速，只在需要时开启
    )
    agent = ValueDQN(env, **dqn_params)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_PATH)
    
    print(f"Device: {agent.device}")
    print(f"Agent Network:\n{agent.online_net}")

    # --- 3. 训练循环 (基于总步数) ---
    print("\n--- Starting Training ---")
    
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_rewards = []
    episode_lengths = []
    
    # 【核心修改】tqdm 的 total 设置为 TOTAL_TIMESTEPS
    with tqdm(total=TOTAL_TIMESTEPS, desc="Training Timesteps") as pbar:
        for global_step in range(1, TOTAL_TIMESTEPS + 1):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            
            state = next_state
            episode_reward += reward
            episode_length += 1

            # 【核心修改】每一步都更新进度条
            pbar.update(1)

            # 在 TensorBoard 中记录每一步的损失
            if loss is not None:
                writer.add_scalar('Train/Step_Loss', loss, global_step)

            # 当一个回合结束时
            if done:
                # 记录这个回合的数据
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # 重置回合统计数据
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0

            # 【核心修改】定期记录平均回合数据到 TensorBoard 和进度条
            if global_step % LOG_INTERVAL == 0:
                if len(episode_rewards) > 0:
                    mean_reward = np.mean(episode_rewards)
                    mean_length = np.mean(episode_lengths)
                    
                    writer.add_scalar('Rollout/Mean_Episode_Reward', mean_reward, global_step)
                    writer.add_scalar('Rollout/Mean_Episode_Length', mean_length, global_step)
                    
                    # 更新 tqdm 进度条的后缀信息
                    pbar.set_postfix({
                        "reward": f"{mean_reward:.2f}",
                        "len": f"{mean_length:.0f}",
                        "eps": f"{agent.epsilon:.3f}"
                    })
                    
                    # 清空列表，为下一个记录周期做准备
                    episode_rewards = []
                    episode_lengths = []
                
                # 记录探索率
                writer.add_scalar('Params/Epsilon', agent.epsilon, global_step)

    # --- 4. 训练结束 ---
    print("\n--- Training Finished ---")
    torch.save(agent.online_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    writer.close()
    env.close()