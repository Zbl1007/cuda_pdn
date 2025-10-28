import torch
import numpy as np

# 导入你的环境和DQN智能体类
from new_EnvForPaper import CircuitEnvPaper
from my_dqn import ValueDQN
import yaml
import re
case = 'multigpu'
RESULT_FILE_1 = f'dqn_{case}_result_tsv_output.yaml'


def evaluate_agent(model_path, env, num_episodes=10):
    """
    加载并评估一个训练好的DQN智能体。

    参数:
    - model_path (str): 已保存模型文件 (.pth) 的路径。
    - env (gym.Env): 要在其中进行评估的环境实例。
    - num_episodes (int): 要运行的评估回合数。
    """
    # 1. 初始化智能体
    #    注意：这里的DQN参数除了epsilon外，其他都无关紧要，因为我们不进行训练。
    #    但是为了能成功实例化agent，我们还是需要提供一个参数字典。
    eval_dqn_params = {
        'gamma': 0.99,
        'epsilon': 0.0,  # 【关键】在评估时，将epsilon设为0，进行纯粹的贪心选择（利用）。
        'epsilon_min': 0.0,
        'epsilon_decay': 1.0, # 不衰减
        'learning_rate': 0.0, # 不学习
        'buffer_capacity': 1,
        'batch_size': 1,
        'target_update_freq': 999999
    }
    agent = ValueDQN(env, **eval_dqn_params)
    
    # 2. 加载模型权重
    print(f"--- Loading model from: {model_path} ---")
    # map_location='cpu' 确保即使模型是在GPU上训练的，也能在没有GPU的机器上加载。
    agent.online_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    
    # 3. 将网络设置为评估模式
    #    这会关闭诸如Dropout或BatchNorm之类的层，对于评估很重要。
    agent.online_net.eval()
    
    print(f"\n--- Starting Evaluation for {num_episodes} episodes ---")
    
    all_episode_rewards = []
    all_episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 【关键】使用 agent.choose_action()，由于epsilon为0，它将总是选择最优动作。
            action = agent.choose_action(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        all_episode_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # 4. 打印最终的平均统计结果
    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)
    mean_length = np.mean(all_episode_lengths)
    std_length = np.std(all_episode_lengths)
    
    print("\n--- Evaluation Finished ---")
    print(f"Average Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Length: {mean_length:.2f} +/- {std_length:.2f}")
    # 你还可以获取最终的电容布局方案
    final_placements = eval_env.placed_capacitors
    print(final_placements)
    final_placements_info = [eval_env.initial_candidate_branch[i] for i in final_placements]
    print(f"\n最终的电容布局方案 (动作索引): {final_placements}")
    print(f"最终的电容布局方案 (支路名称): {final_placements_info}")

    with open(RESULT_FILE, "r") as f:
        result_dtc = yaml.load(f.read(), Loader=yaml.FullLoader)
    pattern = R"dtc_(\d+)_(\d+)"
    result_dtc["dtcs"] = []
    for tsv in final_placements_info:
        match_result = re.match(R"cd_(\d+)_(\d+)", tsv)
        if match_result:
            x = int(match_result.group(1))
            y = int(match_result.group(2))
            result_dtc["dtcs"].append((x, y))
    with open(RESULT_FILE_1, "w") as f:
        yaml.dump(result_dtc, f)

if __name__ == '__main__':
    # --- 配置 ---
    case = "multigpu"
    run_name = f"2025_8_25_MyDQN_ValueBased_{case}"
    
    CKT_FILE = f'data/{case}.yaml'
    RESULT_FILE = f'data/{case}_result_tsv.yaml'
    # 【重要】确保这个路径指向你想要评估的那个模型文件
    MODEL_PATH = f"models/{run_name}.pth" 
    
    NUM_EVAL_EPISODES = 1 # 评估20个回合以获得可靠的平均性能

    # --- 初始化环境 ---
    # 【重要】在评估时，开启 render_mode='console' 可以让你看到每一步的详细输出！
    eval_env = CircuitEnvPaper(
        ckt_file=CKT_FILE,
        result_file=RESULT_FILE,
        render_mode='console' 
    )

    # --- 运行评估 ---
    evaluate_agent(MODEL_PATH, eval_env, num_episodes=NUM_EVAL_EPISODES)
    
    eval_env.close()