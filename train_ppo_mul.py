import os
import random
import torch
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env
from PPO import CircuitEnv, CustomCombinedExtractor # 确保从您的文件中导入
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # 导入两种并行环境


# ==================== 全局种子设置函数 (保持不变) ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def make_env(rank, seed=0, ckt_file=None, result_file=None):
    """
    为多进程环境创建实例的辅助函数。
    
    :param rank: 每个进程的唯一ID
    :param seed: 随机数种子
    """
    def _init():
        env = CircuitEnv(ckt_file=ckt_file, result_file=result_file, render_mode=None)
        # 关键：为每个子进程设置不同的种子，以增加探索的多样性
        env.reset(seed=seed + rank) 
        return env
    set_seed(seed)
    return _init


if __name__ == '__main__':
    # ==================== 训练参数设置 ====================
    SEED = 12
    case = "multigpu"
    CKT_FILE = f'data/{case}.yaml'
    RESULT_FILE = f'data/{case}_result_tsv.yaml'
    # 【核心修改1】设置要并行运行的环境数量
    # 这个值通常设置为你电脑的CPU核心数，或者稍小一些。
    # 可以从 4, 8, 16 开始尝试。
    N_ENVS = 4
    # 【核心修改1】设置要加载的模型路径和新的保存路径
    # 这个路径必须指向你之前训练并保存的模型文件
    MODEL_LOAD_PATH = f"models/ppo_circuit_model_parallel_continued_{case}_107.zip" 
    # 为了不覆盖之前的模型，我们为继续训练后的模型设置一个新的保存路径
    MODEL_SAVE_PATH_CONTINUED = f"models/ppo_circuit_model_parallel_continued_{case}_107"

    # 【核心修改2】设置新的总训练步数
    # 假设之前训练了 200,000 步，现在我们想再训练 300,000 步，总共达到 500,000 步
    # 所以这里设置的是你希望训练结束时达到的总步数
    TOTAL_TIMESTEPS = 100000 # 可以适当增加总步数，因为训练更快了
    # MODEL_SAVE_PATH = "ppo_circuit_model_parallel"

    # 设置种子
    set_seed(SEED)

    # ==================== 【核心修改2】创建并行化的向量环境 ====================
    # 不再直接创建 env = CircuitEnv(...)
    # 而是使用 make_vec_env
    # 它会自动为你创建 N_ENVS 个 CircuitEnv 实例，并在独立的进程中运行它们。
    # env = make_vec_env(
    #     CircuitEnv,              # 要创建的环境类
    #     n_envs=N_ENVS,           # 并行环境的数量
    #     seed=SEED,
    #     vec_env_cls=SubprocVecEnv,
    #     env_kwargs={             # 传递给 CircuitEnv.__init__ 的参数
    #         'ckt_file': CKT_FILE,
    #         'result_file': RESULT_FILE,
    #         'render_mode': None  # 在并行训练时，通常不进行渲染
    #     }
    # )
    # env_fns = [make_env(i, SEED, CKT_FILE, RESULT_FILE) for i in range(N_ENVS)]
    # env = SubprocVecEnv(env_fns)
    print(f"正在创建 {N_ENVS} 个并行环境...")
    env = make_vec_env(
        CircuitEnv, 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv, # 使用多进程
        env_kwargs=dict(
            ckt_file=CKT_FILE, 
            result_file=RESULT_FILE, 
            render_mode=None  # 在并行训练时通常不渲染
        )
    )
    print("并行环境创建完成！")
    # ==================== 模型和策略定义 (基本不变) ====================
    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"hidden_dim": 256},
    }

    # 【注意】当使用并行环境时，一些超参数可能需要调整以保持训练稳定性。
    # PPO 的总批次大小 = n_envs * n_steps。
    # 为了保持与之前相似的更新频率和数据量，我们可以保持这个总批次大小大致不变。
    # 比如，之前是 1 * 2048 = 2048。现在可以是 8 * 256 = 2048。
    # 或者，我们可以让每个环境都收集更多数据，以利用并行优势，比如 8 * 1024。
    # 我们先保持总批次大小不变，这是一个稳妥的起点。
    # ==================== 【核心修改3】加载模型而不是创建新模型 ====================
    print(f"Loading model from {MODEL_LOAD_PATH}...")
    model = MaskablePPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0001,  # 0.00005
        n_steps=256,         # 每个环境跑256步 (8 * 256 = 2048 总批次)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.99,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        seed=SEED,
        policy_kwargs=policy_kwargs, 
        tensorboard_log=f"./tensorboard_logs/2025_10_07_ppo_{case}/" # 使用新日志目录
    )
    # model = MaskablePPO.load(
    #     MODEL_LOAD_PATH, 
    #     env, 
    #     verbose=1,
    #     # learning_rate=0.00005,  # 0.00005
    #     # n_steps=256,         # 每个环境跑256步 (8 * 256 = 2048 总批次)
    #     # batch_size=64,
    #     # n_epochs=10,
    #     # gamma=0.99,
    #     # gae_lambda=0.99,
    #     # clip_range=0.2,
    #     # vf_coef=0.5,
    #     # ent_coef=0.01,
    #     seed=SEED,
    #     policy_kwargs=policy_kwargs, 
    #     # --- 开始微调：降低学习率和熵，促进收敛 ---
    #     learning_rate=1e-5,  # 从 1e-4 降到 1e-5
    #     ent_coef=0.005,      # 从 0.01 降到 0.001
    #     # clip_range=0.1,       # 也可以尝试减小 clip_range
    #     tensorboard_log=f"./tensorboard_logs/2025_8_12_ppo_{case}/" # 使用新日志目录
    # )
    # print("Model loaded successfully. Continuing training...")


    # ==================== 开始训练 ====================
    print(f"Starting training with {N_ENVS} parallel environments...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True,reset_num_timesteps=False)

    # ==================== 保存模型 ====================
    model.save(MODEL_SAVE_PATH_CONTINUED)
    print(f"Model saved to {MODEL_SAVE_PATH_CONTINUED}")

    # 关闭环境
    env.close()