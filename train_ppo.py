from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks # 关键辅助函数
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv # 导入两种并行环境

from CircuitENV import CircuitEnv, CustomCombinedExtractor
import random
import torch
import numpy as np

# ==================== 第一步：定义一个全局的种子设置函数 ====================
def set_seed(seed):
    """
    为所有相关的库设置随机数种子，以确保可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果您在GPU上训练，还需要设置CUDA的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
        # 下面这两行是为了确保CUDA的计算结果是确定性的，可能会稍微降低性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

if __name__ == '__main__':
    # 设置随机数种子
    SEED = 12
    set_seed(SEED)
    # 1. 实例化环境
    ckt_file = "data/case1.yaml"
    result_file = "data/case1_result_tsv.yaml"
    env = CircuitEnv(ckt_file=ckt_file, result_file=result_file, render_mode="console")
    
    
    # 定义 policy_kwargs 来指定使用哪个特征提取器，并配置其超参数
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            hidden_dim=256 # 您可以在这里自由调整网络的“宽度”，例如 128, 256, 512
        )
    )

    # 2. 检查环境 (可选但强烈推荐)
    # check_env(env)

    # num_cpu = 4
    # print(f"正在创建 {num_cpu} 个并行环境...")
    # env = make_vec_env(
    #     CircuitEnv, 
    #     n_envs=num_cpu, 
    #     vec_env_cls=SubprocVecEnv, # 使用多进程
    #     env_kwargs=dict(
    #         ckt_file=ckt_file, 
    #         result_file=result_file, 
    #         render_mode=None  # 在并行训练时通常不渲染
    #     )
    # )
    # print("并行环境创建完成！")


    # 3. 实例化 PPO 模型
    # PPO 会自动检测并使用环境 info 字典中的 "action_mask"
    model = MaskablePPO(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.00005,  # 学习率保持不变，与论文相似
        n_steps=2048,          # <--- 核心修改：与PPO标准和论文实践对齐
        batch_size=64,         # <--- 使用PPO常用值
        n_epochs=10,           # <--- 使用PPO常用值
        gamma=0.99,            # <--- 标准折扣因子
        gae_lambda=0.99,       # <--- 标准GAE参数
        clip_range=0.2,        # <--- 标准PPO裁剪范围
        vf_coef=0.5,           # c1，价值函数损失的权重，默认通常是0.5
        ent_coef=0.01,         # c2，熵项的权重，默认通常是0.01或更小
        seed=SEED,
        tensorboard_log="./ppo_circuit_tensorboard/"
    )

    # 4. 训练模型
    print("开始训练 PPO 模型...")
    # 训练步数可以根据问题复杂度调整
    model.learn(total_timesteps=500000) 
    print("训练完成！")
    # --- 新增：保存模型 ---
    model_path = "ppo_circuit_model1"
    model.save(model_path)
    print(f"模型已成功保存到 {model_path}.zip")

    # 5. 测试训练好的模型
    print("\n--- 开始测试训练好的模型 ---")
    obs, info = env.reset()
    for i in range(env.num_initial_candidates):
        # predict 函数会自动使用动作掩码
        action, _states = model.predict(obs, deterministic=True, action_masks=info["action_mask"])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("回合结束！")
            final_placements = [env.initial_candidate_branch[idx] for idx in env.placed_capacitor_indices]
            print(f"最终放置的电容位置: {final_placements}")
            print(f"共使用了 {len(final_placements)} 个电容。")
            break
    env.close()