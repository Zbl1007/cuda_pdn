# 文件名: train_dqn.py
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from CircuitEnvDQN import CircuitEnvDQN # <--- 导入新的环境类

# 1. 实例化为DQN设计的环境
ckt_file = "data/case1.yaml"
result_file = "data/case1_result_tsv.yaml"
env = CircuitEnvDQN(ckt_file=ckt_file, result_file=result_file, render_mode="console")

# 检查环境
check_env(env)

# 2. 实例化 DQN 模型
# DQN有自己的一套超参数，例如Replay Buffer大小
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,          # 学习率
    buffer_size=50000,         # Replay Buffer 的大小
    learning_starts=1000,      # 在收集这么多经验后再开始训练
    batch_size=32,             # 每次更新时采样的批量大小
    tau=1.0,                   # 软更新参数
    gamma=0.99,                # 折扣因子
    train_freq=(1, "step"),      # 每10步都训练一次
    gradient_steps=1,
    exploration_fraction=0.1,    # 探索部分占总步数的比例
    exploration_final_eps=0.05,  # 探索率最终下降到的值
    tensorboard_log="./dqn_circuit_tensorboard/"
)

# 3. 训练模型
print("开始训练 DQN 模型...")
model.learn(total_timesteps=100000) # DQN 通常需要更多的样本
print("训练完成！")

# 4. 保存模型
model.save("dqn_circuit_model")
print("模型已保存。")

# 5. 测试模型
print("\n--- 开始测试训练好的DQN模型 ---")
obs, info = env.reset()
for i in range(env.num_initial_candidates):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("回合结束！")
        final_placements = [env.initial_candidate_branch[idx] for idx in env.placed_capacitor_indices]
        print(f"最终放置的电容位置: {final_placements}")
        print(f"共使用了 {len(final_placements)} 个电容。")
        break
env.close()