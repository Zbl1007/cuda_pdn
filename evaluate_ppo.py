import os
import torch
import numpy as np
import time
import re
import yaml
from PPO import CircuitEnv, CustomCombinedExtractor # 确保从您的文件中导入
from sb3_contrib import MaskablePPO

# ==================== 参数设置 ====================
# 指向你训练好的、最终版本的模型文件
MODEL_TO_EVALUATE = "models/ppo_circuit_model_parallel_continued_micro150_812.zip" # 或者你最终保存的那个文件名
case = 'ascend910'
# 环境配置文件 (与训练时保持一致)
CKT_FILE = f'data/{case}.yaml'
RESULT_FILE = f'data/{case}_result_tsv.yaml'

RESULT_FILE_1 = f'{case}_result_dtc_output.yaml'

# ==================== 主评估逻辑 ====================
if __name__ == '__main__':
    
    # 1. --- 创建一个单一的、用于评估的环境 ---
    # 我们不需要并行化，也不需要随机种子，因为我们想看模型的确定性表现
    # 设置 render_mode='console' 可以让我们看到每一步的打印输出
    print("正在创建评估环境...")
    start = time.time()
    eval_env = CircuitEnv(
        ckt_file=CKT_FILE,
        result_file=RESULT_FILE,
        render_mode='console'  # 开启渲染，看到每一步的详细输出
    )
    print("评估环境创建完成！\n")

    # 2. --- 加载训练好的 PPO 模型 ---
    if not os.path.exists(MODEL_TO_EVALUATE):
        print(f"错误：找不到模型文件 '{MODEL_TO_EVALUATE}'。请检查路径是否正确。")
        exit()
        
    print(f"正在从 '{MODEL_TO_EVALUATE}' 加载模型...")
    # 注意：加载时不需要提供环境，但为了确保策略结构匹配，提供一个环境实例是好习惯
    # 我们在这里不传递 env，因为我们将在之后手动设置它
    model = MaskablePPO.load(MODEL_TO_EVALUATE)
    print("模型加载成功！\n")
    
    # 3. --- 运行一个完整的评估回合 ---
    print("="*20 + " 开始评估 " + "="*20)
    
    # 获取环境的初始状态
    # 注意：reset() 会自动调用 render()，打印初始状态
    obs, info = eval_env.reset()
    
    # 初始化一些用于记录的变量
    terminated = False
    truncated = False
    total_reward = 0.0
    num_steps = 0
    
    start_time = time.time()
    
    while not terminated and not truncated:
        # 获取动作掩码
        action_masks = eval_env.action_masks()
        
        # 让模型根据当前观察值和动作掩码来预测一个“最佳”动作
        # deterministic=True 表示我们不进行随机探索，而是选择概率最高的动作
        action, _states = model.predict(
            observation=obs,
            action_masks=action_masks,
            deterministic=True
        )
        action = action.item()
        # print(f"action:{action}")
        # print(type(action))

        
        # 在环境中执行这个动作
        obs, reward, terminated, truncated, info = eval_env.step(action)
        print(f"action:{reward}")
        # 累加奖励和步数
        total_reward += reward
        num_steps += 1
        
        # 短暂暂停，以便观察输出
        time.sleep(0.1) 

    end_time = time.time()
    
    # 4. --- 打印最终的评估结果 ---
    print("\n" + "="*20 + " 评估结束 " + "="*20)
    print(f"评估耗时: {end_time - start_time:.2f} 秒")
    print(f"总步数 (放置的电容数量): {num_steps}")
    print(f"最终累积奖励: {total_reward:.4f}")
    
    # 检查最终状态
    if terminated:
        print("\n🎉 结论: 模型成功找到了一个解决方案！")
    elif truncated:
        print("\n🔥 结论: 模型在用尽所有可用位置后，仍未能找到解决方案。")
        
    
    end = time.time()
    print(f"总运行时间: {end - start:.2f} 秒")
        
    # 你还可以获取最终的电容布局方案
    final_placements = eval_env.placed_capacitor_indices
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
    # 关闭环境
    eval_env.close()