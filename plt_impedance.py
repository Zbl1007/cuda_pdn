import os
import matplotlib.pyplot as plt


def plot_impedance_curve(freq_points, impedance_tensor, num_dtcs, target_imp, iteration_step, save_dir="data/plt4"):
    """
    绘制并保存单次仿真得到的阻抗曲线图。

    Args:
        freq_points (np.array): 频率点数组。
        impedance_tensor (torch.Tensor): 包含复数阻抗的PyTorch张量。
        num_dtcs (int): 当前使用的DTC数量。
        target_imp (float): 目标阻抗值。
        iteration_step (int or str): 当前的迭代步骤或标识，用于文件名。
        save_dir (str): 保存图片的文件夹名称。
    """
    # 确保保存图片的文件夹存在
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 准备绘图数据 ---
    # 1. 计算阻抗的模
    magnitude = impedance_tensor.abs().detach().numpy()
    
    # --- 开始绘图 ---
    plt.figure(figsize=(8, 6))
    
    # 2. 绘制阻抗曲线
    plt.plot(freq_points, magnitude, label=f'Impedance with {num_dtcs} DTCs')
    
    # 3. 绘制目标阻抗参考线
    plt.axhline(y=target_imp, color='r', linestyle='--', label=f'Target Impedance ({target_imp:.4f} Ohm)')
    
    # --- 设置图表格式 ---
    plt.title(f"Step {iteration_step}: Impedance Curve with {num_dtcs} DTCs")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Impedance [Ohm]")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    
    # 4. 保存图表到文件
    filename = os.path.join(save_dir, f"step_{iteration_step}_dtcs_{num_dtcs}.png")
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")
    
    # 5. 关闭当前图表，防止在循环中重复绘制和消耗内存 (非常重要)
    plt.close()