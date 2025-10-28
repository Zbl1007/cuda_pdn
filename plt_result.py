import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from AcAdjoint import AcAdjointFunction
from AcSimulation import AcSimulationPardiso
from Circuit import Circuit, BranchType
from build_ckt import build_ac_ckt

import re

def test_both_result():
    """
    测试both_oneshot.py的优化结果，绘制阻抗曲线对比图
    """
    # --- 配置参数 ---
    case = "ascend910"  # 修改为你的case名称
    file_design = f"data/{case}.yaml"
    # file_result = f"data/{case}_both_result.yaml"  # both_oneshot.py的输出文件
    file_result = f"dqn_{case}_result_tsv_output.yaml"  # both_oneshot.py的输出文件
    # file_result = f"{case}_result_tsv_output.yaml"  # ppo.py的输出文件

    file_plot = f"dqn_{case}_both_result_zin_comparison_new.png"
    # file_plot = f"ppo_{case}_both_result_zin_comparison_new.png"

    
    # 频率点数量
    NUM_FREQUENCY_POINTS = 100
    
    print(f"Testing both optimization result for {case}...")
    
    # --- 读取设计文件和结果文件 ---
    with open(file_design, "r") as f:
        design = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    with open(file_result, "r") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # --- 提取设计参数 ---
    interposer_w = design["interposer"]["w"]
    interposer_h = design["interposer"]["h"]
    vdd = design["vdd"]
    target_impedance = 0.1 * vdd * vdd / design["chiplets"][0]["power"]
    
    print(f"Interposer size: {interposer_w}x{interposer_h}")
    print(f"Target impedance: {target_impedance:.6f} Ohm")
    print(f"Number of TSVs: {len(result['tsvs'])}")
    print(f"TSVs: {result['tsvs']}")
    # original_dtc_placement = [
    #     'cd_8_9', 'cd_3_2', 'cd_8_11', 'cd_8_1', 'cd_8_5', 'cd_5_9', 'cd_18_0', 
    #     'cd_11_0', 'cd_17_1', 'cd_7_0', 'cd_19_0', 'cd_13_0', 'cd_19_1', 
    #     'cd_18_1', 'cd_17_0', 'cd_12_0', 'cd_16_0', 'cd_15_0', 'cd_11_1', 
    #     'cd_19_2', 'cd_9_1', 'cd_4_5', 'cd_16_1', 'cd_12_1', 'cd_8_8', 
    #     'cd_18_10', 'cd_15_1', 'cd_10_1', 'cd_2_11', 'cd_1_6', 'cd_5_0'
    # ]

    # # 定义一个正则表达式来匹配 'cd_数字_数字' 的模式
    # # \d+ 表示匹配一个或多个数字
    # # () 表示一个捕获组，我们用它来分别捕获 x 和 y 的值
    # pattern = re.compile(r'cd_(\d+)_(\d+)')

    # # 创建一个空列表来存放转换后的坐标元组
    # new_coordinates = []

    # # 遍历原始列表中的每一个字符串
    # for location_str in original_dtc_placement:
    #     # 使用正则表达式进行匹配
    #     match = pattern.search(location_str)
        
    #     # 如果匹配成功
    #     if match:
    #         # 从匹配对象中提取捕获组
    #         # match.group(1) 是第一个括号里的内容 (x)
    #         # match.group(2) 是第二个括号里的内容 (y)
    #         # 将提取出的字符串转换为整数
    #         x = int(match.group(1))
    #         y = int(match.group(2))
            
    #         # 将 (x, y) 元组添加到新列表中
    #         new_coordinates.append((x, y))
    #     else:
    #         # 如果某个字符串格式不正确，打印一个警告
    #         print(f"警告：字符串 '{location_str}' 格式不正确，无法解析。")

    # --- 打印结果 ---
    print("转换后的坐标元组列表:")
    # print(new_coordinates)
    # result['dtcs'] = new_coordinates
    print(f"Number of DTCs: {len(result['dtcs'])}")
    print(f"dtcs: {result['dtcs']}")
    
    
    # --- 构建电路 ---
    # 创建一个临时的TSV结果文件用于build_ac_ckt
    temp_tsv_file = f"data/{case}_temp_tsv.yaml"
    with open(temp_tsv_file, "w") as f:
        yaml.dump({"tsvs": result["tsvs"]}, f)
    
    ckt = build_ac_ckt(file_design, temp_tsv_file)
    
    # --- 定义观测分支和候选分支 ---
    observe_branch = ["id"]
    
    # TSV相关分支
    tsv_conductance_branch = []
    tsv_capacitance_branch = []
    tsv_inductance_branch = []
    for x, y in result["tsvs"]:
        tsv_conductance_branch.extend([f"gt1_{x}_{y}", f"gt2_{x}_{y}"])
        tsv_capacitance_branch.append(f"ct_{x}_{y}")
        tsv_inductance_branch.extend([f"xlt1_{x}_{y}", f"xlt2_{x}_{y}"])
    
    # DTC相关分支
    dtc_capacitance_branch = []
    for x, y in result["dtcs"]:
        dtc_capacitance_branch.append(f"cd_{x}_{y}")
        
    print(dtc_capacitance_branch)
    
    # 所有分支
    all_branches = (observe_branch + 
                   tsv_conductance_branch + 
                   tsv_capacitance_branch + 
                   tsv_inductance_branch + 
                   dtc_capacitance_branch)
    
    print(all_branches)
    
    # --- 频率点 ---
    frequency_points = np.geomspace(0.1e9, 10e9, NUM_FREQUENCY_POINTS)
    target_impedance_curve = np.full_like(frequency_points, target_impedance)
    
    # --- 准备仿真 ---
    typ, u, v, val, index = ckt.prepare_sim("0", all_branches)
    
    # 找到各种元件的索引
    observe_start = 0
    tsv_g_start = len(observe_branch)
    tsv_c_start = tsv_g_start + len(tsv_conductance_branch)
    tsv_l_start = tsv_c_start + len(tsv_capacitance_branch)
    dtc_c_start = tsv_l_start + len(tsv_inductance_branch)
    
    # 提取索引和值
    observe_index = index[observe_start:tsv_g_start]
    tsv_g_index = index[tsv_g_start:tsv_c_start] if tsv_conductance_branch else torch.tensor([], dtype=torch.long)
    tsv_c_index = index[tsv_c_start:tsv_l_start] if tsv_capacitance_branch else torch.tensor([], dtype=torch.long)
    tsv_l_index = index[tsv_l_start:dtc_c_start] if tsv_inductance_branch else torch.tensor([], dtype=torch.long)
    dtc_c_index = index[dtc_c_start:] if dtc_capacitance_branch else torch.tensor([], dtype=torch.long)
    
    # 提取基础值
    tsv_g_value = val[tsv_g_index] if len(tsv_g_index) > 0 else torch.tensor([], dtype=torch.float64)
    tsv_c_value = val[tsv_c_index] if len(tsv_c_index) > 0 else torch.tensor([], dtype=torch.float64)
    tsv_l_value = val[tsv_l_index] if len(tsv_l_index) > 0 else torch.tensor([], dtype=torch.float64)
    dtc_c_value = val[dtc_c_index] if len(dtc_c_index) > 0 else torch.tensor([], dtype=torch.float64)
    
    # 设置激励源
    zin_port_index = observe_index[0]
    if typ[zin_port_index] != BranchType.I:
        raise ValueError("Observation branch 'id' is not a current source.")
    
    (all_exc_index,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
    all_exc_base_value = val[all_exc_index]
    
    # 为Zin计算设置激励：id端口为1A，其他为0
    all_exc_value_zin = torch.zeros_like(all_exc_base_value, dtype=torch.complex128)
    is_zin_port = (all_exc_index == zin_port_index)
    all_exc_value_zin[is_zin_port] = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    
    # 准备固定的空索引和值（这个电路没有电阻和其他元件）
    empty_index = torch.tensor([], dtype=torch.long)
    empty_value = torch.tensor([], dtype=torch.float64)
    empty_complex_value = torch.tensor([], dtype=torch.complex128)
    
    # --- 创建仿真器 ---
    print("Initializing simulators...")
    sim_start_time = time.time()
    sims = []
    for freq in frequency_points:
        sims.append(AcSimulationPardiso(typ, u, v, val, freq))
    print(f"Simulators initialized in {time.time() - sim_start_time:.2f}s")
    
    # --- 计算优化后的阻抗曲线（有TSV+DTC） ---
    print("Calculating optimized impedance curve...")
    calc_start_time = time.time()
    optimized_impedances = np.zeros(len(frequency_points))
    
    for i, (freq, sim_instance) in enumerate(zip(frequency_points, sims)):
        try:
            # 合并所有电容索引和值
            all_c_index = torch.cat([tsv_c_index, dtc_c_index]) if len(tsv_c_index) > 0 and len(dtc_c_index) > 0 else (
                tsv_c_index if len(tsv_c_index) > 0 else dtc_c_index
            )
            all_c_value = torch.cat([tsv_c_value, dtc_c_value]) if len(tsv_c_value) > 0 and len(dtc_c_value) > 0 else (
                tsv_c_value if len(tsv_c_value) > 0 else dtc_c_value
            )
            
            i_voltage, _ = AcAdjointFunction.apply(
                tsv_g_index,           # g_index
                empty_index,           # r_index
                all_c_index,           # c_index
                empty_index,           # l_index
                empty_index,           # xc_index
                tsv_l_index,           # xl_index
                observe_index,         # i_index
                empty_index,           # v_index
                all_exc_index,         # all_exc_index
                tsv_g_value,           # g_value
                empty_value,           # r_value
                all_c_value,           # c_value
                empty_value,           # l_value
                empty_complex_value,   # xc_value
                tsv_l_value,           # xl_value
                all_exc_value_zin,     # all_exc_value
                freq,
                sim_instance
            )
            optimized_impedances[i] = i_voltage.abs().item()
        except Exception as e:
            print(f"Error at frequency {freq}: {e}")
            optimized_impedances[i] = np.nan
    
    print(f"Optimized impedance calculation completed in {time.time() - calc_start_time:.2f}s")
    
    # --- 计算基线阻抗曲线（无TSV无DTC） ---
    print("Calculating baseline impedance curve...")
    baseline_start_time = time.time()
    baseline_impedances = np.zeros(len(frequency_points))
    
    # 创建一个没有TSV和DTC的电路
    baseline_ckt = build_ac_ckt(file_design)  # 不传入TSV文件
    baseline_typ, baseline_u, baseline_v, baseline_val, baseline_index = baseline_ckt.prepare_sim("0", observe_branch)
    
    baseline_observe_index = baseline_index[:len(observe_branch)]
    baseline_zin_port_index = baseline_observe_index[0]
    (baseline_all_exc_index,) = torch.where((baseline_typ == BranchType.I) | (baseline_typ == BranchType.V))
    baseline_all_exc_base_value = baseline_val[baseline_all_exc_index]
    
    baseline_all_exc_value_zin = torch.zeros_like(baseline_all_exc_base_value, dtype=torch.complex128)
    baseline_is_zin_port = (baseline_all_exc_index == baseline_zin_port_index)
    baseline_all_exc_value_zin[baseline_is_zin_port] = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    
    # 创建基线仿真器
    baseline_sims = []
    for freq in frequency_points:
        baseline_sims.append(AcSimulationPardiso(baseline_typ, baseline_u, baseline_v, baseline_val, freq))
    
    for i, (freq, sim_instance) in enumerate(zip(frequency_points, baseline_sims)):
        try:
            i_voltage, _ = AcAdjointFunction.apply(
                empty_index,                    # g_index
                empty_index,                    # r_index
                empty_index,                    # c_index
                empty_index,                    # l_index
                empty_index,                    # xc_index
                empty_index,                    # xl_index
                baseline_observe_index,         # i_index
                empty_index,                    # v_index
                baseline_all_exc_index,         # all_exc_index
                empty_value,                    # g_value
                empty_value,                    # r_value
                empty_value,                    # c_value
                empty_value,                    # l_value
                empty_complex_value,            # xc_value
                empty_value,                    # xl_value
                baseline_all_exc_value_zin,     # all_exc_value
                freq,
                sim_instance
            )
            baseline_impedances[i] = i_voltage.abs().item()
        except Exception as e:
            print(f"Error in baseline at frequency {freq}: {e}")
            baseline_impedances[i] = np.nan
    
    print(f"Baseline impedance calculation completed in {time.time() - baseline_start_time:.2f}s")
    
    # --- 绘图 ---
    print("Creating comparison plot...")
    plt.figure(figsize=(12, 8))
    
    # 绘制各条曲线
    plt.loglog(frequency_points, optimized_impedances, 
               label=f'Optimized $Z_{{in}}$ (TSV: {len(result["tsvs"])}, DTC: {len(result["dtcs"])})', 
               linewidth=2, color='blue')
    plt.loglog(frequency_points, target_impedance_curve, '--', 
               label=f'$Z_{{mask}}$ Target ({target_impedance:.4f} Ω)', 
               linewidth=2, color='red')
    plt.loglog(frequency_points, baseline_impedances, ':', 
               label='$Z_{{in}}$ (No TSV/DTC)', 
               linewidth=2, color='gray')
    
    # 计算并显示违规点
    optimized_violations = np.sum(optimized_impedances > target_impedance)
    baseline_violations = np.sum(baseline_impedances > target_impedance)
    
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Impedance Magnitude (Ω)", fontsize=12)
    plt.title(f"PDN Impedance Comparison ({case})\n"
              f"Optimized violations: {optimized_violations}/{len(frequency_points)}, "
              f"Baseline violations: {baseline_violations}/{len(frequency_points)}", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=11)
    
    # 调整Y轴范围
    valid_optimized = optimized_impedances[~np.isnan(optimized_impedances) & ~np.isinf(optimized_impedances)]
    valid_baseline = baseline_impedances[~np.isnan(baseline_impedances) & ~np.isinf(baseline_impedances)]
    
    if len(valid_optimized) > 0 and len(valid_baseline) > 0:
        y_max = max(np.max(valid_optimized), np.max(valid_baseline), target_impedance) * 3
        y_min = min(np.min(valid_optimized), np.min(valid_baseline), target_impedance) / 10
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(file_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {file_plot}")
    
    # --- 输出统计信息 ---
    print("\n" + "="*60)
    print("OPTIMIZATION RESULT ANALYSIS")
    print("="*60)
    print(f"Case: {case}")
    print(f"Target impedance: {target_impedance:.6f} Ω")
    print(f"Frequency range: {frequency_points[0]/1e9:.1f} - {frequency_points[-1]/1e9:.1f} GHz")
    print(f"Number of frequency points: {len(frequency_points)}")
    print(f"\nOptimization components:")
    print(f"  TSVs: {len(result['tsvs'])}")
    print(f"  DTCs: {len(result['dtcs'])}")
    print(f"  Total: {len(result['tsvs']) + len(result['dtcs'])}")
    
    if len(valid_optimized) > 0:
        print(f"\nOptimized impedance statistics:")
        print(f"  Min: {np.min(valid_optimized):.6f} Ω")
        print(f"  Max: {np.max(valid_optimized):.6f} Ω")
        print(f"  Mean: {np.mean(valid_optimized):.6f} Ω")
        print(f"  Violations: {optimized_violations}/{len(frequency_points)} ({100*optimized_violations/len(frequency_points):.1f}%)")
    
    if len(valid_baseline) > 0:
        print(f"\nBaseline impedance statistics:")
        print(f"  Min: {np.min(valid_baseline):.6f} Ω")
        print(f"  Max: {np.max(valid_baseline):.6f} Ω")
        print(f"  Mean: {np.mean(valid_baseline):.6f} Ω")
        print(f"  Violations: {baseline_violations}/{len(frequency_points)} ({100*baseline_violations/len(frequency_points):.1f}%)")
    
    if len(valid_optimized) > 0 and len(valid_baseline) > 0:
        improvement = baseline_violations - optimized_violations
        print(f"\nImprovement: {improvement} fewer violation points")
    
    # 清理临时文件
    import os
    if os.path.exists(temp_tsv_file):
        os.remove(temp_tsv_file)
    
    print(f"\nTest completed successfully!")


if __name__ == "__main__":
    test_both_result() 
    
