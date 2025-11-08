import yaml
import numpy as np
import torch
import re
import matplotlib.pyplot as plt
import time
import random
import math
import copy # 用于深拷贝对象

import sys
import os
# 将上级目录加入 sys.path
# 请确保这个路径根据你的实际项目结构是正确的
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 假设依赖的模块在当前目录或Python路径下，如果不在，需要取消注释并修正上面的路径
from AcAdjoint import AcAdjointFunction
from AcSimulation import AcSimulationPardiso, AcSimulationCuDSS # 或 AcSimulationScipy
from Circuit import Circuit, BranchType
from build_ckt import build_ac_ckt

# --- 辅助函数 ---
def get_z_mask_flat_target(frequencies_hz, flat_target_value_ohm):
    """
    根据一个固定的目标阻抗值生成平坦的 Z_mask。
    """
    return np.full_like(frequencies_hz, flat_target_value_ohm, dtype=float)

# 这里还有待考量 返回的最大阻抗是仿真得到的一组解中的最大值
def calculate_pdn_impedance_objective(individual_indices, sim_data_dict, target_impedance_value_for_n_no_ok):
    """
    计算给定电容布局 (individual_indices) 下的PDN阻抗。
    返回:
        max_impedance (float): 在频率范围内的最大阻抗值 (对应论文中的 Z_obj)。
        n_no_ok (int): 阻抗超过 target_impedance_value_for_n_no_ok 的频点数。
    """
    frequencies = sim_data_dict["frequencies"]
    # z_mask_target = sim_data_dict["z_mask_values"] # 这个是绘图用的，n_no_ok用传入的标量值
    sims = sim_data_dict["sims"]
    num_freq = len(frequencies)
    impedances_np = np.zeros(num_freq, dtype=float)

    current_c_values_for_candidates = torch.zeros_like(sim_data_dict["base_c_values_all"]) # 长度为 num_total_candidates
    if individual_indices: # individual_indices 是 0 到 num_total_candidates-1 之间的索引列表
        selected_indices_tensor = torch.tensor(individual_indices, dtype=torch.long)
        if selected_indices_tensor.numel() > 0:
            # 检查索引有效性
            if selected_indices_tensor.max() < sim_data_dict["base_c_values_all"].shape[0] and selected_indices_tensor.min() >= 0:
                current_c_values_for_candidates[selected_indices_tensor] = sim_data_dict["base_c_values_all"][selected_indices_tensor]
            else:
                print(f"错误: calculate_pdn_impedance_objective 中个体索引无效。个体: {individual_indices}")
                return float('inf'), float('inf')

    for i, (freq, sim_instance) in enumerate(zip(frequencies, sims)):
        try:
            # AcAdjointFunction 需要的 c_idx_sim 是所有候选电容的MNA索引 (sim_data_dict["c_indices_all"])
            # c_val_sim 是这些候选电容当前的值 (current_c_values_for_candidates)
            i_voltage, _ = AcAdjointFunction.apply(
                sim_data_dict["g_index"], sim_data_dict["r_index"], sim_data_dict["c_indices_all"], # 所有候选电容的MNA索引
                sim_data_dict["l_index"], sim_data_dict["xc_index"], sim_data_dict["xl_index"],
                sim_data_dict["i_index"], sim_data_dict["v_index"], sim_data_dict["all_exc_index"],
                sim_data_dict["g_value"], sim_data_dict["r_value"], current_c_values_for_candidates, # 候选电容的当前值
                sim_data_dict["l_value"], sim_data_dict["xc_value"], sim_data_dict["xl_value"],
                sim_data_dict["all_exc_value_zin_calc"], freq, sim_instance
            )
            impedances_np[i] = i_voltage.abs().item()
        except Exception as e:
            # print(f"错误: 目标函数计算中仿真失败，频率 {freq}: {e}")
            impedances_np[i] = float('inf') # 若仿真失败，则惩罚

    if np.isinf(impedances_np).any() or np.isnan(impedances_np).any():
        max_impedance = float('inf')
        n_no_ok = float('inf')
    else:
        # 这里还有待考量 是否最大阻抗就代表了这一组解
        max_impedance = np.max(impedances_np)
        n_no_ok = np.sum(impedances_np >= target_impedance_value_for_n_no_ok)

    return max_impedance, n_no_ok


# --- 你的绘图函数 ---
def plot_zin_curves(individual_to_plot,
                    sim_data_dict,
                    plot_title_prefix,
                    case_name,
                    num_decaps, # 代表当前布局方案中的电容数量 (D)
                    output_filename):
    """
    生成并保存Z_in随频率变化的对数-对数图。
    包括: 给定个体的Z_in, Z_mask, 以及基线Z_in (无电容)。
    """
    print(f"Plotting for: {plot_title_prefix}, Nd={num_decaps}")
    frequencies = sim_data_dict["frequencies"]
    z_mask = sim_data_dict["z_mask_values"] # 这是绘图用的目标阻抗曲线
    sims = sim_data_dict["sims"]

    plt.figure(figsize=(12, 7))

    # 1. 计算并绘制给定个体的 Z_in
    zin_optimized = np.zeros_like(frequencies)
    # c_val_optimized 是一个长度为 num_total_candidates 的张量
    c_val_optimized = torch.zeros_like(sim_data_dict["base_c_values_all"])
    if individual_to_plot: # individual_to_plot 是选中的候选电容索引列表 (0 到 N_cand-1)
        selected_indices = torch.tensor(individual_to_plot, dtype=torch.long)
        if selected_indices.numel() > 0:
            if selected_indices.max() < sim_data_dict["base_c_values_all"].shape[0] and selected_indices.min() >=0:
                 c_val_optimized[selected_indices] = sim_data_dict["base_c_values_all"][selected_indices]
            else:
                print(f"Error: Invalid index in individual_to_plot for plotting. Max index: {selected_indices.max()}, Base C shape: {sim_data_dict['base_c_values_all'].shape[0]}")
                # 出错则不绘制此曲线或绘制为nan
                zin_optimized[:] = np.nan # 将所有值设为nan

    if not np.all(np.isnan(zin_optimized)): # 仅当c_val_optimized有效时才计算
        for i, (freq, sim_instance) in enumerate(zip(frequencies, sims)):
            try:
                i_voltage, _ = AcAdjointFunction.apply(
                    sim_data_dict["g_index"], sim_data_dict["r_index"], sim_data_dict["c_indices_all"],
                    sim_data_dict["l_index"], sim_data_dict["xc_index"], sim_data_dict["xl_index"],
                    sim_data_dict["i_index"], sim_data_dict["v_index"], sim_data_dict["all_exc_index"],
                    sim_data_dict["g_value"], sim_data_dict["r_value"], c_val_optimized, # 使用构建好的c_val_optimized
                    sim_data_dict["l_value"], sim_data_dict["xc_value"], sim_data_dict["xl_value"],
                    sim_data_dict["all_exc_value_zin_calc"], freq, sim_instance
                )
                zin_optimized[i] = i_voltage.abs().item()
            except Exception as e:
                print(f"Error during optimized Zin simulation for plotting at freq {freq}: {e}")
                zin_optimized[i] = np.nan
    
    plt.loglog(frequencies, zin_optimized, label=f'Optimized $Z_{{in}}$ (Nd={num_decaps})', linewidth=2)

    # 2. 绘制 Z_mask
    plt.loglog(frequencies, z_mask, '--', label='$Z_{{mask}}$ Target', color='red')

    # 3. 计算并绘制基线 Z_in (无电容)
    zin_baseline = np.zeros_like(frequencies)
    c_val_baseline = torch.zeros_like(sim_data_dict["base_c_values_all"]) # 全零表示无电容被选中
    for i, (freq, sim_instance) in enumerate(zip(frequencies, sims)):
        try:
            i_voltage, _ = AcAdjointFunction.apply(
                sim_data_dict["g_index"], sim_data_dict["r_index"], sim_data_dict["c_indices_all"],
                sim_data_dict["l_index"], sim_data_dict["xc_index"], sim_data_dict["xl_index"],
                sim_data_dict["i_index"], sim_data_dict["v_index"], sim_data_dict["all_exc_index"],
                sim_data_dict["g_value"], sim_data_dict["r_value"], c_val_baseline, 
                sim_data_dict["l_value"], sim_data_dict["xc_value"], sim_data_dict["xl_value"],
                sim_data_dict["all_exc_value_zin_calc"], freq, sim_instance
            )
            zin_baseline[i] = i_voltage.abs().item()
        except Exception as e:
            print(f"Error during baseline Zin simulation for plotting at freq {freq}: {e}")
            zin_baseline[i] = np.nan

    plt.loglog(frequencies, zin_baseline, ':', label='$Z_{{in}}$ (No Decaps)', color='gray')

    # Plot details
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Impedance Magnitude (Ohm)")
    plt.title(f"{plot_title_prefix} - PDN Impedance ({case_name}, Nd={num_decaps})")
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.legend()

    # 动态Y轴限制
    upper_y_limit = 1.0
    all_valid_data = np.concatenate([
        zin_optimized[~np.isnan(zin_optimized) & ~np.isinf(zin_optimized)],
        zin_baseline[~np.isnan(zin_baseline) & ~np.isinf(zin_baseline)],
        z_mask[~np.isnan(z_mask) & ~np.isinf(z_mask)]
    ])
    if all_valid_data.size > 0:
        upper_y_limit = max(upper_y_limit, np.max(all_valid_data) * 1.5) 
    
    min_y_limit = 1e-4 
    if all_valid_data.size > 0:
        proposed_min = np.min(all_valid_data) * 0.5
        min_y_limit = min(1e-3, proposed_min if proposed_min > 0 else 1e-3)
        min_y_limit = max(1e-4, min_y_limit) 

    plt.ylim(bottom=min_y_limit, top=upper_y_limit)
    plt.tight_layout()
    
    try:
        plt.savefig(output_filename, dpi=300)
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
    plt.close()


# --- PSO 特定组件 ---
class Particle:
    def __init__(self, num_dimensions, candidate_indices_pool):
        self.num_dimensions = num_dimensions  # 等于当前尝试的电容数量 D
        self.candidate_indices_pool = candidate_indices_pool # 所有候选位置的索引列表 [0, 1, ..., N_cand-1]
        self.num_total_candidates = len(self.candidate_indices_pool)

        if self.num_total_candidates == 0 and self.num_dimensions > 0:
            raise ValueError("候选池为空，但尝试选择电容。")
        if self.num_dimensions > self.num_total_candidates:
             raise ValueError(f"尝试选择 {self.num_dimensions} 个电容，但只有 {self.num_total_candidates} 个候选。")


        if num_dimensions > 0:
            # 从 `candidate_indices_pool` 中随机抽取 `num_dimensions` (D) 个不重复的索引作为粒子的初始位置。
            self.position = np.array(random.sample(self.candidate_indices_pool, self.num_dimensions), dtype=int)
        else:
            self.position = np.array([], dtype=int)
        
        # 粒子的速度决定了它在解空间中移动的方向和幅度。
        # 初始速度通常随机设置在一个较小的范围内。
        # `np.random.uniform(-1, 1, self.num_dimensions)` 会生成 D 个在 [-1, 1) 范围内的随机浮点数。
        # `* 0.1` 将初始速度的幅度缩小，这有助于防止粒子一开始就飞得太远，导致搜索不稳定。
        self.velocity = np.random.uniform(-1, 1, self.num_dimensions) * 0.1 # 缩小初始速度范围

        self.pbest_position = copy.deepcopy(self.position)
        self.pbest_value = float('inf')  # PSO 通常是最小化问题
        self.current_value = float('inf')
        self.current_n_no_ok = float('inf')


    def _ensure_unique_and_valid_position(self, temp_pos_float):
        """
        辅助函数，将浮点位置转换为D个唯一且有效的候选索引。
        这是一个关键且复杂的步骤，需要鲁棒的策略。
        """
        if self.num_dimensions == 0:
            return np.array([], dtype=int)

        # 策略：将浮点位置的每个维度值视为对候选池中对应索引的“吸引力”
        # 选择吸引力最大的D个唯一索引。这需要将D维的粒子位置扩展到总候选数维度。
        #
        # 简化（但可能不是最优PSO行为）且更直接的离散化：
        # 1. 将速度加到当前整数位置上，得到浮点位置。
        # 2. 对浮点位置的每个维度进行四舍五入，并clip到候选索引的有效范围 [0, num_total_candidates-1]。
        # 3. 处理重复：如果四舍五入后出现重复的索引，需要进行修复。
        
        # 步骤1 & 2: 更新并初步离散化
        new_position_indices = np.round(temp_pos_float).astype(int)
        new_position_indices = np.clip(new_position_indices, 0, self.num_total_candidates - 1 if self.num_total_candidates > 0 else 0)

        # 步骤3: 保证唯一性
        unique_indices = []
        seen = set()
        
        # 优先保留那些接近原粒子位置且不重复的
        # (为了简单，先直接取，然后修复)
        for idx in new_position_indices:
            if idx not in seen:
                unique_indices.append(idx)
                seen.add(idx)
        
        # 如果唯一索引数量不足 D，从剩余候选池中随机补充
        if len(unique_indices) < self.num_dimensions:
            remaining_candidates = [c for c in self.candidate_indices_pool if c not in seen]
            random.shuffle(remaining_candidates) # 随机打乱
            # 缺多少补多少
            needed = self.num_dimensions - len(unique_indices)
            if len(remaining_candidates) >= needed:
                unique_indices.extend(remaining_candidates[:needed])
            else:
                # 如果可用补充的都不够（例如D非常接近总候选数），这是一个问题
                # print(f"警告：修复重复时，可用补充的候选不足。已选 {len(unique_indices)}，需 {self.num_dimensions}，剩余 {len(remaining_candidates)}")
                unique_indices.extend(remaining_candidates) # 添加所有可用的
                # 如果还不够，从已选的里面随机重复（这会破坏唯一性，除非后续有处理，或者接受更少的维度）
                # 或者，更好的做法是标记此粒子更新无效，或重新随机生成不足的部分。
                # 为了保证D个，如果实在不够，只能从整个候选池中随机再选，但这丢失了PSO的移动信息。
                # 一个折中：如果修复后数量不对，则此次位置不更新（或者只更新部分维度）。
                # 这里我们强制选择D个，如果unique_indices不足D，则从整个候选池中随机补充直到满足D个，同时保证唯一性。
                current_selection_set = set(unique_indices)
                full_candidate_set = set(self.candidate_indices_pool)
                while len(unique_indices) < self.num_dimensions and len(current_selection_set) < self.num_total_candidates :
                    available_for_random_add = list(full_candidate_set - current_selection_set)
                    if not available_for_random_add: break # 没有可添加的了
                    chosen_random = random.choice(available_for_random_add)
                    unique_indices.append(chosen_random)
                    current_selection_set.add(chosen_random)

        # 如果最终数量还是不对（极端情况），则重新随机采样D个，但这几乎完全丢失了PSO的移动
        if len(set(unique_indices)) != self.num_dimensions :
            # print(f"警告：最终选择的索引数量 ({len(set(unique_indices))}) 与维度D ({self.num_dimensions}) 不符。重新随机采样。")
            if self.num_dimensions <= self.num_total_candidates:
                 return np.array(random.sample(self.candidate_indices_pool, self.num_dimensions), dtype=int)
            else: # D 大于总候选数，不应该发生，已在run_pso_for_d开头检查
                 return np.array([], dtype=int)


        # 排序的意义： gpt给出的解释 为了规范化的粒子位置 避免出现[1,3,5] 、[1,5,3]后选择decap位置出错这种现象
        return np.array(sorted(list(set(unique_indices))[:self.num_dimensions])) # 取D个并排序

    def update_velocity(self, gbest_position, w, c1, c2):
        if self.num_dimensions == 0:
            return

        r1 = np.random.rand(self.num_dimensions)
        r2 = np.random.rand(self.num_dimensions)
        
        # 确保gbest_position和self.pbest_position与self.position的维度一致
        # 如果在初始化或某些更新后它们长度不一致，这里会出错
        # 假设它们长度始终为 self.num_dimensions
        if len(gbest_position) != self.num_dimensions or len(self.pbest_position) != self.num_dimensions:
            # print("警告: PSO速度更新时维度不匹配。gbest/pbest可能未正确初始化或D=0。跳过速度更新。")
            # 可以选择将速度设为0或随机小值
            self.velocity = np.random.uniform(-0.1, 0.1, self.num_dimensions) * 0.01
            return


        # 下一代的速度 根据论文中的公式5计算得到
        cognitive_velocity = c1 * r1 * (self.pbest_position - self.position.astype(float))
        social_velocity = c2 * r2 * (gbest_position - self.position.astype(float))
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity
        
        # 限制最大速度，例如为候选范围的10%
        # 为了防止粒子速度过大导致飞出搜索边界太远或搜索不稳定，通常会对速度进行限制。
        # 这里将最大速度限制为总候选位置数量的10% (num_total_candidates * 0.1)。
        # 如果 num_total_candidates 为0，则将 max_vel 设为1.0 (一个小的默认值)。
        # `np.clip` 函数将速度向量的每个元素限制在 [-max_vel, max_vel] 区间内。
        max_vel = self.num_total_candidates * 0.1 if self.num_total_candidates > 0 else 1.0
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)


    # 更新速度
    def update_position(self):
        if self.num_dimensions == 0:
            return
        
        # 计算浮点型新位置
        # self.position 是 int 型, self.velocity 是 float 型
        new_position_float = self.position.astype(float) + self.velocity
        
        # 调用辅助函数进行离散化和唯一性保证
        self.position = self._ensure_unique_and_valid_position(new_position_float)


    def evaluate(self, sim_data_dict, target_impedance_value):
        if self.num_dimensions == 0: # D=0 的情况
            self.current_value, self.current_n_no_ok = calculate_pdn_impedance_objective([], sim_data_dict, target_impedance_value)
        elif len(set(self.position)) != self.num_dimensions : # 检查位置是否有效（D个唯一索引）
            # print(f"警告：评估粒子时发现位置索引不唯一或数量不符。Pos: {self.position}, D: {self.num_dimensions}")
            self.current_value = float('inf')
            self.current_n_no_ok = float('inf')
        else:
            self.current_value, self.current_n_no_ok = calculate_pdn_impedance_objective(self.position.tolist(), sim_data_dict, target_impedance_value)

        # PSO是最小化问题, current_value 越小越好
        if self.current_value < self.pbest_value:
            self.pbest_value = self.current_value
            self.pbest_position = copy.deepcopy(self.position)
            # self.pbest_n_no_ok = self.current_n_no_ok # 如果也想追踪pbest的n_no_ok


def run_pso_for_d(num_decaps_to_place_D, sim_data_dict, target_impedance_value_objective,
                  candidate_indices_pool, pso_params, case_name, plot_dir_for_pso_run):
    num_particles = pso_params["num_particles"]
    max_iterations = pso_params["max_iterations"] # N
    w_start = pso_params.get("w_start", 0.9)      # Wi
    w_end = pso_params.get("w_end", 0.4)          # Wf
    c1 = pso_params.get("c1", 1.5)
    c2 = pso_params.get("c2", 1.5)
    inertia_strategy = pso_params.get("inertia_strategy", "LDIW") # 默认为线性递减

    num_total_candidates = len(candidate_indices_pool)

    if num_decaps_to_place_D == 0: # 直接计算基线
        max_imp_baseline, n_no_ok_baseline = calculate_pdn_impedance_objective([], sim_data_dict, target_impedance_value_objective)
        print(f"   [PSO D=0] 基线: 最大阻抗 = {max_imp_baseline:.4f} Ohm, N_no_ok = {n_no_ok_baseline}")
        if plot_dir_for_pso_run:
            os.makedirs(plot_dir_for_pso_run, exist_ok=True)
            plot_filename = os.path.join(plot_dir_for_pso_run, f"{case_name}_Zin_PSO_D0_Baseline.png")
            plot_zin_curves([], sim_data_dict, "基线 (D=0)", case_name, 0, plot_filename)
        return [], max_imp_baseline, n_no_ok_baseline


    if num_total_candidates == 0 and num_decaps_to_place_D > 0:
        print(f"   [PSO D={num_decaps_to_place_D}] 错误: 无候选位置可选，但尝试选择 {num_decaps_to_place_D} 个电容。")
        return None, float('inf'), float('inf')
    if num_decaps_to_place_D > num_total_candidates:
        print(f"   [PSO D={num_decaps_to_place_D}] 错误: 无法从 {num_total_candidates} 个候选位置中选择 {num_decaps_to_place_D} 个。跳过。")
        return None, float('inf'), float('inf')

    swarm = []
    for _ in range(num_particles):
        try:
            swarm.append(Particle(num_decaps_to_place_D, candidate_indices_pool))
        except ValueError as e:
            print(f"   [PSO D={num_decaps_to_place_D}] 初始化粒子时出错: {e}. 可能 D 对于候选池太大了。")
            return None, float('inf'), float('inf')


    gbest_position = None
    gbest_value = float('inf')
    gbest_n_no_ok = float('inf')

    # 初始化全局最优
    for particle in swarm:
        particle.evaluate(sim_data_dict, target_impedance_value_objective)
        if particle.pbest_value < gbest_value:
            gbest_value = particle.pbest_value
            gbest_position = copy.deepcopy(particle.pbest_position)
            gbest_n_no_ok = particle.current_n_no_ok

    print(f"   [PSO D={num_decaps_to_place_D}] 初始化后 gbest: {gbest_value:.4f}, N_no_ok: {gbest_n_no_ok}")

    for k_iter in range(max_iterations):
        kk = k_iter + 1
        # 更新惯性权重 w
        if inertia_strategy == "LDIW":
            w = w_end + (w_start - w_end) * ((max_iterations - kk) / max_iterations)
        elif inertia_strategy == "OIW": # 示例：振荡惯性权重 (参数 N1, t 来自论文或需调整)
            N1_oiw = max_iterations # 假设振荡周期参数 (这里论文中没说)
            t_oiw = 0 #  （论文中没说）
            T_oiw = (2 * N1_oiw) / (3 + 2 * t_oiw) if (3 + 2 * t_oiw) != 0 else N1_oiw # 避免除零
            w = ((w_start + w_end) / 2) + ((w_start - w_end) / 2) * np.cos((2 * np.pi * kk) / T_oiw if T_oiw !=0 else 1)
        elif inertia_strategy == "EDIW":
            # W = (Wi - Wf - d1) * exp[1 / (1 + d2*k/N)]
            denominator_ediw = 1 + (0.7 * kk) / max_iterations
            w = (w_start - w_end - kk) * math.exp(1 / denominator_ediw)
            # 确保w在合理范围内 (例如，论文中 Wi Wf 定义了范围，但公式结果可能超出)
            w = np.clip(w, 0.0, 1.5) # 裁剪到一个较大的可能范围，或严格裁剪到[Wf, Wi]
        elif inertia_strategy == "SAIW":
            # W = Wf + (Wi - Wf) * lambda^(k-1)
            # k-1 对应 k_iter (0 to N-1)
            w = w_end + (w_start - w_end) * (0.95 ** k_iter)
        elif inertia_strategy == "LIW":
            # W = Wf + (Wi - Wf) * log(a + 10k/N)
            log_term_liw = math.log(1 + (10 * kk) / max_iterations)
            w = w_end + (w_start - w_end) * log_term_liw
            # 这个w值可能非常大或为负，需要裁剪
            w = np.clip(w, w_end, w_start * 1.5) # 尝试限制在合理范围，但仍可能不按预期工作
        else: # 默认 LDIW
            w = w_end + (w_start - w_end) * ((max_iterations - kk) / max_iterations)


        for particle in swarm:
            # 这个判断是否多余，因为在上面已经遍历了swarm，并且更新了gbest_position
            if gbest_position is None and particle.num_dimensions > 0: # 如果还没有gbest(例如所有初始粒子都无效)
                 # print(f"   [PSO D={num_decaps_to_place_D}] Iter {k_iter+1}: gbest_position is None, re-evaluating a particle for gbest init.")
                 # particle.evaluate(sim_data_dict, target_impedance_value_objective) # 确保有pbest
                 if particle.pbest_value < gbest_value: # 尝试用这个粒子的pbest初始化gbest
                      gbest_value = particle.pbest_value
                      gbest_position = copy.deepcopy(particle.pbest_position)
                      gbest_n_no_ok = particle.current_n_no_ok if hasattr(particle, 'current_n_no_ok') else float('inf')

            if gbest_position is not None : # 只有当存在全局最优时才更新速度和位置
                particle.update_velocity(gbest_position, w, c1, c2)
                particle.update_position()
                particle.evaluate(sim_data_dict, target_impedance_value_objective)

                if particle.pbest_value < gbest_value:
                    gbest_value = particle.pbest_value
                    gbest_position = copy.deepcopy(particle.pbest_position)
                    gbest_n_no_ok = particle.current_n_no_ok
        
        if (k_iter + 1) % 5 == 0: # 更频繁地打印进度 每5次迭代打印一次
            print(f"   [PSO D={num_decaps_to_place_D}] Iter {k_iter+1}/{max_iterations} - gbest Z_obj: {gbest_value:.4f} (N_no_ok: {gbest_n_no_ok})")
        
        # 论文的PSO内循环终止条件是达到最大迭代次数 N
        # 也可以加入停滞检测：如果 gbest_value 连续多代没有改善

    print(f"   [PSO D={num_decaps_to_place_D}] 结束。最终 gbest Z_obj: {gbest_value:.4f}, N_no_ok: {gbest_n_no_ok}")
    if plot_dir_for_pso_run and gbest_position is not None:
        os.makedirs(plot_dir_for_pso_run, exist_ok=True)
        plot_filename = os.path.join(plot_dir_for_pso_run, f"{case_name}_Zin_PSO_D{num_decaps_to_place_D}_gbest_Zobj{gbest_value:.3f}_NnoOk{gbest_n_no_ok}.png")
        plot_zin_curves(gbest_position.tolist(), sim_data_dict, 
                        f"PSO Best (D={num_decaps_to_place_D}, $Z_{{obj}}$={gbest_value:.3f})", 
                        case_name, num_decaps_to_place_D, plot_filename)

    return gbest_position.tolist() if gbest_position is not None else None, gbest_value, gbest_n_no_ok


def find_optimal_d_with_pso():
    # --- 主要配置 ---
    CASE_NAME = "ascend910"  # 你可以更改这个来运行不同的案例
    NUM_FREQUENCY_POINTS = 100 # 频率点数量
    OUTPUT_DIR = f"../results/{CASE_NAME}_PSO_Paper" # 修改输出目录名以区分
    PLOT_DIR = os.path.join(OUTPUT_DIR, "plots_pso")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # PSO 参数 (参考论文或进行调整)
    pso_run_params = {
        "num_particles": 20,    # 论文中的 n=20
        "max_iterations": 30,   # 论文中的 N=30
        "w_start": 0.9,         # 论文中 Wi=0.9
        "w_end": 0.4,           # 论文中 Wf=0.4
        "c1": 1.5,              # 论文中 C1=1.5
        "c2": 1.5,              # 论文中 C2=1.5
        "inertia_strategy": "LDIW" # 可选: "LDIW", "OIW" (或其他你实现的策略)
    }

    # --- 文件和电路设置 ---
    file = f"../data/{CASE_NAME}.yaml"
    file_result_tsv = f"../data/2025_11/{CASE_NAME}_result_tsv.yaml"

    with open(file, "r") as f:
        design = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(file_result_tsv, "r") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    interposer_w = design["interposer"]["w"] * 5
    interposer_h = design["interposer"]["h"] * 5
    vdd = design["vdd"]
    print("构建交流电路模型...")
    ckt = build_ac_ckt(file, file_result_tsv)

    observe_branch = ["id"] # 假设 'id' 是用于计算输入阻抗的电流源名称
    candidate_branch = [
        "cd_{}_{}".format(x, y)
        for x in range(interposer_w)
        for y in range(interposer_h)
        if (x, y) not in result["tsvs"]
    ]
    candidate_dtc = [
        "dtc_{}_{}".format(x, y)
        for x in range(interposer_w)
        for y in range(interposer_h)
        if (x, y) not in result["tsvs"]
    ]
    num_total_candidates = len(candidate_dtc)

    # candidate_indices_pool 是一个从0到num_total_candidates-1的整数列表，PSO粒子位置将是这个列表的子集
    candidate_indices_pool = list(range(num_total_candidates))


    frequency_points = np.geomspace(0.1e9, 10e9, NUM_FREQUENCY_POINTS)
    
    # 目标阻抗 Z_target (论文中为 60mOhm)
    chiplet_power = design["chiplets"][0]["power"]
    # target_impedance = 0.1 * vdd * vdd / chiplet_power
    target_impedance = 9999999
    for chiplet in design["chiplets"]:
        target_impedance = min(target_impedance, 0.1 * vdd * vdd / chiplet['power'])
    # target_impedance_value_objective = 0.06 # 60 mOhm, 这是PSO内部优化 Z_obj 时比较的阈值  论文的数值
    
    # z_mask_target_values 用于绘图，可以与目标阻抗值相同（平坦），或根据需要设定形状
    z_mask_target_values = get_z_mask_flat_target(frequency_points, target_impedance)
    print(f"目标阻抗 (Z_target for PSO objective): {target_impedance:.4f} Ohm.")

    # --- 准备基础仿真数据字典 ---
    print("准备基础仿真数据字典...")
    typ, u, v, val, index = ckt.prepare_sim("0", observe_branch + candidate_branch)
    
    zin_port_index_in_typ = index[0]

        
    # 获取这些候选电容的基础电容值 (按 candidate_indices_pool 的顺序)
    c_indices_all_candidates = index[len(observe_branch) :]
    base_c_values_all_candidates = val[c_indices_all_candidates]

    

    if typ[zin_port_index_in_typ] != BranchType.I: raise ValueError("观测分支 'id' 不是电流源类型 (Type I)。")
    
    (all_exc_indices_in_typ,) = torch.where((typ == BranchType.I) | (typ == BranchType.V))
    all_exc_base_values = val[all_exc_indices_in_typ] 
    
    all_exc_val_for_zin_calc = torch.zeros_like(all_exc_base_values, dtype=torch.complex128)
    is_zin_port_in_all_exc = (all_exc_indices_in_typ == zin_port_index_in_typ)
    if not torch.any(is_zin_port_in_all_exc): 
        raise ValueError("Zin 端口 'id' 未在激励源中找到。")
    # 4. 将 "id" 端口位置的值设置为 1A (1 + 0j)  初始化
    all_exc_val_for_zin_calc[is_zin_port_in_all_exc] = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)

    # 构建 sim_data_dict_base
    # 重要：这里的 "c_indices_all" 和 "base_c_values_all" 是为了适配你的 calculate_pdn_impedance_objective 和 plot_zin_curves
    # 它们指的是 *所有候选电容* 的MNA索引和它们对应的值（按 candidate_indices_pool 的顺序）
    #存储信息
    sim_data_dict = {
        "frequencies": frequency_points,
        "z_mask_values": z_mask_target_values,
        "c_indices_all": c_indices_all_candidates,
        "base_c_values_all": base_c_values_all_candidates,
        "i_index": torch.tensor([zin_port_index_in_typ], dtype=torch.long), # For AcAdjointFunction
        "all_exc_index": all_exc_indices_in_typ, # For AcAdjointFunction
        "all_exc_value_zin_calc": all_exc_val_for_zin_calc, # For AcAdjointFunction Zin calc
        # Other fixed indices/values for AcAdjointFunction
        "g_index": torch.tensor([], dtype=torch.long), 
        "r_index": torch.tensor([], dtype=torch.long),
        "l_index": torch.tensor([], dtype=torch.long), 
        "xc_index": torch.tensor([], dtype=torch.long),
        "xl_index": torch.tensor([], dtype=torch.long), 
        "v_index": torch.tensor([], dtype=torch.long),
        "g_value": torch.tensor([], dtype=torch.float64), 
        "r_value": torch.tensor([], dtype=torch.float64),
        "l_value": torch.tensor([], dtype=torch.float64), 
        "xc_value": torch.tensor([], dtype=torch.float64),
        "xl_value": torch.tensor([], dtype=torch.float64),
        "sims": []
    }

    # Initializing simulators for each frequency...
    sim_init_start_time = time.time()
    for freq in frequency_points:
            sim_data_dict["sims"].append(AcSimulationCuDSS(typ, u, v, val, freq))
    print(f"Simulators initialized in {time.time() - sim_init_start_time:.2f}s")


    # ----------------- 粒子群算法start   -----------------
    # --- 外层循环：迭代增加电容数量 D ---
    print(f"\n开始基于PSO迭代搜索最优电容数量 D (案例: {CASE_NAME})...")
    current_D = 0 # 从0个电容开始 (论文中是从 D=0 初始化，然后循环内 D=D+1)
    
    best_solution_overall_X = None      # 最终找到的最佳布局
    best_solution_overall_Z_obj = float('inf') # 对应的最小Z_obj
    best_solution_overall_N_no_ok = float('inf')# 对应的N_no_ok
    optimal_D_found = -1                # 满足条件的最小D

    # 论文中 Algorithm 1 的循环条件是 Z_obj > Z_T
    # 我们将迭代增加 D，直到 Z_obj <= Z_T 且 N_no_ok == 0
    
    max_D_to_try = num_total_candidates # 最多尝试到所有候选位置都被用上

    start_time_total_search = time.time()
    
    # 计算D=0 (无电容)时的情况
    # 无去耦电容器时 Z_mark的值
    Z_obj_current, N_no_ok_current = calculate_pdn_impedance_objective([], sim_data_dict, target_impedance)
    print(f"D=0 (无电容): Z_obj = {Z_obj_current:.4f}, N_no_ok = {N_no_ok_current}")
    
    if Z_obj_current <= target_impedance and N_no_ok_current == 0:
        print(f"目标已在D=0时满足。")
        optimal_D_found = 0
        best_solution_overall_X = []
        best_solution_overall_Z_obj = Z_obj_current
        best_solution_overall_N_no_ok = N_no_ok_current
    else:
        current_D = 1 # 开始尝试 D=1
        while current_D <= max_D_to_try:
            print(f"\n===== 运行 PSO, 电容数量 D = {current_D} =====")
            
            pso_best_X_for_this_D, pso_Z_obj_for_this_D, pso_N_no_ok_for_this_D = run_pso_for_d(
                current_D, 
                sim_data_dict, 
                target_impedance,
                candidate_indices_pool, 
                pso_run_params, 
                CASE_NAME, 
                PLOT_DIR
            )

            print(f"D={current_D} 的结果: PSO Z_obj = {pso_Z_obj_for_this_D:.4f}, N_no_ok = {pso_N_no_ok_for_this_D}")

            # 记录全局最优解（即使当前解不满足Z_target，也可能是在所有尝试中Z_obj最小的）
            if pso_best_X_for_this_D is not None and pso_Z_obj_for_this_D < best_solution_overall_Z_obj :
                best_solution_overall_Z_obj = pso_Z_obj_for_this_D
                best_solution_overall_X = pso_best_X_for_this_D
                best_solution_overall_N_no_ok = pso_N_no_ok_for_this_D
            elif pso_best_X_for_this_D is not None and pso_Z_obj_for_this_D == best_solution_overall_Z_obj and pso_N_no_ok_for_this_D < best_solution_overall_N_no_ok:
                 best_solution_overall_X = pso_best_X_for_this_D # N_no_ok 更优
                 best_solution_overall_N_no_ok = pso_N_no_ok_for_this_D


            # 检查是否满足论文中的停止条件 Z_obj <= Z_target
            # 同时我们也希望 N_no_ok == 0
            if pso_best_X_for_this_D is not None and pso_Z_obj_for_this_D <= target_impedance and pso_N_no_ok_for_this_D == 0:
                print(f"找到满足 Z_obj <= Z_target ({target_impedance:.4f} Ohm) 且 N_no_ok=0 的解，电容数量 D = {current_D}.")
                optimal_D_found = current_D
                # 更新最优解为当前D的结果，因为这是第一个满足条件的D
                best_solution_overall_X = pso_best_X_for_this_D
                best_solution_overall_Z_obj = pso_Z_obj_for_this_D
                best_solution_overall_N_no_ok = pso_N_no_ok_for_this_D
                break # 停止外层循环，因为已找到满足条件的最小D

            current_D += 1

    total_search_time = time.time() - start_time_total_search
    print(f"\nPSO整体搜索完成，耗时 {total_search_time:.2f} 秒。")

    # --- 输出最终结果 ---
    if best_solution_overall_X is not None:
        final_D_reported = len(best_solution_overall_X)
        final_Z_obj_reported = best_solution_overall_Z_obj
        final_N_no_ok_reported = best_solution_overall_N_no_ok

        if optimal_D_found != -1:
            print(f"严格满足条件的最优D值为: {optimal_D_found}")
            # 确保我们使用的是对应 optimal_D_found 的解 (如果迭代中途更新了 overall best)
            # 如果 optimal_D_found 被设置，best_solution_overall 应该就是那时的解
        else:
            print(f"未找到严格满足 Z_obj <= Z_target 且 N_no_ok=0 的解。将报告已找到的最佳Z_obj解。")
        
        print(f"最终选定结果 -> D = {final_D_reported}, Z_obj = {final_Z_obj_reported:.4f}, N_no_ok = {final_N_no_ok_reported}")
        
        file_output_pso_dtc = os.path.join(OUTPUT_DIR, f"{CASE_NAME}_result_dtc_PSO_FinalD{final_D_reported}_Zobj{final_Z_obj_reported:.3f}.yaml")
        result_yaml_output = {
            "dtcs": [],
            "tsvs": result["tsvs"],
            # "pso_params_used": {
            #                **pso_run_params, 
            #                "final_D_reported": final_D_reported,
            #                "final_Z_obj": final_Z_obj_reported, 
            #                "final_N_no_ok": final_N_no_ok_reported,
            #                "total_search_time_sec": total_search_time, 
            #                "case_name": CASE_NAME,
            #                "target_impedance_ohm_for_Z_obj": target_impedance,
            #                "num_frequency_points": NUM_FREQUENCY_POINTS,
            #                "search_method": "PSO with Iterative D (Paper)"},
            # "solution_details": {
            #                      "num_total_candidates": num_total_candidates }
        }
        
        # 根据选中的候选索引获取坐标
        pattern = R"dtc_(\d+)_(\d+)"
        for candidate_idx in best_solution_overall_X:
             match_result = re.match(pattern, candidate_dtc[candidate_idx])
             if match_result:
                 x = int(match_result.group(1)); y = int(match_result.group(2))
                 result_yaml_output["dtcs"].append((x, y))
        with open(file_output_pso_dtc, "w") as f: 
            yaml.dump(result_yaml_output, f) 
        
        print(f"最终PSO结果已保存至 {file_output_pso_dtc}")

        if best_solution_overall_X is not None:
            final_plot_filename = os.path.join(PLOT_DIR, f"{CASE_NAME}_Zin_PSO_Final_D{final_D_reported}_Zobj{final_Z_obj_reported:.3f}_NnoOk{final_N_no_ok_reported}.png")
            plot_zin_curves(best_solution_overall_X, sim_data_dict,
                            f"PSO Fimal Result (D={final_D_reported}, $Z_{{obj}}$={final_Z_obj_reported:.3f}, $N_{{no\_ok}}$={final_N_no_ok_reported})",
                            CASE_NAME, final_D_reported, final_plot_filename)
    else:
        print("PSO搜索未能产生任何有效解。")


if __name__ == "__main__":
    # 设置随机种子以保证结果可复现性 (对于PSO等随机算法很重要)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    find_optimal_d_with_pso()