#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : Cluster.py
    @Time       : 2022/8/26 16:28
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description：分割main.mac文件
"""
import os
import re
import time
import shutil
import SimpleITK as sitk

from utils import HardwareInfo
from GATE.OutputProcess import StatisticAnalyzer, FileMerger


# ======================================================================================================================
# Split macro script
# ======================================================================================================================
def split_mac(origin_file_path, parallel: int):
    # 将已经生成的mac文件进行分割, 全部的main文件保存在同名文件夹下

    # 创建文件夹
    mac_name = os.path.basename(origin_file_path)[:-4]
    folder_path = os.path.join(os.path.dirname(origin_file_path), mac_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 读取原mac文件
    with open(origin_file_path, "r") as f:
        origin_lines = f.readlines()
    # 获取基本信息
    row_N = -1
    row_stat = -1
    row_dose = -1
    row_region = -1
    for row in range(len(origin_lines)):
        if re.match(pattern="/gate/application/setNumberOfPrimariesPerRun\\s", string=origin_lines[row]):
            row_N = row
        if re.match(pattern="/gate/actor/stat/save\\s", string=origin_lines[row]):
            row_stat = row
        if re.match(pattern="/gate/actor/dose3d/save\\s", string=origin_lines[row]):
            row_dose = row
        if re.match(pattern="/gate/actor/dose3d/outputDoseByRegions\\s", string=origin_lines[row]):
            row_region = row
    assert -1 not in [row_N, row_stat, row_dose]

    N = int(float(origin_lines[row_N].strip().split()[-1]))
    stat_path = origin_lines[row_stat].strip().split()[-1].split('/')
    dose_path = origin_lines[row_dose].strip().split()[-1].split('/')
    if row_region != -1:
        region_path = origin_lines[row_region].strip().split()[-1].split('/')
    else:
        region_path = None

    # 循环生成子文件
    for i in range(1, parallel + 1):
        # 复制原文件
        lines = origin_lines.copy()
        # 修改部分行
        lines[row_N] = f"/gate/application/setNumberOfPrimariesPerRun {int(N / parallel + 1)}\n"
        lines[row_stat] = f"/gate/actor/stat/save {stat_path[0]}/{stat_path[1]}/cluster{i}/{stat_path[2]}\n"
        lines[row_dose] = f"/gate/actor/dose3d/save {dose_path[0]}/{dose_path[1]}/cluster{i}/{dose_path[2]}\n"
        if region_path is not None:
            lines[row_region] = f"/gate/actor/dose3d/outputDoseByRegions {region_path[0]}/{region_path[1]}/cluster{i}/{region_path[2]}\n"

        with open(os.path.join(folder_path, str(i) + ".mac"), 'w') as f:
            f.writelines(lines)

        # 创建output文件夹
        out = os.path.join(stat_path[0], stat_path[1], "cluster" + str(i))
        if not os.path.exists(out):
            os.makedirs(out)
    pass


# ======================================================================================================================
# Run simulation simultaneously (Have to run the function by main.py in terminal)
# ======================================================================================================================

def auto_cluster_parallel(pname: str, memory_percent: float = 0.8, CPU_core_left: int = 4) -> int:
    if os.path.exists(os.path.join("data", pname, "CT.nii")):
        ct_size = os.path.getsize(os.path.join("data", pname, "CT.nii"))
    elif os.path.exists(os.path.join("data", pname, "Atlas.nii")):
        ct_size = os.path.getsize(os.path.join("data", pname, "Atlas.nii"))
    else:
        raise FileExistsError("No file to calculate size")
    ct_size = round(ct_size / 1E6, 1)
    # memory
    memory_per_process = 0.05 * ct_size  # Statistic regulation
    N_memory_restriction = int((HardwareInfo.MEMORY * memory_percent - HardwareInfo.STATIC_MEMORY) / memory_per_process)
    # CPU
    N_CPU_restriction = HardwareInfo.CPU_CORE - CPU_core_left

    # use the miner one
    if N_CPU_restriction <= N_memory_restriction:
        N_max = N_CPU_restriction
    else:
        N_max = N_memory_restriction

    return N_max


# "/home/pc/Desktop/PETDose"
def run_cluster(mac_name, parallel, working_dir=HardwareInfo.WORKING_DIR):
    # Call Cluster.split_mac to generate (or regenerate) cluster scripts
    if os.path.exists(os.path.join("mac", mac_name)):
        shutil.rmtree(os.path.join("mac", mac_name))
    split_mac(origin_file_path=os.path.join("mac", mac_name + ".mac"), parallel=parallel)

    # Run Simulation
    for i in range(1, parallel + 1):
        os.system(f"""
        gnome-terminal --working-directory {working_dir} --command 'bash -c "Gate mac/{mac_name}/{i}.mac"'
        """)
        # ; exec bash
        time.sleep(0)


def run_clusters(mac_names, output, parallel=0):
    if mac_names == "all":
        mac_names = [fname.split(".")[0] for fname in os.listdir("mac")]
    else:
        assert isinstance(mac_names, list)

    for mac_name in mac_names:
        print(mac_name)
        if parallel == 0:
            parallel = auto_cluster_parallel(mac_name, CPU_core_left=HardwareInfo.CPU_CORE_LEFT)
        run_cluster(mac_name=mac_name, parallel=parallel)
        # White for the Statistic.txt to generate.
        time.sleep(600)

        # Get number of primary of each cluster from one .mac script.
        N_cluster = 0
        with open(os.path.join("mac", mac_name, "1.mac"), "r") as f:
            for line in f.readlines():
                if re.match(pattern="/gate/application/setNumberOfPrimariesPerRun\\s", string=line):
                    N_cluster = int(float(line.strip().split()[-1]))
            assert N_cluster != 0
        fpath_stats = [os.path.join(output, mac_name, "cluster" + str(i), "Statistic.txt") for i in range(1, parallel + 1)]

        MoveOn = False
        while not MoveOn:
            n_cluster = [StatisticAnalyzer(fpath).n for fpath in fpath_stats]
            if sum(n_cluster) == N_cluster * parallel:
                MoveOn = True
                print("Next Patient: ", end=" ")
            else:
                print(f"Current Patient: {mac_name} still running...")
                time.sleep(600)


# ======================================================================================================================
# Supervise simulations
# ======================================================================================================================
def supervision(output, pname):
    parallel = len(os.listdir(os.path.join("mac", pname)))

    n_cluster = 0
    with open(os.path.join("mac", pname, "1.mac"), "r") as f:
        for line in f.readlines():
            if re.match(pattern="/gate/application/setNumberOfPrimariesPerRun\\s", string=line):
                n_cluster = int(float(line.strip().split()[-1]))
        assert n_cluster != 0

    fpath_stats = [os.path.join(output, pname, "cluster" + str(i), "Statistic.txt") for i in range(1, parallel + 1)]

    RunStat = [0] * parallel
    while sum(RunStat) < parallel:
        stats = [StatisticAnalyzer(fpath) for fpath in fpath_stats]
        print("==============================Simulation Supervision==============================")
        print(f"Parallel: {parallel};\t Done: {sum(RunStat)}", end='\t\t')
        ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"Current Time: {ctime}")
        print(pname)
        print("==================================================================================")
        print("{:10}{:12}{:25}".format("PERCENT", "SPEED_AVE", "END_AVE"))

        for stat in stats:
            end = stat.seconds_end_AvePred(N=n_cluster)
            end = stat.time_output(end)
            print(f"{stat.n / n_cluster:<10.2%}{int(stat.average_speed()):<12}{end:25}")
        print()
        time.sleep(600)


# ======================================================================================================================
# Merge Output
# ======================================================================================================================

class ClusterMerger(object):
    @staticmethod
    def merge_Dose_Edep(folder_cluster, file_type) -> None:
        """

        :param folder_cluster:
        :param file_type:
        :return:
        """
        doses = list()
        squareds = list()
        Ns = list()
        for sub in os.listdir(folder_cluster):
            folder_sub = os.path.join(folder_cluster, sub)
            if os.path.isdir(folder_sub):
                doses.append(os.path.join(folder_sub, "output-" + file_type + ".mhd"))
                squareds.append(os.path.join(folder_sub, "output-" + file_type + "-Squared.mhd"))
                Ns.append(StatisticAnalyzer(os.path.join(folder_sub, "Statistic.txt")).n)

        dose, squared, N, uncertainty = FileMerger.DoseOrEdep(doses, squareds, Ns)

        sitk.WriteImage(dose, os.path.join(folder_cluster, file_type + ".nii"))
        sitk.WriteImage(squared, os.path.join(folder_cluster, file_type + "Squared.nii"))
        sitk.WriteImage(uncertainty, os.path.join(folder_cluster, file_type + "Uncertainty.nii"))
        pass

    @staticmethod
    def merge_NOH(folder_cluster) -> None:
        NOHs = list()
        for sub in os.listdir(folder_cluster):
            folder_sub = os.path.join(folder_cluster, sub)
            if os.path.isdir(folder_sub):
                NOHs.append(os.path.join(folder_sub, "output-NbOfHits.mhd"))
        NOH = FileMerger.NumberOfHits(NOHs)

        sitk.WriteImage(NOH, os.path.join(folder_cluster, "NbOfHits.nii"))
        pass

    @staticmethod
    def merge_Stat(folder_cluster) -> None:
        fpath_list = list()
        for sub in os.listdir(folder_cluster):
            folder_sub = os.path.join(folder_cluster, sub)
            if os.path.isdir(folder_sub):
                fpath_list.append(os.path.join(folder_sub, "Statistic.txt"))
        stat = FileMerger.Statistic(fpath_list)

        with open(os.path.join(folder_cluster, "MergeStatistic.txt"), 'w') as f:
            f.writelines(stat)

    @staticmethod
    def merge_DoseByRegion(folder_cluster):
        fpath_list = list()
        stat_list = list()
        for sub in os.listdir(folder_cluster):
            folder_sub = os.path.join(folder_cluster, sub)
            if os.path.isdir(folder_sub):
                fpath_list.append(os.path.join(folder_sub, "DoseByRegion.txt"))
                stat_list.append(os.path.join(folder_sub, "Statistic.txt"))

        DoseByRegion = FileMerger.DoseByRegion(fpath_list, stat_list)

        with open(os.path.join(folder_cluster, "DoseByRegion.txt"), 'w') as f:
            f.writelines(DoseByRegion)

    @staticmethod
    def merge_output(folder_cluster, remerge=True):
        if (remerge is False) and os.path.exists(os.path.join(folder_cluster, "DoseUncertainty.nii")):
            pass
        else:
            ClusterMerger.merge_Dose_Edep(folder_cluster, "Dose")

        if (remerge is False) and os.path.exists(os.path.join(folder_cluster, "EdepUncertainty.nii")):
            pass
        else:
            ClusterMerger.merge_Dose_Edep(folder_cluster, "Edep")

        if (remerge is False) and os.path.exists(os.path.join(folder_cluster, "NbOfHits.nii")):
            pass
        else:
            ClusterMerger.merge_NOH(folder_cluster)

        if (remerge is False) and os.path.exists(os.path.join(folder_cluster, "MergeStatistic.txt")):
            pass
        else:
            ClusterMerger.merge_Stat(folder_cluster)

        if (remerge is False) and os.path.exists(os.path.join(folder_cluster, "DoseByRegion.txt")):
            pass
        else:
            ClusterMerger.merge_DoseByRegion(folder_cluster)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.getcwd()))

    pass
