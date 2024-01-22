#!/usr/bin/env python
# encoding: utf-8

import os
import re
import time
import shutil
import subprocess
import SimpleITK as sitk
import multiprocessing as mp
import tkinter as tk
from tkinter import ttk

# from utils import HardwareInfo
from output import StatisticAnalyzer, FileMerger


# ======================================= Run Simulation =======================================

def is_docker_engine_running(system: str = "Windows") -> bool:
    """Check if docker engine is running.
    Args:
        system (str, optional): System type. Defaults to "Windows".
    Returns:
        bool: True if docker engine is running.
    """
    if system == "Windows":
        try:
            subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

def start_docker():
    try:
        # 尝试启动 Docker Desktop（根据您的 Docker Desktop 版本，路径可能不同）
        subprocess.run("C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe", check=True)
    except Exception as e:
        print(f"启动 Docker 失败: {e}")

def run_macro(working_dir, macro_dir, how='docker', **kwargs):
    """Run macro file in docker.

    Args:
        working_dir (str): GATE working directory.
        macro_dir (dir): Macro file directory in working directory.
        how (str, optional): How to Run Gate. Defaults to 'docker'.
    """
    if how == 'docker':
        while not is_docker_engine_running():
            start_docker()
            print("Docker is starting...")
            time.sleep(5)
        print("Docker is running.")

        command = f"docker run -i --rm -e TZ=Asia/Shanghai -v {working_dir}:/APP opengatecollaboration/gate {macro_dir}"
        print("Running command: ", command)
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        print("Process finished with code: ", process.returncode)
        log = False if 'log' not in kwargs else kwargs['log']
        if log:
            with open(os.path.join(working_dir, macro_dir.replace('.mac', '.txt')), 'w') as f:
                f.write(process.stdout)
    else:
        raise NotImplementedError

# ======================================= Split Macro file =======================================

def split_macro(macro_dir, parallel: int):
    """Splic main.mac into several mac files (i.mac) into folder "main" for parallel simulation, 
    which will store output files in folder "main/i"
    Args:
        origin_file (_type_): path of main.mac
        parallel (int): number of parallel simulation
    """
    mac_name = os.path.basename(macro_dir)[:-4]
    folder_path = os.path.join(os.path.dirname(macro_dir), mac_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    with open(macro_dir, "r") as f:
        origin_lines = f.readlines()
    # Find the row of N, Stat, Dose, Region to modify
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

    # Generate i.mac files
    for i in range(1, parallel + 1):
        output_folder = "/".join(stat_path[:-1]) + "/" + mac_name + f"/{i}/"
        lines = origin_lines.copy()
        lines[row_N] = f"/gate/application/setNumberOfPrimariesPerRun {int(N / parallel + 1)}\n"
        lines[row_stat] = "/gate/actor/stat/save " + output_folder + f"{stat_path[-1]}\n"
        lines[row_dose] = "/gate/actor/dose3d/save " + output_folder + f"{dose_path[-1]}\n"
        if region_path is not None:
            lines[row_region] = "/gate/actor/dose3d/outputDoseByRegions " + output_folder + f"{region_path[-1]}\n"

        with open(os.path.join(folder_path, str(i) + ".mac"), 'w') as f:
            f.writelines(lines)
        
        if not os.path.exists(os.path.join(folder_path, str(i))):
            os.makedirs(os.path.join(folder_path, str(i)), exist_ok=True)

# ======================================= Run Simlation in cluster =======================================

def run_cluster(working_dir, sim_folder: str, parallel: int, how='docker', **kwargs):
    # First, split macro file
    split_macro(f"{working_dir}/{sim_folder}/main.mac", parallel)

    # Run Simulation
    ps = list()
    for i in range(1, parallel + 1):
        macro_dir = f"{sim_folder}/main/{str(i)}.mac"
        ps.append(mp.Process(target=run_macro, args=(working_dir, macro_dir, how), kwargs=kwargs))
        ps[-1].start()

    print("All processes started.")
    for p in ps:
        p.join()
    print("All processes finished.")

class SimulationSupervisor(object):
    """
    Supervise simulation processes based on the Statistic.txt file.
    """
    def __init__(self, working_dir: str, sim_folder: str) -> None:
        self.macros = [os.path.join(working_dir, sim_folder, "main", fname).replace("\\", "/")
              for fname in os.listdir(os.path.join(working_dir, sim_folder, "main")) if fname.endswith(".mac")]
        self.parallel = len(self.macros)

        self.stats = [os.path.join(macro[:-4], "Statistic.txt") for macro in self.macros]
        while sum([os.path.exists(fpath) for fpath in self.stats]) < self.parallel:
            print("Waiting for the All Statistic.txt to generate...")
            time.sleep(2)

        self.n_per_run = 0
        with open(self.macros[0], "r") as f:
            for line in f.readlines():
                if re.match(pattern="/gate/application/setNumberOfPrimariesPerRun\\s", string=line):
                    self.n_per_run = int(float(line.strip().split()[-1]))
        assert self.n_per_run != 0, "No number of primary per run in macro file."
    
    def generate_report(self):
        
        analyzers = [StatisticAnalyzer(fpath, self.n_per_run) for fpath in self.stats]
        percents = [a.current_n / self.n_per_run for a in analyzers]
        report = list()
        report.append("==============================Simulation Supervision==============================")
        report.append(f"Parallel: {self.parallel};\t Done: {sum(percents)/self.parallel:.2%}\t\t")
        ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        report.append(f"Current Time: {ctime}")
        report.append(f"Simulation Folder: {self.macros[0].split('/')[-4:-2]}")
        report.append("==================================================================================")
        report.append("{:10}{:12}{:25}".format("PERCENT", "SPEED(N/s)", "Finished Time"))

        for i, a in enumerate(analyzers):
            report.append(f"{percents[i]:<10.2%}{int(a.speed):<12}{a.final_time:25}")
        
        report = "\n".join(report)
        
        return report, percents

    def display_in_command(self, refresh_time=3):
        percents = [0] * self.parallel
        while sum(percents) < self.parallel:
            os.system("cls")
            report, percents = self.generate_report()
            print(report)
            time.sleep(refresh_time)
    
    def display_in_tkinter(self, frame: tk.Frame, refresh_time=3, pbar_length=600):

        pbar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=600, mode='determinate')
        pbar.pack()

        report_frame = tk.Frame(frame)
        report_frame.pack()
        percents = [0] * self.parallel
        while sum(percents) < self.parallel:
            for widget in report_frame.winfo_children():
                widget.destroy()
            report, percents = self.generate_report()
            tk.Label(report_frame, text=report, font=("Courier New", 12), justify=tk.LEFT
                     ).pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
            pbar['value'] = sum(percents) / self.parallel * 100
            time.sleep(refresh_time)
        
        tk.Label(report_frame, text="Simulation Finished!", font=("Courier New", 20), justify=tk.LEFT
                     ).pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

class ClusterMerger(object):
    def __init__(self, working_dir, sim_folder) -> None:
        self.sim_dir = os.path.join(working_dir, sim_folder)

        self.folders = [os.path.join(self.sim_dir, "main", folder) 
                        for folder in os.listdir(os.path.join(self.sim_dir, "main")) 
                        if os.path.isdir(os.path.join(self.sim_dir, "main", folder))]
        self.parallel = len(self.folders)

    def merge_Dose_Edep(self, file_type) -> None:
        assert file_type in ["Dose", "Edep"], "file_type must be Dose or Edep"

        doses = [os.path.join(folder, "output-" + file_type + ".mhd") for folder in self.folders]
        squareds = [os.path.join(folder, "output-" + file_type + "-Squared.mhd") for folder in self.folders]
        Ns = [StatisticAnalyzer(os.path.join(folder, "Statistic.txt")).current_n for folder in self.folders]
        if not (all([os.path.exists(dose) for dose in doses]) and
                all([os.path.exists(squared) for squared in squareds])):
            print("File not found in at least one folder.")
            return 0

        dose, squared, uncertainty = FileMerger.DoseOrEdep(doses, squareds, Ns)

        sitk.WriteImage(dose, os.path.join(self.sim_dir, file_type + ".nii"))
        sitk.WriteImage(squared, os.path.join(self.sim_dir, file_type + "Squared.nii"))
        sitk.WriteImage(uncertainty, os.path.join(self.sim_dir, file_type + "Uncertainty.nii"))

    def merge_NOH(self) -> None:
        NOHs = [os.path.join(folder, "output-NbOfHits.mhd") for folder in self.folders]
        if not all([os.path.exists(NOH) for NOH in NOHs]):
            print("NbOfHits.mhd not found in at least one folder.")
            return 0
        NOH = FileMerger.NumberOfHits(NOHs)
        sitk.WriteImage(NOH, os.path.join(self.sim_dir, "NbOfHits.nii"))

    def merge_Stat(self) -> None:
        fpath_list = [os.path.join(folder, "Statistic.txt") for folder in self.folders]
        if not all([os.path.exists(fpath) for fpath in fpath_list]):
            print("Statistic.txt not found in at least one folder.")
            return 0

        stat = FileMerger.Statistic(fpath_list)
        with open(os.path.join(self.sim_dir, "MergeStatistic.txt"), 'w') as f:
            f.writelines(stat)

    def merge_DoseByRegion(self) -> None:
        fpath_list = [os.path.join(folder, "DoseByRegion.txt") for folder in self.folders]
        stat_list = [os.path.join(folder, "Statistic.txt") for folder in self.folders]
        if not (all([os.path.exists(fpath) for fpath in fpath_list]) and 
                all([os.path.exists(fpath) for fpath in stat_list])):
            print("DoseByRegion not found in at least one folder.")
            return 0

        DoseByRegion = FileMerger.DoseByRegion(fpath_list, stat_list)
        with open(os.path.join(self.sim_dir, "DoseByRegion.txt"), 'w') as f:
            f.writelines(DoseByRegion)

    def __call__(self, remerge=True):

        if (remerge is False) and os.path.exists(os.path.join(self.sim_dir, "DoseUncertainty.nii")):
            pass
        else:
            self.merge_Dose_Edep("Dose")

        if (remerge is False) and os.path.exists(os.path.join(self.sim_dir, "EdepUncertainty.nii")):
            pass
        else:
            self.merge_Dose_Edep("Edep")

        if (remerge is False) and os.path.exists(os.path.join(self.sim_dir, "NbOfHits.nii")):
            pass
        else:
            self.merge_NOH()

        if (remerge is False) and os.path.exists(os.path.join(self.sim_dir, "MergeStatistic.txt")):
            pass
        else:
            self.merge_Stat()

        if (remerge is False) and os.path.exists(os.path.join(self.sim_dir, "DoseByRegion.txt")):
            pass
        else:
            self.merge_DoseByRegion()


if __name__ == "__main__":
    # split_macro(r"D:\MP\PSDoseCalculator_data\data\case0\sim0\main.mac", 4)

    # run_cluster("D:/MP/PSDoseCalculator_data", "data/case0/sim0", 4, log=True)

    s = SimulationSupervisor("D:/MP/PSDoseCalculator_data", "data/case0/sim0")
    s.display_in_command()

    # ClusterMerger("D:/MP/PSDoseCalculator_data", "data/case0/sim0")()

    
