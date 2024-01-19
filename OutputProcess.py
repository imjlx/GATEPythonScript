#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : PrepareSimulation.py
    @Time       : 2023/2/14 11:32
    @Author     : Haoran Jia
    @license    : Copyright(c) 2023 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description：
"""
import os
import time

import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List

from utils import OrganDict
from ImageProcess.Image import ImageProcessor, AtlasProcessor
from ImageProcess.InfoStat import PropertyCalculator
from ImageProcess.PET import PETSeriesProcessor, OrganCumulatedActivityCalculator
from utils.ICRPReference import F18_bladder_cumulate_activity


# ======================================================================================================================
# File Analyzation
# ======================================================================================================================

# Dosemap Process: Organ Dose
class OrganDoseCalculator(PropertyCalculator):
    def __init__(self, dosemap=None, atlas=None, pet=None, folder=None, **kwargs):
        super().__init__(dosemap=dosemap, atlas=atlas, pet=pet, **kwargs)
        self.folder = folder
        self.Ac = 1E6  # MBq·s

    def SetCumulatedActivityByInjection(self, isPerInjection=True):
        if isPerInjection:
            injection = 1E6  # Bq
        else:
            assert self.folder is not None
            pet_reader = PETSeriesProcessor(folder=self.folder)
            injection = pet_reader.GetInjectionActivityInBq()

        self.Ac = injection / PETSeriesProcessor.lamb_s

        return self.Ac

    def SetCumulatedActivityByPET(self, isPerInjection=True, **kwargs):
        assert (self.pet is not None) and (self.atlas is not None)
        calculator = OrganCumulatedActivityCalculator(pet=self.pet, atlas=self.atlas, folder=self.folder)
        self.Ac = calculator.CalculateOneOrgan(ID=10, **kwargs)

        if isPerInjection:
            assert self.folder is not None
            pet_reader = PETSeriesProcessor(folder=self.folder)
            self.Ac /= pet_reader.GetInjectionActivityInBq()
        return self.Ac

    def SetCumulatedActivityByICRP(self, age=18, isPerInjection=True):

        activity_bladder = F18_bladder_cumulate_activity(age=age)
        self.Ac = (0.21 + 0.11 + 0.079 + 0.13 + 1.7 + activity_bladder) * 3600

        if not isPerInjection:
            assert self.folder is not None
            pet_reader = PETSeriesProcessor(folder=self.folder)
            self.Ac *= pet_reader.GetInjectionActivityInBq()

        return self.Ac

    def CalculateOneOrgan(self, ID: int, N=1E9, **kwargs):
        assert (self.dosemap is not None) and (self.atlas is not None)
        dosemap_arr = AtlasProcessor.GenerateMaskedOneLineArray(
            img=self.dosemap,
            mask=AtlasProcessor.GenerateOrganMask(atlas=self.atlas, ID=ID, **kwargs)
        )
        if dosemap_arr is not None:
            # Ac[MBq·s], N[], dosemap_arr[Gy],
            dose = np.average(dosemap_arr) / N * self.Ac * 1E3  # dose[mGy]
        else:
            dose = None
        return dose


class OrganDoseUncertaintyCalculator(OrganDoseCalculator):
    def __init__(self, uncertainty=None, dosemap=None, atlas=None, **kwargs):
        super().__init__(dosemap=dosemap, atlas=atlas, uncertainty=uncertainty, **kwargs)
        pass

    def CalculateOneOrgan(self, ID: int, **kwargs) -> float:
        assert self.dosemap is not None
        assert self.atlas is not None
        assert self.uncertainty is not None

        # Call father method to calculate dose
        dose = super().CalculateOneOrgan(ID)

        if dose is not None:
            mask = AtlasProcessor.GenerateOrganMask(atlas=self.atlas, ID=ID)
            uncertainty_arr = AtlasProcessor.GenerateMaskedOneLineArray(img=self.uncertainty, mask=mask)
            dose_arr = AtlasProcessor.GenerateMaskedOneLineArray(img=self.dosemap, mask=mask)

            uncertainty_arr *= dose_arr  # change to absolute uncertainty
            uncertainty = np.sqrt(np.average(uncertainty_arr ** 2))
            uncertainty = uncertainty / dose  # change back to relative uncertainty
        else:
            uncertainty = None

        return uncertainty


class StatisticAnalyzer(object):

    def __init__(self, fpath, isGetTime=True):
        # 按行读取文件
        with open(fpath, "r") as file:
            self.lines = file.readlines()

        self.n = self.GetN()
        if isGetTime:
            self.seconds_start = self.GetTimeStart()
            self.seconds_end = self.GetTimeEnd()

    def GetN(self):
        # 获取当前运行粒子数（第二行 “# NumberOfEvents = 100000000”）
        self.n = int(self.lines[1].split("=")[-1].strip().split('.')[0])
        return self.n

    def GetTimeStart(self):
        # 获取开始时间、当前结束时间 （第9、10行 “# StartDate = Sun Aug 7 18:00:33 2022”）
        time_start = self.lines[8].split("=")[-1].strip()
        self.seconds_start = time.mktime(time.strptime(time_start, "%a %b %d %H:%M:%S %Y"))
        return self.seconds_start

    def GetTimeEnd(self):
        time_end = self.lines[9].split("=")[-1].strip()
        self.seconds_end = time.mktime(time.strptime(time_end, "%a %b %d %H:%M:%S %Y"))
        return self.seconds_end

    def average_speed(self):
        if self.n != -1:
            return self.n / (self.seconds_end - self.seconds_start)
        else:
            return -1

    def seconds_end_AvePred(self, N):
        if self.n != -1:
            return self.seconds_start + N / self.average_speed()
        else:
            return 0

    @staticmethod
    def time_output(seconds):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


class DoseByRegionAnalyzer(object):
    def __init__(self, fpath):
        # read lines
        self.file = open(fpath, 'r').readlines()
        # split white space
        self.file = [line.split() for line in self.file]
        # Get all Info: id, vol(mm3), dep(MeV), std_edep, sq_edep, dose(Gy), std_dose, sq_dose, n_hits , n_event_hits
        self.ID = [int(line[0]) for line in self.file[1:]]
        self.vol = [float(line[1]) for line in self.file[1:]]
        self.dep = [float(line[2]) for line in self.file[1:]]
        self.std_edep = [float(line[3]) for line in self.file[1:]]
        self.sq_edep = [float(line[4]) for line in self.file[1:]]
        self.dose = [float(line[5]) for line in self.file[1:]]
        self.std_dose = [float(line[6]) for line in self.file[1:]]
        self.sq_dose = [float(line[7]) for line in self.file[1:]]
        self.n_hits = [int(line[8]) for line in self.file[1:]]
        self.n_event_hits = [int(line[9]) for line in self.file[1:]]


# ======================================================================================================================
# File Generation
# ======================================================================================================================

def generate_uncertainty(dose, squared, N) -> sitk.Image:
    """
    利用DoseActor输出的dose和dose_squared计算不确定度Uncertainty
    :param dose: Dose（sums of doses due to each successive primary）
    :param squared: Dose-Squared（sum of squares of doses due to each successive primary）
    :param N: 相对应的粒子数（Number of Primary）
    :return: 计算出的不确定度（standard deviation divided by the value）
    """

    dose_arr = ImageProcessor.ReadImageAsArray(dose)
    squared_arr = ImageProcessor.ReadImageAsArray(squared)

    # 根据公式计算不确定度（若Dose为0，不确定度为1）
    uncertainty_arr = np.divide(np.sqrt(squared_arr - dose_arr ** 2 / N), dose_arr,
                                out=np.ones_like(dose_arr), where=dose_arr != 0)

    uncertainty: sitk.Image = sitk.GetImageFromArray(uncertainty_arr)
    uncertainty.CopyInformation(dose)

    return uncertainty


class FileMerger(object):
    @staticmethod
    def DoseOrEdep(value_list, squared_list, N_list) -> Tuple[sitk.Image, sitk.Image, int, sitk.Image]:
        value_list = [sitk.ReadImage(fname) for fname in value_list]
        squared_list = [sitk.ReadImage(fname) for fname in squared_list]

        value = sum(value_list)
        squared = sum(squared_list)

        N = sum(N_list)

        uncertainty = generate_uncertainty(value, squared, N)

        return value, squared, N, uncertainty

    @staticmethod
    def NumberOfHits(value_list: List) -> sitk.Image:
        if isinstance(value_list[0], str):
            value_list = [sitk.ReadImage(fname) for fname in value_list]

        return sum(value_list)

    @staticmethod
    def Statistic(stat_list: List[str]) -> List[str]:
        # Some helper functions
        def line_calculate(line_list: List[str], method: str, dtype="float"):
            # The entry and values of the lines
            entry = line_list[0].split('=')[0].strip()
            if dtype == "float":
                value_list = [float(line.split('=')[1].strip()) for line in line_list]
            elif dtype == "str":
                value_list = [line.split('=')[1].strip() for line in line_list]
            else:
                raise ValueError("Unsupported dtype")

            # calculate
            if method == "sum" and dtype == "float":
                value = sum(value_list)
            elif method == "average" and dtype == "float":
                value = np.average(value_list)
            elif method == "same":
                value = value_list[0]
            elif method == "list" and dtype == "str":
                value = "\t".join(value_list)
            else:
                raise ValueError("Unsupported method or method-dtype mismatch")

            return f"{entry}\t=\t{value}\n", entry, value

        def seconds_to_HMS(seconds) -> str:
            seconds = int(seconds)
            H = int(seconds / 3600)
            M = int((seconds - H * 3600) / 60)
            S = seconds - H * 3600 - M * 60
            return f"{H}:{M}:{S}"

        # read lines for all Statistic.txt files
        stat_list = [open(stat, 'r').readlines() for stat in stat_list]
        # Write merged stat
        stat = ["\n"] * 30
        for row in [0, 10, 11, 12, 13, 14, 15]:
            stat[row], _, _ = line_calculate([f[row] for f in stat_list], method="same")
        for row in [1, 2, 3, 4, 5, 6, 7]:
            stat[row], _, _ = line_calculate([f[row] for f in stat_list], method="sum")
        for row in [8, 9]:
            stat[row], _, _ = line_calculate([f[row] for f in stat_list], method="list", dtype="str")
        for row in [16, 17, 18]:
            stat[row], _, _ = line_calculate([f[row] for f in stat_list], method="average")

        # row 20
        stat[20] = f"# ClusterNumber\t=\t{len(stat_list)}\n"
        # row 21
        _, _, average_elapsed_time = line_calculate([f[6] for f in stat_list], method="average")
        stat[21] = f"# AverageElapsedTime\t=\t{average_elapsed_time}\n"
        # row 22
        _, _, elapsed_time = line_calculate([f[6] for f in stat_list], method="sum")
        stat[22] = f"# ElapsedTime (H:M:S)\t=\t{seconds_to_HMS(elapsed_time)}\n"
        # row 23
        stat[23] = f"# AverageElapsedTime (H:M:S)\t=\t{seconds_to_HMS(average_elapsed_time)}\n"

        return stat

    @staticmethod
    def DoseByRegion(fpath_list: List[str], stat_path_list: List[str]) -> List[str]:
        analyzer_list = [DoseByRegionAnalyzer(stat) for stat in fpath_list]
        N_all = np.sum([StatisticAnalyzer(stat_path).GetN() for stat_path in stat_path_list])
        out_lines = ['#id\tvol(mm3)\tedep(MeV)\tstd_edep\tsq_edep\tdose(Gy)\tstd_dose\tsq_dose\tn_hits\tn_event_hits\n']
        for i in range(len(analyzer_list[0].ID)):
            # id, vol(mm3), dep(MeV), std_edep, sq_edep, dose(Gy), std_dose, sq_dose, n_hits, n_event_hits
            ID = analyzer_list[0].ID[i]
            vol = analyzer_list[0].vol[i]

            dep = np.sum([analyzer.dep[i] for analyzer in analyzer_list])
            sq_edep = np.sum([analyzer.sq_edep[i] for analyzer in analyzer_list])
            std_edep = np.sqrt(sq_edep - dep**2/N_all) / dep

            dose = np.sum([analyzer.dose[i] for analyzer in analyzer_list])
            sq_dose = np.sum([analyzer.sq_dose[i] for analyzer in analyzer_list])
            std_dose = np.sqrt(sq_dose - dose**2/N_all) / dose

            n_hits = np.sum([analyzer.n_hits[i] for analyzer in analyzer_list])
            n_event_hits = np.sum([analyzer.n_event_hits[i] for analyzer in analyzer_list])
            line = [str(v) for v in [ID, vol, dep, std_edep, sq_edep, dose, std_dose, sq_dose, n_hits, n_event_hits]]
            line = "\t".join(line) + '\n'
            out_lines.append(line)

        return out_lines


if __name__ == "__main__":

    def t():
        os.chdir(r"E:\PETDose_dataset\Pediatric")
        for i, pname in enumerate(os.listdir()):
            if i == 0:
                print(pname)
                p = OrganDoseUncertaintyCalculator(
                    dosemap=os.path.join(pname, "GATE_output", "PET_CT", "Dose.nii"),
                    uncertainty=os.path.join(pname, "GATE_output", "PET_CT", "DoseUncertainty.nii"),
                    atlas=os.path.join(pname, "atlas.nii")
                )
                print(p.CalculateAllOrgans())


    os.chdir(r"/home/vgate/Desktop/PETDose/GATE/output_PET_CT/AnonyP1S1_PETCT19659")

    fpaths = []
    stats = []
    for cluster in os.listdir():
        fpaths.append(os.path.join(cluster, "DoseByRegion.txt"))
        stats.append(os.path.join(cluster, "Statistic.txt"))
    t = FileMerger.DoseByRegion(fpaths, stats)
    pass
