#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : GATE_Runner_online.py
    @Time       : 2022/11/2 11:19
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description: Only used to update codes, DO NOT RUN THE FILE. Copy the code to GATE.py
"""
import os
from GATE import PrepareSimulation, cluster
from tqdm import tqdm
import shutil


def prepare_simulation_for_patients(pnames, source_type, phantom_type, output, N=5E8, ages=None):
    """
    :param pnames:
    :param source_type: source type and name, "PET", "ICRP" or "sPET"
    :param phantom_type: phantom type and name, "CT", "Atlas" or "Registration"
    :param output: output folder name
    :param N: Number of particles
    :param ages:
    :return:
    """

    if pnames == "all":
        pnames = [name for name in os.listdir("data") if name != "utils"]
    elif pnames == "left":
        pnames = [name for name in os.listdir("data") if (name != "utils") and (name not in os.listdir(output))]
    else:
        assert isinstance(pnames, list)

    if ages is None:
        ages = [None] * len(pnames)

    for pname, age in zip(pnames, ages):
        PrepareSimulation.prepare_simulation(
            pname=pname, source_type=source_type, phantom_type=phantom_type, output=output, N=N, age=age
        )


def copy_from_Linux_to_SSD(output, base_folder, folder_to):
    for pname in tqdm(os.listdir(base_folder)):
        if not os.path.exists(os.path.join(folder_to, pname, output)):
            os.makedirs(os.path.join(folder_to, pname, output))
        for file in os.listdir(os.path.join(base_folder, pname)):
            if os.path.isfile(os.path.join(base_folder, pname, file)):
                shutil.copyfile(src=os.path.join(base_folder, pname, file),
                                dst=os.path.join(folder_to, pname, output, file))


if __name__ == "__main__":
    os.chdir("GATE")
    output = "output_" + " "

    # Get all the patients in data folder
    pnames = [name for name in os.listdir("data") if name != "utils"]

    # ==================================================================================================================
    # PrepareSimulation
    # ==================================================================================================================
    prepare_simulation_for_patients(pnames=pnames, N=5E8, ages=None, output=output,
                                    source_type="PET", phantom_type="CT")

    for organ in ["Heart", "Lung", "Brain", "Kidney", "Liver", "Others"]:
        prepare_simulation_for_patients(pnames=pnames, N=5E8, ages=None, output=output+"_"+organ,
                                        source_type="SourceOrgan"+organ, phantom_type="CT")

    # ==================================================================================================================
    # Cluster
    # ==================================================================================================================

    # Split mac (do not need to call directly, called in run_cluster())
    # nb_parallel = cluster.auto_cluster_parallel(pname=pname)
    # cluster.split_mac(origin_file_path=os.path.join("mac", pname+".mac"), parallel=nb_parallel)

    # Start Simulation! Comment other codes and Call GATE.py from cmd
    cluster.run_clusters(mac_names="all", output=output)

    # ==================================================================================================================
    # Supervision
    # ==================================================================================================================
    pname = "DE_DOMPIERRE_DANIEL_FRANCOIS_356081"
    cluster.supervision(output=output, pname=pname)

    # ==================================================================================================================
    # Merge Output files
    # ==================================================================================================================
    for pname in tqdm(pnames):
        cluster.ClusterMerger.merge_output(folder_cluster=os.path.join(output, pname), remerge=False)

    # ==================================================================================================================
    # Copy output file to SSD
    # ==================================================================================================================
    copy_from_Linux_to_SSD(output, base_folder=" ",
                           folder_to="/media/pc/JHR-SSD/Data/PETDoseAdults_output")
