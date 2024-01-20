#!/usr/bin/env python
# encoding: utf-8
"""
    @File       : PrepareSimulation.py
    @Time       : 2022/8/5 9:14
    @Author     : Haoran Jia
    @license    : Copyright(c) 2022 Haoran Jia. All rights reserved.
    @contact    : 21211140001@m.fudan.edu.cn
    @Description：
"""
import os; os.chdir("D:/MP/PSDoseCalculator")
import SimpleITK as sitk
import nibabel as nib
import numpy as np

# import GATEPythonScript as gps
# from GATEPythonScript import GateFileGenerate
# from ImageProcess.Image import ImageProcessor, AtlasProcessor
# from utils.ICRPReference import F18_bladder_cumulate_activity
# from utils import OrganDict


def nii_to_hdr(fpath_nii, fpath_hdr):
    # 使用nibabel将nii转换为hdr，以便GATE读取
    assert os.path.exists(fpath_nii)
    if not os.path.exists(fpath_hdr):
        img_nib = nib.load(fpath_nii)
        nib.save(img_nib, fpath_hdr)

# ======================================================================================================================
# Data File Generate
# ======================================================================================================================

def ICRP_F18PET_source(fpath_atlas, fpath_save, age):
    """
    根据ICRP128号报告的器官累计活度，生成对应的活度源
    :param age: 患者的年龄
    :param fpath_save: 保存nii文件的路径
    :param fpath_atlas: 分割文件的路径
    :return:
    """
    # 读取分割
    seg: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(fpath_atlas))

    # 指定剂量器官ID，填补其中空缺
    ID_specific = [18, 26, 33, 32, 15]
    for ID in ID_specific:
        if ID in OrganDict.MultipleOrgans:
            for ID_sub in OrganDict.MultipleOrgans[ID]:
                seg[seg == ID_sub] = ID

    # 将其他器官均设为body 10
    ID_ignore = [x for x in OrganDict.OrganID if (x not in ID_specific and x != 10)]
    for ID in ID_ignore:
        seg[seg == ID] = 10

    # 根据年龄获取膀胱的ICRP值
    bladder_activity = F18_bladder_cumulate_activity(age)

    # source为生成的源
    source = np.zeros_like(seg, dtype=float)
    # 赋值
    for ID, ICRP_value in zip([18, 26, 33, 32, 15, 10], [0.21, 0.11, 0.079, 0.13, bladder_activity, 1.7]):
        mask = seg.copy()
        mask[mask != ID] = 0
        mask[mask == ID] = 1
        if mask.sum() == 0:
            print(f"{fpath_atlas} misses organ {ID}")
            source[seg == ID] = 0
        else:
            source[seg == ID] = ICRP_value * 10E6 / mask.sum()

    # 保存图像文件
    source = sitk.GetImageFromArray(source)
    source.CopyInformation(sitk.ReadImage(fpath_atlas))
    source = sitk.Cast(source, sitk.sitkFloat32)
    sitk.WriteImage(source, fpath_save)

    # 转换为hdr
    source_nib = nib.load(fpath_save)
    nib.save(source_nib, fpath_save[:-4] + ".hdr")

    return 0


def organ_source(organ, fpath_atlas, fpath_pet, fpath_save):
    """
    Create a single organ source, which contains the same inner voxel scale as the PET image.
    :param organ: the organ name, supporting: "Heart", "Lung", "Brain", "Kidney", "Liver", "Others"
    :param fpath_atlas:
    :param fpath_pet:
    :param fpath_save:
    :return:
    """
    OrganNameDict = OrganDict.OrganName
    if organ == "Others":
        # Others is the body left, which contains only a few hollow organs,
        # which contains here, "Heart", "Lung", "Brain", "Kidney" and "Liver"

        # whole_body = AtlasProcessor.GenerateMaskedArray(
        #     img=fpath_pet, mask=AtlasProcessor.GenerateOrganMask(atlas=fpath_atlas, ID=10))
        whole_body = ImageProcessor.ReadImageAsArray(fpath_pet)
        for hollow_organ in ["Heart", "Lung", "Brain", "Kidney", "Liver"]:
            hollow_mask = AtlasProcessor.GenerateOrganMask(atlas=fpath_atlas, ID=OrganNameDict[hollow_organ])
            whole_body[hollow_mask == 1] = 0

        source = sitk.GetImageFromArray(whole_body)
        source.CopyInformation(ImageProcessor.ReadImageAsImage(fpath_pet))

    else:
        assert organ in OrganNameDict, "Unsupported Organ."
        source = AtlasProcessor.GenerateMaskedImage(
            img=fpath_pet, mask=AtlasProcessor.GenerateOrganMask(atlas=fpath_atlas, ID=OrganNameDict[organ]))

    source = sitk.Cast(source, sitk.sitkFloat32)
    sitk.WriteImage(source, fpath_save)

    source_nib = nib.load(fpath_save)
    nib.save(source_nib, fpath_save[:-4] + ".hdr")


# ======================================================================================================================
# Prepare Simulation
# ======================================================================================================================
def prepare_simulation(pname, source_type, phantom_type, output, N=5E8, **kwargs):
    """
    function to generate relative Data Files and create .mac macro
    :param pname: name of the patient
    :param source_type: source type and name, "PET", "ICRP", "SourceOrganXXX"
    :param phantom_type: phantom type and name, "CT", "Atlas", "Registration", "ICRP"
    :param output: output folder name
    :param N: Number of particles
    :return:
    """
    # main folders and paths
    mac_name = pname
    folder_data = os.path.join("data", pname)
    folder_output = os.path.join(output, pname)
    fpath_mac = os.path.join("mac", pname + ".mac")
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)

    def nii_to_hdr(fname):
        assert os.path.exists(os.path.join(folder_data, fname + ".nii"))
        if not os.path.exists(os.path.join(folder_data, fname + ".hdr")):
            img_nib = nib.load(os.path.join(folder_data, fname + ".nii"))
            nib.save(img_nib, os.path.join(folder_data, fname + ".hdr"))

    # 0. Creating Atlas (DoseByRegions):
    nii_to_hdr("Atlas")

    # 1. Creating Source:
    if source_type == "PET":
        # Directly use PET image as source input, which uses the statistic distribution as cumulated activities.
        nii_to_hdr("PET")
    elif source_type == "ICRP":
        # Use ICRP128 cumulated activities as source input
        if not os.path.exists(os.path.join(folder_data, "ICRP.nii")):
            if "age" not in kwargs:
                kwargs["age"] = 18
            ICRP_F18PET_source(fpath_atlas=os.path.join(folder_data, phantom_type + ".nii"),
                               fpath_save=os.path.join(folder_data, "ICRP.nii"), age=kwargs["age"])
    elif source_type[0:11] == "SourceOrgan":
        # Create a source input that only contains one organ, which is used to calculate S-Values.
        # The inner distribution stays unchanged according to the PET image.
        mac_name = pname + "_" + source_type[11:]
        fpath_mac = os.path.join("mac", mac_name + ".mac")
        folder_output = os.path.join(output, mac_name)
        if not os.path.exists(folder_output):
            os.makedirs(folder_output)
        if not os.path.exists(os.path.join(folder_data, source_type+".hdr")):
            organ_source(
                organ=source_type[11:],
                fpath_atlas=os.path.join(folder_data, "Atlas.nii"),
                fpath_pet=os.path.join(folder_data, "PET.nii"),
                fpath_save=os.path.join(folder_data, source_type+".nii")
            )
    else:
        raise KeyError("Unsupported source type")

    # 2. Creating phantom
    assert phantom_type in ["CT", "Atlas", "Registration", "ICRP"], "Unsupported phantom type"
    nii_to_hdr(phantom_type)

    # 3. Creating .mac
    source = sitk.ReadImage(os.path.join(folder_data, source_type + ".nii"))
    phantom = sitk.ReadImage(os.path.join(folder_data, phantom_type + ".nii"))
    if phantom_type == "CT":
        GateFileGenerate.create_mac_PETLike_CTLike(
            fpath=fpath_mac, pname=pname, mac_name=mac_name,
            phantom=phantom, source=source,
            phantom_name=phantom_type, source_name=source_type,
            N=N, output=output
        )
    elif phantom_type in ["Atlas", "Registration", "ICRP"]:
        GateFileGenerate.create_mac_PETLike_AtlasLike(
            fpath=fpath_mac, pname=pname, mac_name=mac_name,
            phantom=phantom, source=source,
            phantom_name=phantom_type, source_name=source_type,
            N=N, output=output
        )
    else:
        raise KeyError("Unsupported phantom type")


if __name__ == "__main__":
    os.makedirs("GATEPythonScript/data/case0", exist_ok=True)
    # nii_to_hdr("temp/case0/atlas.nii", "GATEPythonScript/data/case0/atlas.hdr")
    nii_to_hdr("temp/case0/CT.nii", "GATEPythonScript/data/case0/CT.hdr")
    nii_to_hdr("temp/case0/PET.nii", "GATEPythonScript/data/case0/PET.hdr")