"""
从smali文件中提取函数调用图,并转换为family粒度的调用次数矩阵
一下代码均只操作于单个apk,单线程
"""


import re
import os
import numpy as np
import shutil
import pickle

INVOKE_PATTERN = re.compile(
    '(?P<invoketype>invoke-(?:virtual|direct|static|super|interface)) (?:\{.*\}), (?P<method>L.+;->.+)(?:\n)')

CLASS_NAME_PATTERN = re.compile('\.class.*(?P<clsname>L.*)(?:;)')
METHOD_BLOCK_PATTERN = re.compile('\.method.* (?P<methodname>.*)\n.*\n((?:.|\n)*?)\.end method')


def get_family_caller_callee_from_function_pair(function_pair,Families_list):
    cur_pair = function_pair.split(' ')
    cur_pair[0] = cur_pair[0][1:]
    for family in Families_list:
        if cur_pair[0].startswith(family):
            cur_pair[0] = family
            break

    for family in Families_list:
        if family in cur_pair[2]:
            cur_pair[2] = family
            break
    if cur_pair[0] not in Families_list:
        splitted = cur_pair[0].split(';->')[0].split('/')
        obfcount = 0
        for k in range(0, len(splitted)):
            if len(splitted[k]) < 3:
                obfcount += 1
        if obfcount >= len(splitted) / 2:
            cur_pair[0] = 'obfuscated'
        else:
            cur_pair[0] = 'self-defined'
    caller = Families_list.index(cur_pair[0])
    if cur_pair[2] not in Families_list:
        splitted = cur_pair[2].split(';->')[0].split('/')
        obfcount = 0
        for k in range(0, len(splitted)):
            if len(splitted[k]) < 3:
                obfcount += 1
        if obfcount >= len(splitted) / 2:
            cur_pair[2] = 'obfuscated'
        else:
            cur_pair[2] = 'self-defined'
    callee = Families_list.index(cur_pair[2])
    return caller, callee



def get_family_feature_from_function_pair(all_funcs,function_calls):
    family_granularity_path="Families.txt"
    family_granularity_list=[]  #MamaDroid所使用的的系统函数的9个family名称
    for line in open(family_granularity_path,"r",encoding="utf-8"):
        family_granularity_list.append(line.strip().replace(".","/"))
    family_granularity_list.append("self-defined")
    family_granularity_list.append("obfuscated")
    print(family_granularity_list)

    #构建调用数目矩阵
    family_call_times=np.zeros(shape=(len(family_granularity_list),len(family_granularity_list)))
    #将各个函数调用对与family粒度对应起来
    for i_pair in range(len(function_calls)):
        caller = int(function_calls[i_pair].split(' ')[0])
        callee = int(function_calls[i_pair].split(' ')[2])
        # 将函数调用列表的索引形式转化为具体的函数形式
        function_calls[i_pair] = all_funcs[caller] + ' ' + function_calls[i_pair].split(' ')[1] + ' ' + all_funcs[
            callee]
    # with open("func_Call_add_xml_v15","wb") as f:
    #     pickle.dump(function_calls,f)

    family_call_prob = np.zeros((len(family_granularity_list), len(family_granularity_list)))
    for pair in function_calls:
        caller, callee = get_family_caller_callee_from_function_pair(pair,family_granularity_list)
        family_call_prob[caller][callee] += 1
    return family_call_prob


def extract_Mama_family_feature_for_single_apk(apk_path,output_path):
    """
    借助apktool将apk逆编译成samli文件,
    从中提取函数调用图,
    转换成family粒度的系统调用
    :param apk_path: 原始apk路径
    :param out_smali_path: 逆变异后的文件路径名
    :return: family粒度的特征
    """

    shutil.rmtree(output_path,True)
    os.system("apktool d %s -o %s -f"%(apk_path,output_path))
    out_smali_path=os.path.join(output_path,"smali")

    all_funcs, func_calls=get_all_funcs_and_func_call_from_smali_folder_path1(out_smali_path)
    family_call_times=get_family_feature_from_function_pair(all_funcs,func_calls)
    return family_call_times


def get_all_funcs_and_func_call_from_smali_folder_path1(smali_folder_path):
    '''
    从smali文件夹中提取所有的函数列表和函数调用对
    :param smali_folder_path: smali文件夹路径
    :output all_funcs:所有的函数名
    :output func_calls:函数调用对
    '''
    all_funcs = []
    func_calls = []
    smali_file_path_all = []
    print(smali_folder_path)
    for root, dirs, files in os.walk(smali_folder_path):  # os.walk()文件目录遍历器,递归遍历所有文件
        for file in files:
            smali_file_path_all.append(os.path.join(root, file))
    for smali_file_path in smali_file_path_all:
        try:
            with open(smali_file_path, 'r', encoding='utf-8') as f:
                # print(smali_file_path)
                s = f.read()
                # 获取class名
                class_name_match = CLASS_NAME_PATTERN.search(s)
                class_name = class_name_match.group(
                    'clsname') if class_name_match is not None else ''
                # 获取function名
                for method_block_match in METHOD_BLOCK_PATTERN.finditer(s):
                    method_name = method_block_match.group('methodname')
                    for invoke_match in INVOKE_PATTERN.finditer(method_block_match.group()):
                        cur_pair = class_name + ';->' + method_name + ' ' + \
                                   invoke_match.group('invoketype') + \
                                   ' ' + invoke_match.group('method')

                        if cur_pair.split(' ')[0] not in all_funcs:
                            all_funcs.append(cur_pair.split(' ')[0])
                        if cur_pair.split(' ')[2] not in all_funcs:
                            all_funcs.append(cur_pair.split(' ')[2])
                        cur_pair_new = '%d %s %d' % (all_funcs.index(cur_pair.split(' ')[0]), cur_pair.split(' ')[1],
                                                     all_funcs.index(cur_pair.split(' ')[2]))
                        func_calls.append(cur_pair_new)
                        # print(cur_pair_new)
        except Exception as e:
            with open("\\\?\\" + smali_file_path, 'r', encoding='utf-8') as f:
                # print(smali_file_path)
                s = f.read()
                # 获取class名
                class_name_match = CLASS_NAME_PATTERN.search(s)
                class_name = class_name_match.group(
                    'clsname') if class_name_match is not None else ''
                # 获取function名
                for method_block_match in METHOD_BLOCK_PATTERN.finditer(s):
                    method_name = method_block_match.group('methodname')
                    for invoke_match in INVOKE_PATTERN.finditer(method_block_match.group()):
                        cur_pair = class_name + ';->' + method_name + ' ' + \
                                   invoke_match.group('invoketype') + \
                                   ' ' + invoke_match.group('method')

                        if cur_pair.split(' ')[0] not in all_funcs:
                            all_funcs.append(cur_pair.split(' ')[0])
                        if cur_pair.split(' ')[2] not in all_funcs:
                            all_funcs.append(cur_pair.split(' ')[2])
                        cur_pair_new = '%d %s %d' % (all_funcs.index(cur_pair.split(' ')[0]), cur_pair.split(' ')[1],
                                                     all_funcs.index(cur_pair.split(' ')[2]))
                        func_calls.append(cur_pair_new)
    return all_funcs, func_calls






if __name__=="__main__":

    #原始apk
    result1 = extract_Mama_family_feature_for_single_apk("apk_data/malware_ori.apk", "apk_data/depress")

    #添加扰动后apk
    result2 = extract_Mama_family_feature_for_single_apk("apk_data/malware_release_v1.apk", "apk_data/depress")

    #插入的扰动量扰动量
    print(result2-result1)


