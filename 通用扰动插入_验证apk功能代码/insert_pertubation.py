"""
将通用扰动注入到原始apk
通用扰动注入方式:
①将所有扰动量全部写入一个jar文件,将jar编译成smali1文件,
②将原始apk编译成smali2文件
③从原始apk中任意选取一个自定义函数来调用扰动量的一个自定义函数,从而将所有调用关系串联起来,
避免通过静态分析被察觉
"""
#自动化插入通用扰动代码待完成
def insert_universalPer(apk_path,depress_path):
    pass