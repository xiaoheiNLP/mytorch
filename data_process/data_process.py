# -*- coding:utf-8 -*-
"""
@author: xinquan
@file: data_process.py
@time: 2021/7/23 12:09
@desc: 
"""
import os
import warnings

warnings.filterwarnings('ignore')  # 忽略一些警告,可以删除
root_path = os.path.split(os.path.realpath(__file__))[0]  # 获取该脚本的地址,有效避免Linux和Windows文件路径格式不一致等问题,可以删除


def step_merge_file():
    import glob
    from tqdm import tqdm
    all_file_path = glob.glob(r"D:\资料\模型数据\THUCNews\*")
    for one_file_path in tqdm(all_file_path, desc='Processing'):
        title_name = one_file_path.split("\\")[-1]
        # print(title_name)
        fwrite = open(os.path.join(r"D:\资料\模型数据\THUCNews", title_name + ".txt"), 'w')

        txt_path = glob.glob(os.path.join(one_file_path, '*.txt'))
        # amout_txt = len(txt_path)
        # count_num = 0
        """处理数据"""
        # for one_txt_path in tqdm(txt_path, desc=title_name):
        for one_txt_path in txt_path:
            # print(title_name, "{}/{}".format(count_num, amout_txt), "{}%".format(round(float(count_num/amout_txt)*100, 3)))
            with open(one_txt_path, 'r', encoding='utf-8') as f:
                f_readlines = f.readlines()
                f_readlines = list(map(lambda x: str(x).strip().replace(" ", ""), f_readlines))
                f_readlines = ''.join(f_readlines)
                f_readlines = list(filter(lambda x: '\u4e00' <= x <= '\u9fa5', f_readlines))
                f_readlines = ''.join(f_readlines)
                f_readlines = f_readlines.strip()
                try:
                    if len(f_readlines.strip().replace(' ', '')) > 0:
                        fwrite.write(f_readlines + '\n')
                except Exception as e:
                    print(e)
                    exit(e)
                fwrite.flush()
            # count_num += 1
        # exit("1111")


def step_merge_txt():
    import glob
    path_txt_list = glob.glob(r"D:\资料\模型数据\THUCNews\*.txt")
    fwrite = open('THUCNews.txt', 'w', encoding='utf-8')
    for one_txt_path in path_txt_list:
        print(one_txt_path)
        one_txt_open = open(one_txt_path, 'r')
        f_readlines = one_txt_open.readlines()
        amount_len = list(filter(lambda x: len(x.strip().replace(" ", "")) > 0, f_readlines))
        print(len(f_readlines), len(amount_len))
        fwrite.writelines(amount_len)
        fwrite.flush()
        # exit("111111")


if __name__ == '__main__':
    step_merge_file()
    # step_merge_txt()


