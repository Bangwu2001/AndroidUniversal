- `MamaData`：提取好特征的数据集，后续试验的数据都是在此基础上二次处理后的样本
  - `dataset_attack_MaMa`:4064条恶意样本
  - `dataset_target_MaMa`:40000条数据，20000良性，20000恶意
- `family`:主要算法模块，包含实验过程中所有代码和数据，大多数数据文件可能无用
  - 主要文件如下
    - `GCNModel`:包含GCN模型构建和训练、测试代码
    - `generator`:存储生成器权重文件，前期试验的已删除
    - `iter_dataset`:利用优化方法生成扰动相关数据
    - `markov_classifier`:分类模型权重
    - `markov_result`:各个分类模型训练结果
    - `targetF_result`:本地替代模型训练结果
    - `dataset`:试验所用数据
      - `targetF_data`,`targetF_label`:用于训练本地替代模型和分类模型的数据,10000条良性+10000条恶意
      - `trainG_data,trainG_label`:用于训练GAN的数据，10000条恶意样本
      - `benginG_data,benginG_label`：删选出来的良性样本，10000条
      - `evaluate_100_x,evaluate_100_y`:100个恶意样本
      - `evaluate_1000_x,evaluate_1000_y`:1000个恶意样本，用于测试扰动有效性，这1000个样本均能够被本地替代模型正确分类
      - `trainGCN_Family`:训练GCN模型的数据
    - 分类模型训练代码+测试代码
      - `targetF_Model、targetF_train`:本地替代模型
      - `Adaboost`
      - `CNN`
      - `DNN`
      - `KNN1`
      - `KNN3`
      - `RF`
      - `本地替代模型评估`:测试对抗样本对本地替代模型效果
    - 生成模型生成扰动代码
      - `AE_GAN_MAKOV_CW.py`:损失函数改为`hingLoss`
      - `AE_GAN_MARKOV.py`:损失函数为交叉熵
    - `使用迭代方法生成对抗样本_v5.py`:利用优化方法生成通用扰动代码
    - `迭代方法对抗样本存储`:根据迭代算法生成的通用扰动，生成对应的测试对抗样本
    - `data_process.py,样本选择.py`：对原始数据二次删选的代码