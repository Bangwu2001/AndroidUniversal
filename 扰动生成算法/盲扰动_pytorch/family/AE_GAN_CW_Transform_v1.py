"""
利用生成网络生成对抗扰动
网络说明:
    1.输入:
        当前恶意样本
    2.损失函数:
        ①对抗样本损失,采用HingLoss
        ②对抗扰动量损失,限制插入扰动的比例
    3.仍然采用本地替代模型的方式,本地替代模型仍然选用CNN

    4.此过程中设计到的模型

        ①扰动生成网络,用于生成插入的调用次数扰动量
        ②本地替代模型，对抗损失反馈


网络结构设计:
    扰动生成网络GModel:
        新的良性样本与当前恶意样本拼接输入扰动生成网络，输入[N,2,11,11]

        生成的扰动为[N,1,11,11]

    本地替代模型:
        由Markov转移概率矩阵特征事先训练好的本地替代模型,
        输入[N,1,11,11],输出[N,1],为概率值,大于等于0.5:良性;小于0.5:恶意
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle



def StandardScale(data):
    for index in range(data.shape[0]):
        for row in range(data.shape[2]):
            sumer = torch.sum(data[index, :, row, :])
            if sumer != 0:
                data[index, :, row, :] /= sumer
    return data

def StandardScaleNotInplace(x):
    data=x.clone()
    for index in range(data.shape[0]):
        for row in range(data.shape[2]):
            sumer = torch.sum(data[index, :, row, :])
            if sumer != 0:
                data[index, :, row, :] /= sumer
    return data





class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.layer1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1)
        self.layer2=nn.GELU()
        self.layer3=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1)
        self.layer4=nn.GELU()
        self.layer5=nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=3,stride=1)
        self.layer6=nn.GELU()
        self.layer7=nn.ConvTranspose2d(in_channels=4,out_channels=1,kernel_size=3,stride=1)
        self.layer8=nn.ReLU()  #保证全为正

    def forward(self,inx):
        x=self.layer2(self.layer1(inx))
        x=self.layer4(self.layer3(x))
        x=self.layer6(self.layer5(x))
        x=self.layer8(self.layer7(x))

        return x



#模型训练
def train():

    #掩膜和噪声
    mask = torch.cat((torch.zeros(1, 1, 9, 11), torch.ones(1, 1, 2, 11)), dim=2)
    mask[:, :, -2:, -5:-2] = 0
    noise = torch.zeros(size=(11, 11))
    noise[9, 9] = 1
    noise[9, 4] = 2
    noise[10, 4] = 1
    noise[9, 10] = 1
    #1.加载数据

    #加载训练模型的恶意样本数据和测试数据集
    with open("dataset/trainG_data","rb") as f:
        trainG_data=pickle.load(f)
    trainG_data=torch.from_numpy(trainG_data).float()

    with open("dataset/markov_evaluate_100_X","rb") as f:
        testG_data=pickle.load(f)
    testG_data=torch.from_numpy(testG_data).float()

    print(trainG_data.shape,testG_data.shape)



    #③实例化模型
    model=GModel()

    #④超参数设置
    batch_size=100
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    epochs=100
    alpha=10
    beta=50

    #⑤加载本地替代模型
    targetF=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")


    #训练
    train_acc=[]
    test_acc=[]

    for epoch in range(1,epochs+1):
        epoch_loss1 = []
        epoch_loss2 = []
        epoch_both_loss = []
        epoch_correct = 0.

        index=list(range(trainG_data.shape[0]))
        np.random.shuffle(index)
        batch_trainG_data=trainG_data[index]
        for index in range(batch_trainG_data.shape[0]//batch_size):
            x=batch_trainG_data[index*batch_size:(index+1)*batch_size]

            per=model(x)
            per=per*mask+noise
            x_adv=x+per
            x_adv_pred=targetF(StandardScale(x_adv))
            correct = torch.sum(x_adv_pred < 0.5).detach().numpy()
            epoch_correct+=correct
            loss1=torch.sum(per)/torch.sum(x)

            loss2=torch.sum(torch.clamp(0.5-x_adv_pred,min=0))/x.shape[0]

            loss = alpha * loss1 + beta * loss2

            epoch_loss1.append(loss1.detach().numpy())
            epoch_loss2.append(loss2.detach().numpy())
            epoch_both_loss.append(loss.detach().numpy())
            # 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(per[0,0])

        print("Epoch:%d || all_loss:%.3f || overHeadLoss:%.3f || advLoss:%.3f || acc:%.3f" % (
            epoch, np.mean(epoch_both_loss), np.mean(epoch_loss1), np.mean(epoch_loss2), epoch_correct / len(trainG_data)))

        train_acc.append(epoch_correct / len(trainG_data))

        #测试集
        with torch.no_grad():
            per = model(testG_data).int().float()
            per=per*mask+noise
            testG_data_adv = testG_data + per
            testG_data_adv_pred = targetF(StandardScale(testG_data_adv))
            correct = torch.sum(testG_data_adv_pred < 0.5).detach().numpy()
            per_count=(torch.sum(per)/len(testG_data))
            per_ratio=(torch.sum(per)/torch.sum(testG_data)).detach().numpy()
            print("Epoch:%d || test_acc:%.3f || per_count:%.2f || per_ratio:%.3f"%(epoch,correct/testG_data.shape[0],per_count,per_ratio))
            print(per[0,0])
            test_acc.append(correct/testG_data.shape[0])


    torch.save(model,"generator/transform_v1_10_50_100")
    import matplotlib.pyplot as plt

    plt.title("train&test acc")
    plt.plot(train_acc,label="train")
    plt.plot(test_acc,label="test")
    plt.savefig("generator_result/transform_v1_10_50_100")
    plt.show()

#对抗样本生成测试
def evaluate():
    # 掩膜和噪声
    mask = torch.cat((torch.zeros(1, 1, 9, 11), torch.ones(1, 1, 2, 11)), dim=2)
    mask[:, :, -2:, -5:-2] = 0
    noise = torch.zeros(size=(11, 11))
    noise[9, 9] = 1
    noise[9, 4] = 2
    noise[10, 4] = 1
    noise[9, 10] = 1
    #加载训练好的生成器模型
    generatorModel = torch.load("generator/transform_v1_10_50_100")

    #加载本地测试模型
    targetF=torch.load("markov_classifier/Family_targetF_CNNModel.pkl")
    #加载测试数据
    with open("dataset/markov_evaluate_1000_X", "rb") as f:
        evaluate_x = pickle.load(f)
    with open("dataset/markov_evaluate_1000_Y", "rb") as f:
        evaluate_y = pickle.load(f)

    evaluate_x=torch.from_numpy(evaluate_x).float()

    #干净样本测试
    clean_pred=targetF.forward(StandardScaleNotInplace(evaluate_x))
    clean_acc=torch.sum(clean_pred<0.5).detach().numpy()/len(evaluate_x)

    #生成扰动
    per=generatorModel.forward(evaluate_x).int().float()
    per=per*mask+noise
    evaluate_x_adv=evaluate_x+per
    adv_pred=targetF.forward(StandardScaleNotInplace(evaluate_x_adv))
    adv_acc=torch.sum(adv_pred<0.5).detach().numpy()/len(evaluate_x_adv)

    #计算平均扰动量和比例
    average_per_count=(torch.sum(per)/len(per)).detach().numpy()
    average_per_ratio=(torch.sum(per)/torch.sum(evaluate_x)).detach().numpy()
    #对抗样本存储
    with open("generator_dataset/markov_evaluate_1000_X_adv_transform_v1_10_50_100","wb") as f:
        pickle.dump(evaluate_x_adv,f)
    with open("generator_dataset/markov_evaluate_1000_Y_adv_transform_v1_10_50_100","wb") as f:
        pickle.dump(evaluate_y,f)
    evaluate_x_adv=evaluate_x_adv.numpy()
    print("干净样本准确率:",clean_acc,"对抗样本准确率:",adv_acc)
    print("平均扰动量:",average_per_count,"平均扰动率:",average_per_ratio)


if __name__=="__main__":
    # train()
    evaluate()
#     pass
    # pool=torch.randn(size=(100,1,11,11))
    # inx=torch.randn(size=(10,1,11,11))
    # model=FGModel()
    # model(pool,inx)



