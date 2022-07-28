"""
利用生成网络生成对抗扰动
网络说明:
    改动7_25:本地模型采用集成,CNN1+CNN2的平均
    1.输入:
        良性样本+当前恶意样本+事先计算好的通用对抗扰动
        ①事先选择100个良性样本构建样本池
        ②构建一个特征提取网络,对良性样本和当前恶意样本进行特征提取,得到qb1,qb2,...gm
        ③根据②中得到的特征，计算当前恶意样本与池子中的良性样本的相关性，按照相关性比例合成一个新的样本
        ④将③中合成的新的样本和当前恶意样本进行拼接，输入到生成网络
    2.损失函数:
        ①对抗样本损失,采用HingLoss
        ②对抗扰动量损失,限制插入扰动的比例
    3.仍然采用本地替代模型的方式,本地替代模型仍然选用CNN

    4.此过程中设计到的模型
        ①良性和恶意样本的特征提取网络,用于计算特征，以此来计算相似度
        ②扰动生成网络,用于生成插入的调用次数扰动量
        ③本地替代模型，对抗损失反馈


网络结构设计:
    特征提取网络FModel:
        输入:[N,1,11,11]-flatten->[N,121]-Linear->[N,64],->[N,16]
        利用内积计算相似度,再经过Softmax得到各个样本概率,依据各个样本概率合成新的良性样本
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
import argparse

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1)
        self.layer2=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1)
        self.layer3=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1)
        self.layer4=nn.Conv2d(in_channels=16,out_channels=8,kernel_size=3,stride=1)
        self.layer5=nn.Conv2d(in_channels=8,out_channels=4,kernel_size=3,stride=1)
        self.layer6=nn.Flatten()
        self.layer7=nn.Linear(4,1)

    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        x=F.relu(self.layer4(x))
        x=F.relu(self.layer5(x))
        x=self.layer6(x)
        x=self.layer7(x)
        out=torch.sigmoid(x)
        return out

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


class Reshape(nn.Module):
    def __init__(self,*args):
        super(Reshape, self).__init__()
        self.shape=args

    def forward(self,x):
        return x.view((x.size(0),)+self.shape)


class FModel(nn.Module):
    def __init__(self):
        super(FModel, self).__init__()
        self.layer1=nn.Flatten()
        self.layer2=nn.Linear(121,64)
        self.layer3=nn.GELU()
        self.layer4=nn.Linear(64,16)


    def forward(self,inx):
        x=self.layer1(inx)
        x=self.layer3(self.layer2(x))
        x=self.layer4(x)
        return x


class GModel(nn.Module):
    def __init__(self):
        super(GModel, self).__init__()
        self.layer1=nn.Conv2d(in_channels=2,out_channels=4,kernel_size=3,stride=1)
        self.layer2=nn.GELU()
        self.layer3=nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,stride=1)
        self.layer4=nn.GELU()
        self.layer5=nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=3,stride=1)
        self.layer6=nn.GELU()
        self.layer7=nn.ConvTranspose2d(in_channels=4,out_channels=1,kernel_size=3,stride=1)

        self.layer8=nn.Tanh()

    def forward(self,inx):
        x=self.layer2(self.layer1(inx))
        x=self.layer4(self.layer3(x))
        x=self.layer6(self.layer5(x))
        x=self.layer7(x)
        x = self.layer8(x)
        return x


class FGModel(nn.Module):
    def __init__(self):
        super(FGModel, self).__init__()
        self.fmodel=FModel()
        self.attend=nn.Softmax(dim=-1)
        self.scale=16.**-0.5
        self.reshape1=nn.Flatten()
        self.reshape2=Reshape(1,11,11)
        self.gmodel=GModel()

        # 通用对抗扰动
        with open("iter_dataset/markov_universal_1000_trainLoop2000_1_1000_selfDefined_v3", "rb") as f:
            self.universal_per = pickle.load(f)


    def forward(self,benginPoolSample,currentMalwareSample):
        benginPoolSampleQ=self.fmodel(benginPoolSample)
        currentMalwareSampleQ=self.fmodel(currentMalwareSample)


        dots=torch.matmul(currentMalwareSampleQ,benginPoolSampleQ.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        # print(attn.shape)

        benginPoolSampleFaltten=self.reshape1(benginPoolSample)
        # print(benginPoolSampleFaltten.shape)

        newBenginSample=torch.matmul(attn,benginPoolSampleFaltten)
        # print(newBenginSample.shape)

        newBenginSample=self.reshape2(newBenginSample)
        # print(newBenginSample.shape)

        newInputGSample=torch.cat((currentMalwareSample,newBenginSample),dim=1)
        # print(newInputGSample.shape)

        per=self.gmodel(newInputGSample)


        return per



class Attack():
    def __init__(self,args):
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.alpha = args.alpha
        self.beta = args.beta
        # get mask
        self.mask = torch.cat((torch.zeros(1, 1, 9, 11), torch.ones(1, 1, 2, 11)), dim=2)
        self.mask[:, :, -2:, -5:-2] = 0
        #get noise
        self.noise = torch.zeros(size=(11, 11))
        self.noise[9, 9] = 1
        self.noise[9, 4] = 2
        self.noise[10, 4] = 1
        self.noise[9, 10] = 1
        #get file
        self.up_file = args.up_file
        self.beni_file = 'dataset/benginSamplePool_100'
        self.mali_train_file = 'dataset/trainG_data'
        self.mali_test_file = 'dataset/markov_evaluate_100_X'
        self.markov_evaluate_X_file = 'dataset/markov_evaluate_1000_X'
        self.markov_evaluate_Y_file = 'dataset/markov_evaluate_1000_Y'
        self.mali_test_file = 'dataset/markov_evaluate_100_X'
        self.target1_file = 'markov_classifier/cnn.pkl'
        self.target2_file = 'markov_classifier/cnn.pkl'
        self.output_file = args.output_file
        # get up
        with open(self.up_file, "rb") as f:
            self.universal_per = pickle.load(f)

    def get_round_with_gradient(self,x):
        return torch.round(x) - x.detach() + x

    def get_adv(self,original, GP):
        per_fanal = ( torch.clamp(self.universal_per + GP, min=0.)) * self.mask
        per_fanal = torch.maximum(per_fanal, self.noise)
        x_adv = original + per_fanal
        return x_adv,per_fanal
#模型训练
    def train(self):

        #1.加载数据
        #①加载良性样本池
        with open(self.beni_file,"rb") as f:
            benginData=pickle.load(f)
        benginData=torch.from_numpy(benginData).float()

        #②加载训练模型的恶意样本数据和测试数据集
        with open(self.mali_train_file,"rb") as f:
            trainG_data=pickle.load(f)
        trainG_data=torch.from_numpy(trainG_data).float()

        with open(self.mali_test_file,"rb") as f:
            testG_data=pickle.load(f)
        testG_data=torch.from_numpy(testG_data).float()

        print(benginData.shape,trainG_data.shape,testG_data.shape)
        #③实例化模型
        model=FGModel()

        #④超参数设置
        optimizer=torch.optim.Adam(model.parameters(),lr=self.lr)

        #⑤加载本地替代模型
        targetF=torch.load(self.target1_file)

        #加载另一CNN模型
        CNNModel=torch.load(self.target1_file)

        #训练
        train_acc=[]
        test_acc=[]

        for epoch in range(1,self.epochs+1):
            epoch_loss1 = []
            epoch_loss2 = []
            epoch_both_loss = []
            epoch_correct = 0.
            epoch_correct_2 = 0.

            index=list(range(trainG_data.shape[0]))
            np.random.shuffle(index)
            batch_trainG_data=trainG_data[index]
            for index in range(batch_trainG_data.shape[0]//self.batch_size):
                x=batch_trainG_data[index*self.batch_size:(index+1)*self.batch_size]

                probability=model(benginData,x+self.universal_per)   ##### add universal perunation before the generator
                random = torch.rand(1)
                per = self.get_round_with_gradient((torch.abs(probability) > random).type(torch.float32) * probability)



                # per = torch.clamp(per,min=0)
                x_adv,per_fianl = self.get_adv(x, per)

                x_adv_markov=StandardScale(x_adv)
                x_adv_pred=targetF(x_adv_markov)
                x_adv_pred2=CNNModel(x_adv_markov)
                correct = torch.sum(x_adv_pred < 0.5).detach().numpy()
                correct2= torch.sum(x_adv_pred2 < 0.5).detach().numpy()
                epoch_correct+=correct
                epoch_correct_2+=correct2
                # loss1=torch.sum(per_fianl)/torch.sum(x)
                # print(per.shape)
                # print(torch.sum(per,axis=(1,2,3)).shape)
                # input()
                # loss1 = torch.sum(torch.clamp( torch.sum(torch.abs(per),axis=(1,2,3)) - 30,min=0))

                loss1 = torch.sum(torch.abs(per) )/ torch.sum(x)
                loss2=0.5*torch.sum(torch.clamp(1-x_adv_pred,min=0))/x.shape[0]+0.5*torch.sum(torch.clamp(1-x_adv_pred2,min=0))/x.shape[0]

                loss = self.alpha * loss1 + self.beta * loss2

                epoch_loss1.append(loss1.detach().numpy())
                epoch_loss2.append(loss2.detach().numpy())
                epoch_both_loss.append(loss.detach().numpy())
                # 参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(per[0,0])

            print("Epoch:%d || all_loss:%.3f || overHeadLoss:%.3f || advLoss:%.3f || acc:%.3f || acc2:%.3f" % (
                epoch, np.mean(epoch_both_loss), np.mean(epoch_loss1), np.mean(epoch_loss2), epoch_correct / len(trainG_data),epoch_correct_2 / len(trainG_data)))

            train_acc.append(epoch_correct / len(trainG_data))

            #测试集
            with torch.no_grad():
                probability = model(benginData, testG_data+self.universal_per).int().float()
                random = torch.tensor(0.5)
                per = self.get_round_with_gradient((torch.abs(probability) > random).type(torch.float32) * probability)


                testG_data_adv,per_fianl = self.get_adv(testG_data, per)
                # print(per[0])

                testG_data_adv_pred = targetF(StandardScale(testG_data_adv))
                correct = torch.sum(testG_data_adv_pred < 0.5).detach().numpy()
                # print(per_fi?
                per_count=(torch.sum(per_fianl)/len(per_fianl))
                per_ratio=(torch.sum(per_fianl)/torch.sum(testG_data)).detach().numpy()
                print("Epoch:%d || test_acc:%.3f || per_count:%.2f || per_ratio:%.3f"%(epoch,correct/testG_data.shape[0],per_count,per_ratio))
                # print(per[0,0])
                test_acc.append(correct/testG_data.shape[0])


        torch.save(model,self.output_file+"/transform_v5_10_50_100_0.5_0.5")
        import matplotlib.pyplot as plt

        plt.title("train&test acc")
        plt.plot(train_acc,label="train")
        plt.plot(test_acc,label="test")
        plt.savefig(self.output_file+"/transform_v5_10_50_100_0.5_0.5.jpg")
        plt.show()

#对抗样本生成测试
    def evaluate(self):
        # 加载良性样本池
        with open(self.beni_file, "rb") as f:
            benginData = pickle.load(f)
        benginData = torch.from_numpy(benginData).float()

        #加载训练好的生成器模型
        generatorModel = torch.load(self.output_file+"/transform_v5_10_50_100_0.5_0.5")

        #加载本地测试模型
        targetF=torch.load(self.target1_file)

        #加载测试数据
        with open(self.markov_evaluate_X_file, "rb") as f:
            evaluate_x = pickle.load(f)
        with open(self.markov_evaluate_Y_file, "rb") as f:
            evaluate_y = pickle.load(f)

        evaluate_x=torch.from_numpy(evaluate_x).float()

        #干净样本测试
        clean_pred=targetF.forward(StandardScaleNotInplace(evaluate_x))
        clean_acc=torch.sum(clean_pred<0.5).detach().numpy()/len(evaluate_x)

        #生成扰动
        probability=generatorModel.forward(benginData,evaluate_x+self.universal_per).int().float()
        random = torch.tensor(0.5)
        per = self.get_round_with_gradient((torch.abs(probability) > random).type(torch.float32) * probability)
        evaluate_x_adv,per_fianl = self.get_adv(evaluate_x, per)

        for i in range(2):
            probability = generatorModel.forward(benginData, evaluate_x_adv + self.universal_per).int().float()
            random = torch.tensor(0.5)
            per = self.get_round_with_gradient((torch.abs(probability) > random).type(torch.float32) * probability)
            evaluate_x_adv, per_fianl = self.get_adv(evaluate_x_adv, per)

        adv_pred=targetF.forward(StandardScaleNotInplace(evaluate_x_adv))
        adv_acc=torch.sum(adv_pred<0.5).detach().numpy()/len(evaluate_x_adv)

        #计算平均扰动量和比例
        average_per_count=(torch.sum(per_fianl)/len(per_fianl)).detach().numpy()
        average_per_ratio=(torch.sum(per_fianl)/torch.sum(evaluate_x)).detach().numpy()

        #对抗样本存储
        with open(self.output_file+"/markov_evaluate_1000_X_adv_transform_v5_10_50_100_0.5_0.5","wb") as f:
            pickle.dump(evaluate_x_adv,f)
        with open(self.output_file+"/markov_evaluate_1000_Y_adv_transform_v5_10_50_100_0.5_0.5","wb") as f:
            pickle.dump(evaluate_y,f)
        # evaluate_x_adv=evaluate_x_adv.numpy()
        print("干净样本准确率:",clean_acc,"对抗样本准确率:",adv_acc)
        print("平均扰动量:",average_per_count,"平均扰动率:",average_per_ratio)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alpha',
        type=int,
        default=30,
        help="the weight of pertubaiton amount")
    parser.add_argument(
        '--beta',
        type=int,
        default=50,
        help="the weight of attack success rate")
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help="training epochs")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help="batch_size of training")
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help="learning rate")
    parser.add_argument(
        '--up_file',
        type=str,
        default='iter_dataset/markov_universal_1000_trainLoop2000_1_1000_selfDefined_v3',
        help="the file path of universe pertubation")
    parser.add_argument(
        '--output_file',
        type=str,
        default='generator_dataset/',
        help="the file path of output")

    args = parser.parse_args()
    attack = Attack(args)
    # attack.train()
    attack.evaluate()



