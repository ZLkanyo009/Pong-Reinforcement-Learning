import torch
import torch.nn as nn
import numpy as np 
import Pong as game

import random
import os

import cv2

class MyNet(nn.Module):
    def __init__(self,learning_rate):
        super(MyNet,self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(4,8,3,1,padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8,8,1,2, bias=False),
            nn.ReLU(),
            nn.Conv2d(8,8,3,1,padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(8,16,3,1,padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16,16,1,2, bias=False),
            nn.ReLU(),
            nn.Conv2d(16,16,3,1,padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16,16,1,2, bias=False),
            nn.ReLU(),
            nn.Conv2d(16,5,3,1,padding=1, bias=False),
            nn.ReLU()
        )
        self.fc = nn.Sequential(    
            nn.Linear(48*5,48*5),
            nn.ReLU(),
            nn.Linear(48*5,3)
        )
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr = learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def forward(self, inputs):
        inputs = self.ConvNet(inputs)
        inputs = inputs.view(-1,48*5)
        inputs = self.fc(inputs)
        return inputs

width_net = 64
height_net = 48
store_count = 0
store_size = 50000
decline = 0.6
learning_rate = 0.0001
learn_time = 0
update_time = 50 # 1000 
gama = 0.99
b_size = 32          #每次训练网络，从store_size个状态中取32组来训练
store_picture = np.zeros((store_size,2,4,height_net,width_net)) #存储store_size个state(即game截图)
store_other = np.zeros((store_size,3))        #存储store_size个用到的值，包括：采取的下一步行动a, 报酬值reward, 表示游戏是否结束的done_show
start_study = False
log_dir = './model/RL-pong.pth'
log_dir2 = './best/best-RL-pong.pth'
log_dir3 = './best/best-RL-pong-all.pth'
train = True

net1 = MyNet(learning_rate).cuda()
net2 = MyNet(learning_rate).cuda()
optimizer = torch.optim.Adam(net1.parameters(), lr = learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

if os.path.exists(log_dir):
    checkpoint = torch.load(log_dir)
    net1.load_state_dict(checkpoint['model'])
    net2.load_state_dict(checkpoint['model'])
    net1.opt.load_state_dict(checkpoint['optimizer'])
    net2.opt.load_state_dict(checkpoint['optimizer'])
    learn_time = checkpoint['epoch'] 
    print("--------------成功加载模型--------------")
else:
    print("--------------训练全新模型--------------")
                
s = game.game_init()
s = np.reshape(cv2.resize(s,(width_net,height_net)),(1,1,height_net,width_net)) 
s = np.concatenate((s,s,s,s),axis = 1)
s = s / 255.
for i in range(5000000):
    while True:
        if random.randint(0,100) < 100*(decline ** (learn_time/10000)):  #if后面的式子表达的意思：训练轮数越多，每次越倾向于选择网络预测值中最大的a作为抉择，而不倾向于随机选择一个抉择
            a = random.choice([0, 1, 2])
            print("random")
        else:
            inputs = torch.Tensor(s).cuda()
            out = net1(inputs).detach()
            a = torch.argmax(out).data.item() 
        s_part,r,score_one_round,score,done_show = game.game_step(a)
        s_part = np.reshape(cv2.resize(s_part,(width_net,height_net)),(1,1,height_net,width_net))
        s_part = s_part / 255.
        s_ = np.concatenate((s[:,1:4,:,:],s_part),axis = 1)

        if (np.all(s_[:,0,:,:] == s_[:,1,:,:]) and np.all(s_[:,0,:,:] == s_[:,2,:,:])):
            print("3个s")
        done = True
        
        store_picture[store_count % store_size][0] = s
        store_picture[store_count % store_size][1] = s_ 

        store_other[store_count % store_size][0] = a
        store_other[store_count % store_size][1] = r
        store_other[store_count % store_size][2] = done_show
        
        store_count += 1
        s = s_
        
        rank_file_r = open("rank.txt","r")
        best = int(rank_file_r.readline())
        rank_file_r.close()
        
        if score_one_round > best:
            state = {'model':net1.state_dict(), 'epoch':learn_time}
            torch.save(state, log_dir2)
            rank_file_w = open("rank.txt","w")
            rank_file_w.write("%d" % score_one_round)
            print("********** best score_one_round updated!! *********")
            rank_file_w.close()

        rank_file_r2 = open("rank_all_round.txt","r")
        best2 = int(rank_file_r2.readline())
        rank_file_r2.close()
        
        if score > best2:
            state = {'model':net1.state_dict(), 'epoch':learn_time}
            torch.save(state, log_dir3)
            rank_file_w2 = open("rank_all_round.txt","w")
            rank_file_w2.write("%d" % score)
            print("********** best score updated!! *********")
            rank_file_w2.close()

        if store_count > store_size and train:
            if learn_time % update_time == 0 :
                net2.load_state_dict(net1.state_dict())
            
            index = random.randint(0, store_size - b_size - 1)
            
            b_s = torch.Tensor(store_picture[index:index + b_size, 0])
            b_s = b_s.cuda()
            
            b_s_ = torch.Tensor(store_picture[index:index + b_size, 1])
            b_s_ = b_s_.cuda()
            
            b_a = torch.Tensor(store_other[index:index + b_size, 0:1]).long()  #转为LongTensor
            b_a = b_a.cuda() 
            
            b_r = torch.Tensor(store_other[index:index + b_size, 1:2])
            b_r = b_r.cuda()

            b_done = torch.Tensor(store_other[index:index + b_size, 2:3])
            b_done = b_done.cuda()

            q = net1(b_s).gather(1,b_a)            #dim = 1，使用b_a来选取net中最终采取的抉择带来的Q值，防止选最大一方为最后Q值后，忽略了随机选择的存在。
            q_next = net2(b_s_).detach().max(1)[0].reshape(b_size, 1)
            q_truth = b_r + gama*q_next*(1-b_done)
            
            q = q.cuda()
            q_truth = q_truth.cuda()
           
            loss = nn.MSELoss()(q, q_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            learn_time += 1
            if not start_study:
                print("===========start study===========")
                start_study = True
                print("epcho = %d" % learn_time)
                break 
            if done:
                print("epcho = %d" % learn_time)
                print("loss = %f" % loss.item())

                if learn_time % 1000 == 0 :
                    state = {'model':net1.state_dict(), 'optimizer':net1.opt.state_dict(), 'epoch':learn_time}
                    torch.save(state, log_dir)
                    f = open("scores.txt","a")
                    f.write("========= %d ========== \n" % learn_time)
                    f.close()
                break
            