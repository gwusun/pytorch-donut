import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def best_threshold(y_true, scores):
    """
    :param y_true: np.array
    :param scores: np.array
    :return:
    """
    # scores 标准化
    _min = np.min(scores)
    _max = np.max(scores)
    scores_ = (scores - _min) / (_max - _min + 1e-8)
    pr, re, thrs = precision_recall_curve(y_true, scores_)
    fs = 2.0 * pr * re / np.clip(pr + re, a_min=1e-4, a_max=None)
    print("best F1 score", max(fs))
    return max(fs)

def proprocess(x, y, slide_win=120):
    """
    Standardize(zero mean ??) and fill missing with zero

    :param sliding_window: sliding window size
    :param x: 1-D array
        origin kpi
    :param y: 1-D array
        label
    :return: zero mean standardized kpi,
    """
    # todo 标准化和用0补充缺失点
    ret_x, ret_y = slide_sampling(x, y, slide_win=slide_win)
    return ret_x, ret_y


def slide_sampling(x, y, slide_win):
    ret_x = []
    ret_y = []
    for i in range(len(x) - slide_win + 1):
        ret_x.append(x[i: i + slide_win])
        # ret_y.append(y[i + slide_win - 1])
        ret_y.append(y[i: i + slide_win])
    ret_x = np.array(ret_x)
    ret_y = np.array(ret_y)
    return ret_x, ret_y


class TsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).int()

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return data, label

    def __len__(self):
        return self.x.shape[0]





class VAE(nn.Module):
    def __init__(self, input_size=120, output_size=100, latent_dim=3):
        # encoder， decoder都是两个隐藏层
        super(VAE, self).__init__()
        self.win = input_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )

        self.en_miu = nn.Linear(output_size, latent_dim)
        self.en_std = nn.Sequential(
            nn.Linear(output_size, latent_dim),
            nn.Softplus()
        )
        self.epsilon = 0.0001

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )

        self.de_miu = nn.Linear(output_size, input_size)
        self.de_std = nn.Sequential(
            nn.Linear(output_size, input_size),
            nn.Softplus()
        )

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)




    def forward(self, x, n_sample=1):
        """
        :param x:
        :param n_sample: z的采样次数
        :return:
        """
        if self.training:
            # 训练时z的采样次数为1
            # Variational
            # out_z_miu, out_z_std = self.encoder(x), self.encoder(x)
            # z_miu = self.en_miu(out_z_miu)
            # z_std = self.en_std(out_z_std) + self.epsilon

            encoder_out = self.encoder(x)
            z_miu = self.en_miu(encoder_out)
            z_std = self.en_std(encoder_out) + self.epsilon
            z = z_miu + z_std * torch.randn(z_miu.shape[0], z_miu.shape[1])  # 重参数化采样

            # Generative
            # out_x_miu, out_x_std = self.decoder(z), self.decoder(z)
            # x_miu = self.de_miu(out_x_miu)
            # x_std = self.de_std(out_x_std) + self.epsilon

            decoder_out =  self.decoder(z)
            x_bar_miu = self.de_miu(decoder_out)
            x_bar_std = self.de_std(decoder_out) + self.epsilon
            return z, x_bar_miu, x_bar_std, z_miu, z_std

        else:
            # out_z_miu, out_z_std = self.encoder(x), self.encoder(x)
            # z_miu = self.en_miu(out_z_miu)
            # z_std = self.en_std(out_z_std) + self.epsilon
            encoder_out = self.encoder(x)
            z_miu = self.en_miu(encoder_out)
            z_std = self.en_std(encoder_out) + self.epsilon
            # 测试时z的采样次数为 n_sample, 直接采样，不用重参数化
            batch_size = z_miu.shape[0]
            z_miu = z_miu.repeat(n_sample, 1).view(n_sample, batch_size, -1)  # size: n_sample * batch_size * win
            z_std = z_std.repeat(n_sample, 1).view(n_sample, batch_size, -1)
            z = torch.normal(mean=z_miu, std=z_std)  # size: n_sample * batch_size * win

            # Generative
            # out_x_miu, out_x_std = self.decoder(z), self.decoder(z)
            # x_miu = self.de_miu(out_x_miu)
            # x_std = self.de_std(out_x_std) + self.epsilon

            decoder_out = self.decoder(z)
            x_bar_miu = self.de_miu(decoder_out)
            x_bar_std = self.de_std(decoder_out) + self.epsilon
            # gen_x = torch.normal(mean=x_miu, std=x_std)  # 直接采样
            return z, x_bar_miu, x_bar_std, z_miu, z_std


def plot_predict(origin_x,origin_y, ret_x_bar, ret_x_bar_std, ret_scores):
    N = len(ret_x_bar)
    index = np.arange(N)
    fig, ax = plt.subplots(3,figsize=(16,14))
    ax[0].set_title("X and it corresponding predicting value")
    ax[0].plot(index, origin_x, '-')
    ax[0].fill_between(index, ret_x_bar-ret_x_bar_std,ret_x_bar+ret_x_bar_std,alpha=0.2)
    ax[1].plot(index,origin_y)
    ax[1].set_title("Label for the associated x")
    ax[2].plot(index,ret_scores)
    ax[2].set_title("Reconstruction Probability (log probability)")
    plt.show()



class Donut:
    def __init__(self, CKPT_path=None):
        self._vae = VAE()
        self.optimizer = Adam(self._vae.parameters(), lr=0.001, weight_decay=0.001)
        if CKPT_path is not None:
            model_CKPT = torch.load(CKPT_path)
            self._vae.load_state_dict(model_CKPT['state_dict'])
            self.optimizer.load_state_dict(model_CKPT['optimizer'])
            print('load vae and optimizer form file')

    def m_elbo_loss(self, train_x, train_y, z, x_bar_miu, x_bar_std, z_miu, z_std):
        """
        采用蒙特卡洛估计， 根据论文，设置采样次数 L=1
        :param train_x: batch_size * win, 样本
        :param train_y: batch_size * 1, 标签， 0代表正常， 1代表异常
        :param z: batch_size * latent_size, 观察到的隐变量
        :param x_bar_miu: batch_size * win, x服从正态分布的均值
        :param x_bar_std: batch_size * win, x服从正态分布的标准差
        :param z_miu: batch_size * latent_size, 后验z服从正态分布的均值
        :param z_std: batch_size * latent_size, 后验z服从正态分布的标准差
        :param z_prior_mean: int, 先验z服从正态分布的均值， 一般设置为0
        :param z_prior_std: int, 先验z服从正态分布的标准差， 一般设置为1
        :return:
        """
        z_prior_mean = torch.zeros(size=z_miu.shape)
        z_prior_std = torch.ones(size=z_miu.shape)

        # 以下蒙特卡洛估计的采样次数均为1
        # 蒙特卡洛估计 log p(x|z)。在重构的x的正态分布上， 取值为train_x时，概率密度函数的log值； batch_size * win
        log_p_x_given_z = - torch.log(math.sqrt(2 * math.pi) * x_bar_std) - ((train_x - x_bar_miu) ** 2) / (
                    2 * x_bar_std ** 2)

        # 蒙特卡洛估计 log p(z). p(z)为先验分布， 一般设置为标准正态分布； batch_size * latent_size
        log_p_z = - torch.log(math.sqrt(2 * math.pi) * z_prior_std) - ((z - z_prior_mean) ** 2) / (2 * z_prior_std ** 2)

        # 蒙特卡洛估计 log q(z|x). q(z|x)为z的后验分布; batch_size * latent_size
        log_q_z_given_x = - torch.log(math.sqrt(2 * math.pi) * z_std) - ((z - z_miu) ** 2) / (2 * z_std ** 2)

        # 去除缺失点的影响，也是m-elbo的精髓
        normal = 1 - train_y  # batch_size * win
        log_p_x_given_z = normal * log_p_x_given_z  # batch_size * win

        # beta, log_p_z 的放缩系数
        beta = torch.sum(normal, dim=1) / normal.shape[1]  # size = batch_size

        # m-elbo的值
        m_elbo = torch.sum(log_p_x_given_z, dim=1) + beta * torch.sum(log_p_z, dim=1) - torch.sum(log_q_z_given_x,
                                                                                                  dim=1)
        m_elbo = torch.mean(m_elbo) * (-1)
        return m_elbo

    def fit(self, x, y, n_epoch=30, valid_x=None, valid_y=None):
        '''
        如果在实际应用中没有标签， 可以把大多数样本当做正常样本， 即把y全部设置为0
        :param x:(batch, window_size)
        :param y:(batch, window_size)
        :param n_epoch: int
        :param valid_x: (batch, window_size)
        :param valid_y: (batch, window_size)
        :return:
        '''
        # todo missing injection
        # Sets the module in training mode, so we can train it.
        self._vae.train(mode=True)
        train_dataset = TsDataset(x, y)
        train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)

        if valid_x is not None:
            valid_dataset = TsDataset(valid_x, valid_y)
            valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=0)

        optimizer = self.optimizer
        lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.75)
        for epoch in range(n_epoch):
            for train_x, train_y in train_iter:
                optimizer.zero_grad()
                z, x_bar_miu, x_bar_std, z_miu, z_std = self._vae(train_x)  # 前向传播
                l = self.m_elbo_loss(train_x, train_y, z, x_bar_miu, x_bar_std, z_miu, z_std)
                # 计算loss对每个参数的梯度
                l.backward()
                # 更新每个参数： a -= learning_rate * a.grad
                optimizer.step()
            lr_scheduler.step()

            # 保存模型
            # if epoch % 100 == 0:
            #     print("保存模型")
            #     torch.save({"state_dict": self._vae.state_dict(),
            #                 "optimizer": optimizer.state_dict(),
            #                 "loss": l.item()}, "./model_parameters/epoch{}-loss{:.2f}.tar".format(epoch, l.item()))

            # 验证集
            if epoch % 50 == 0 and valid_x is not None:
                with torch.no_grad():
                    flag = 1
                    for v_x, v_y in valid_iter:
                        z, x_bar_miu, x_bar_std, z_miu, z_std = self._vae(v_x)
                        v_l = self.m_elbo_loss(v_x, v_y, z, x_bar_miu, x_bar_std, z_miu, z_std)
                        if flag == 1:
                            flag = 0
                            v_x_ = v_x[0].view(1, 120)
                            z, x_bar_miu, x_bar_std, z_miu, z_std = self._vae(v_x_)
                            # restruct_compare_plot(v_x_.view(120), x_miu.view(120))
                    print("train loss %.4f,  valid loss %.4f" % (l.item(), v_l.item()))
                    with open("log.txt", "a") as f:
                        f.writelines("%d %.4f %.4f\n" % (epoch, l.item(), v_l.item()))
            else:
                print("loss", l.item(), "  lr,", lr_scheduler.get_last_lr())

    def predict(self, test_x, test_y):
        # Sets the module in evaluation mode so we can predict.
        # It is actually calling the function: self.train(False)
        self._vae.eval()  # equal to self.train(False)

        test_dataset = TsDataset(test_x, test_y)
        test_iter = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
        ret_scores = []
        ret_x_bar_mean=[]
        ret_x_bar_std=[]
        with torch.no_grad():
            for batch_x, batch_y in test_iter:
                # batch_x.shape=(256,120)
                # batch_y.shape=(256,120)

                # n_sample denotes the number of sampling
                # z.shape=z_miu.shape=z_std.shape=（50，256，3）
                # x_bar_mu.shape=x_bar_std.shape=(50,256,120)
                z, x_bar_mu, x_bar_std, z_miu, z_std = self._vae(batch_x, n_sample=50)  # forward by calling VAE.forward()
                # 蒙特卡洛估计 E log p(x|z), 重构概率
                # log_p_x_given_z.shape=(50,256,120)
                log_p_x_given_z = - torch.log(math.sqrt(2 * math.pi) * x_bar_std) - \
                                  ((batch_x - x_bar_mu) ** 2) / (2 * x_bar_std ** 2)
                # log_p_x_given_z[:,:,-1].shape=（50，256）
                anomaly_score = torch.mean(log_p_x_given_z[:, :, -1], dim=0)  # 异常分数越小，越可能为异常(对数概率）

                # Using the last value in each window to represent a value
                ret_scores.append(anomaly_score)
                ret_x_bar_mean.append(torch.mean(x_bar_mu[:,:,-1],dim=0))
                ret_x_bar_std.append(torch.mean(x_bar_std[:,:,-1],dim=0))
            # socres.shape=(
            ret_scores = torch.cat(ret_scores)
            ret_x_bar_mean=torch.cat(ret_x_bar_mean)
            ret_x_bar_std=torch.cat(ret_x_bar_std)
            # scores = torch.cat((torch.ones(self._vae.win - 1) * torch.min(scores), scores), dim=0)
            # todo 使用segment 评判， 需要将时序恢复成原来的样本，而不是滑动窗口取到的片段
            assert len(ret_scores) == len(test_dataset)
            return ret_scores,ret_x_bar_mean,ret_x_bar_std





