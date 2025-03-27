from cgi import test
from torch.utils.data.dataset import Dataset
from data.stock_data_handle import Stock_Data,DatasetStock,DatasetStock_PRED
from exp.exp_basic import Exp_Basic
from models.transformer import Transformer_base as Transformer

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, ranking_loss
import utils.tools as utils
import utils.metrics_object as metrics_object

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import pdb
from tqdm import tqdm  # 添加tqdm导入

import os
import time

dataset_dict = {
    'stock': DatasetStock_PRED,
}

class Exp_pred(Exp_Basic):
    def __init__(self, args, data_all, id):
        super(Exp_pred, self).__init__(args)
        log_dir = os.path.join('log', 'pred_'+args.project_name+'_'+str(args.rank_alpha)+'_'+id)
        print(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.data_all = data_all
    
    def _build_model(self):
        model_dict = {
            'Transformer':Transformer,
        }

        if self.args.model=='Transformer':
            model = model_dict[self.args.model](
                # self.args
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.activation
                # self.device
            )

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model.float()

    def _get_data(self, flag):
        # 确保使用正确的参数名称
        dataset = dataset_dict[self.args.data_type](self.data_all, type=flag, pred_type=self.args.pred_type)
        
        # 添加缺失的变量定义
        batch_size = self.args.batch_size
        shuffle_flag = True if flag == 'train' else False
        drop_last = True if flag == 'train' else False
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last)

        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, metric_builders, stage='test'):
        self.model.eval()
        total_loss = []
        metric_objs = [builder(stage) for builder in metric_builders]

        for i, (batch_x1, batch_x2, batch_y) in enumerate(vali_loader):
            bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
            # 修改这里：保留批次和股票维度的信息
            batch_x1_reshaped = batch_x1.reshape(bs * stock_num, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2_reshaped = batch_x2.reshape(bs * stock_num, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            _, _, output = self.model(batch_x1_reshaped, batch_x2_reshaped)
            # 修改这里：根据输出形状正确重塑
            if output.dim() == 3 and output.shape[1] == 1:
                output = output.reshape(bs, stock_num, output.shape[-1])
                output = output[:, :, 0]  # 取第一个特征作为预测值
            elif output.dim() == 2:
                output = output.reshape(bs, stock_num, output.shape[-1])
                output = output[:, :, 0]  # 取第一个特征作为预测值
            elif output.dim() == 1:
                output = output.reshape(bs, stock_num)
            else:
                try:
                    output = output.reshape(bs, stock_num)
                except:
                    output = torch.mean(output.reshape(-1, output.shape[-1]), dim=1).reshape(bs, stock_num)

            loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)

            total_loss.append(loss.item())

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, metric_objs
        
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'valid')
        test_data, test_loader = self._get_data(flag = 'test')

        metrics_builders = [
        metrics_object.MIRRTop1,
    ]

        path = os.path.join('./checkpoints/',setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        metric_objs = [builder('train') for builder in metrics_builders]

        valid_loss_global = np.inf
        best_model_index = -1

        # 添加总体训练进度条
        epoch_bar = tqdm(range(self.args.train_epochs), desc="训练进度", position=0)
        
        for epoch in epoch_bar:
            iter_count = 0
            train_loss = []
            
            self.model.train()
            # 添加每个epoch内的进度条
            batch_bar = tqdm(enumerate(train_loader), total=train_steps, 
                             desc=f"Epoch {epoch+1}/{self.args.train_epochs}", 
                             position=1, leave=False)
            
            for i, (batch_x1, batch_x2, batch_y) in batch_bar:
                iter_count += 1
                # pdb.set_trace()
                bs, stock_num = batch_x1.shape[0], batch_x1.shape[1]
                
                # 修改这里：保留批次和股票维度的信息
                batch_x1_reshaped = batch_x1.reshape(bs * stock_num, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
                batch_x2_reshaped = batch_x2.reshape(bs * stock_num, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                _,_, output = self.model(batch_x1_reshaped, batch_x2_reshaped)
                # 打印形状信息以便调试
                if i == 0 and epoch == 0:
                    print(f"batch_x1 shape: {batch_x1.shape}")
                    print(f"batch_x1_reshaped shape: {batch_x1_reshaped.shape}")
                    print(f"batch_x2 shape: {batch_x2.shape}")
                    print(f"batch_x2_reshaped shape: {batch_x2_reshaped.shape}")
                    print(f"batch_y shape: {batch_y.shape}")
                    print(f"output shape: {output.shape}")
                
                # 修改这里：根据输出形状正确重塑
                # 如果输出是 [bs*stock_num, 1, 10]，则重塑为 [bs, stock_num, 10]，然后取最后一个维度
                if output.dim() == 3 and output.shape[1] == 1:
                    output = output.reshape(bs, stock_num, output.shape[-1])
                    # 如果需要，可以取最后一个维度的平均值或某个特定索引
                    output = output[:, :, 0]  # 取第一个特征作为预测值
                # 如果输出是 [bs*stock_num, 10]，则重塑为 [bs, stock_num, 10]，然后取最后一个维度
                elif output.dim() == 2:
                    output = output.reshape(bs, stock_num, output.shape[-1])
                    output = output[:, :, 0]  # 取第一个特征作为预测值
                # 如果输出是 [bs*stock_num]，则重塑为 [bs, stock_num]
                elif output.dim() == 1:
                    output = output.reshape(bs, stock_num)
                else:
                    # 如果以上都不适用，打印更多调试信息
                    print(f"无法自动重塑输出，输出形状: {output.shape}, 预期形状: [{bs}, {stock_num}]")
                    # 尝试强制重塑为 [bs, stock_num]
                    try:
                        output = output.reshape(bs, stock_num)
                    except:
                        # 如果重塑失败，尝试其他方法
                        print("警告：无法重塑输出，将使用平均值")
                        output = torch.mean(output.reshape(-1, output.shape[-1]), dim=1).reshape(bs, stock_num)
            
                loss = criterion(output, batch_y) + self.args.rank_alpha * ranking_loss(output, batch_y)
                train_loss.append(loss.item())

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                
                # 更新进度条信息
                batch_bar.set_postfix(loss=f"{loss.item():.7f}")
                
                if (i+1) % 100==0:
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    # 不再需要打印迭代信息，因为进度条已经显示了
                    iter_count = 0
                    time_now = time.time()

                with torch.no_grad():
                    for metric in metric_objs:
                        metric.update(output, batch_y)

        train_loss = np.average(train_loss)
        valid_loss, valid_metrics = self.vali(vali_data, vali_loader, criterion, metrics_builders, stage='valid')
        test_loss, test_metrics = self.vali(test_data, test_loader, criterion, metrics_builders, stage='test')

        self.writer.add_scalar('Train/loss', train_loss, epoch)
        self.writer.add_scalar('Valid/loss', valid_loss, epoch)
        self.writer.add_scalar('Test/loss', test_loss, epoch)

        # pdb.set_trace()

        all_logs = {
            metric.name: metric.value for metric in metric_objs + valid_metrics + test_metrics
        }
        for name, value in all_logs.items():
            self.writer.add_scalar(name, value.mean(), global_step=epoch)

        # 更新总进度条信息
        epoch_bar.set_postfix(train_loss=f"{train_loss:.7f}", 
                             valid_loss=f"{valid_loss:.7f}", 
                             test_loss=f"{test_loss:.7f}")

        torch.save(self.model.state_dict(), path+'/'+'checkpoint_{0}.pth'.format(epoch+1))

        if valid_loss.item() < valid_loss_global:
            valid_loss_global = valid_loss.item()
            best_model_index = epoch+1

        adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint_{0}.pth'.format(best_model_index)
        self.model.load_state_dict(torch.load(best_model_path))
        print('best model index: ', best_model_index)
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        outputs = []
        real = []
        
        self.model.eval()

        metrics_builders = [
        metrics_object.MIRRTop1,
        metrics_object.RankIC
    ]
        
        metric_objs = [builder('test') for builder in metrics_builders]
        
        for i, (batch_x1, batch_x2, batch_y) in enumerate(test_loader):
            bs, stock_num = batch_x1.shape[0], batch_x2.shape[1]
            batch_x1 = batch_x1.reshape(-1, batch_x1.shape[-2], batch_x1.shape[-1]).float().to(self.device)
            batch_x2 = batch_x2.reshape(-1, batch_x2.shape[-2], batch_x2.shape[-1]).float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            _,_, output = self.model(batch_x1, batch_x2)

            output = output.reshape(bs,stock_num)

            with torch.no_grad():
                for metric in metric_objs:
                    metric.update(output, batch_y)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        all_logs = {
                metric.name: metric.value for metric in metric_objs
            }
        for name, value in all_logs.items():
            print(name, value.mean())

        return