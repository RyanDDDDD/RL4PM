import os
import numpy as np
import pandas as pd
import glob
from torch.utils.data import Dataset, DataLoader
import torch
# 删除重复的导入
# import glob  # 添加这一行导入glob模块

# 删除重复的导入
# import torch
# from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import sys
from utils.preprocess import FeatureEngineer, data_split
import config
import datetime

import pdb


class Stock_Data():
    def __init__(self, root_path, dataset_name, full_stock_path, size, attr = config.TECHNICAL_INDICATORS_LIST, temporal_feature = config.TEMPORAL_FEATURE, scale=True, prediction_len=[2,5]):
        # size [seq_len, label_len, pred_len]
        self.scale = scale
        self.attr = attr
        self.temporal_feature = temporal_feature
        self.root_path = root_path
        self.full_stock = full_stock_path
        self.ticker_list = config.use_ticker_dict[dataset_name]
        self.border_dates = config.date_dict[dataset_name]
        self.prediction_len = prediction_len

        self.seq_len = size[0] # seq_len
        self.type_map = {'train':0, 'valid':1, 'test':2}
        self.pred_type_map = {'label_short_term':0, 'label_long_term':1}


        self.__read_data__() 

    def __read_data__(self):
        scaler = StandardScaler()
        stock_num = len(self.ticker_list)

        full_stock_dir = os.path.join(self.root_path, self.full_stock)
        
        # 检查目录是否存在
        if not os.path.exists(full_stock_dir):
            raise ValueError(f"目录不存在: {full_stock_dir}")
        
        # 列出目录中的所有文件
        all_files = os.listdir(full_stock_dir)
        print(f"目录中的文件: {all_files[:10]}...")  # 打印前10个文件用于调试
        
        # 首先读取一个文件来检查列名
        sample_files = [f for f in all_files if f.startswith('processed_')]
        if not sample_files:
            raise ValueError(f"在 {full_stock_dir} 中没有找到处理过的股票数据文件")
        
        sample_file = os.path.join(full_stock_dir, sample_files[0])
        sample_df = pd.read_csv(sample_file)
        available_columns = sample_df.columns.tolist()
        print(f"可用列: {available_columns}")
        
        # 确定要使用的列
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        optional_columns = ['dopen', 'dclose', 'dhigh', 'dlow', 'dvolume', 'price']
        
        use_columns = ['date']
        for col in required_columns[1:] + optional_columns:
            if col in available_columns:
                use_columns.append(col)
            elif col == 'price' and 'close' in available_columns:
                # 如果没有price列但有close列，可以用close代替
                pass
        
        print(f"将使用以下列: {use_columns}")
        
        df = pd.DataFrame([], columns=use_columns + ['tic'])
        for ticket in self.ticker_list:
            # 尝试不同的文件名格式
            file_path = os.path.join(full_stock_dir, ticket+'.csv')
            if not os.path.exists(file_path):
                # 尝试其他可能的文件名格式
                possible_files = [f for f in all_files if ticket in f]
                if possible_files:
                    file_path = os.path.join(full_stock_dir, possible_files[0])
                    print(f"使用替代文件: {file_path}")
                else:
                    print(f"警告: 找不到股票 {ticket} 的数据文件，跳过")
                    continue
            
            try:
                temp_df = pd.read_csv(file_path, usecols=use_columns)
                
                temp_df['date'] = temp_df['date'].apply(lambda x:str(x))
                temp_df['date'] = pd.to_datetime(temp_df['date'])
                
                # 添加缺失的列
                if 'price' not in temp_df.columns and 'close' in temp_df.columns:
                    temp_df['price'] = temp_df['close']
                
                # 添加短期和长期标签
                temp_df['label_short_term'] = temp_df['close'].pct_change(periods=self.prediction_len[0]).shift(periods=(-1*self.prediction_len[0]))
                temp_df['label_long_term'] = temp_df['close'].pct_change(periods=self.prediction_len[1]).shift(periods=(-1*self.prediction_len[1]))
                
                # 添加缺失的差分列
                for col in ['open', 'close', 'high', 'low', 'volume']:
                    if col in temp_df.columns and f'd{col}' not in temp_df.columns:
                        temp_df[f'd{col}'] = temp_df[col].pct_change()
                
                temp_df['tic'] = ticket
                df = pd.concat((df, temp_df))
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        # 如果没有成功读取任何数据，抛出错误
        if len(df) == 0:
            raise ValueError("未能读取任何股票数据")
            
        df = df.sort_values(by=['date','tic'])
        
        # 确保所有必需的列都存在
        for col in self.temporal_feature:
            if col not in df.columns:
                print(f"警告: 列 {col} 不存在，将使用零填充")
                df[col] = 0
        
        fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

        print("generate technical indicator...")
        df = fe.preprocess_data(df)

        # add covariance matrix as states
        df=df.sort_values(['date','tic'],ignore_index=True)
        df.index = df.date.factorize()[0]

        cov_list = []
        return_list = []

        # look back is one year
        print("generate convariate matrix...")
        lookback=252
        
        # 获取实际读取到的股票数量
        actual_stock_num = len(df['tic'].unique())
        print(f"实际读取到的股票数量: {actual_stock_num}")
        
        for i in range(lookback,len(df.index.unique())):
            data_lookback = df.loc[i-lookback:i,:]
            price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close') 
            return_lookback = price_lookback.pct_change().dropna()
            return_list.append(return_lookback)
    
            covs = return_lookback.cov().values 
            cov_list.append(covs)

        df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
        df = df.merge(df_cov, on='date')
        df = df.sort_values(['date','tic']).reset_index(drop=True)

        df['date_str'] = df['date'].apply(lambda x: datetime.datetime.strftime(x,'%Y%m%d'))

        dates = df['date_str'].unique().tolist()
        print(f"可用日期范围: {dates[0]} 到 {dates[-1]}")
        print(f"配置的日期边界: {self.border_dates}")
        
        # 处理日期边界问题
        def find_closest_date(target_date, date_list):
            # 如果目标日期存在，直接返回
            if target_date in date_list:
                return date_list.index(target_date)
            
            # 否则找最接近的日期
            # 将日期转换为整数进行比较
            target = int(target_date)
            dates_int = [int(d) for d in date_list]
            
            # 找到第一个大于等于目标日期的索引
            for i, d in enumerate(dates_int):
                if d >= target:
                    return i
            
            # 如果所有日期都小于目标日期，返回最后一个日期
            return len(date_list) - 1
        
        try:
            boarder1_ = dates.index(self.border_dates[0])
        except ValueError:
            print(f"警告: 日期 {self.border_dates[0]} 不在数据中，使用最接近的日期")
            boarder1_ = find_closest_date(self.border_dates[0], dates)
            
        try:
            boarder1 = dates.index(self.border_dates[1])
        except ValueError:
            print(f"警告: 日期 {self.border_dates[1]} 不在数据中，使用最接近的日期")
            boarder1 = find_closest_date(self.border_dates[1], dates)
            
        try:
            boarder2_ = dates.index(self.border_dates[2])
        except ValueError:
            print(f"警告: 日期 {self.border_dates[2]} 不在数据中，使用最接近的日期")
            boarder2_ = find_closest_date(self.border_dates[2], dates)
            
        try:
            boarder2 = dates.index(self.border_dates[3])
        except ValueError:
            print(f"警告: 日期 {self.border_dates[3]} 不在数据中，使用最接近的日期")
            boarder2 = find_closest_date(self.border_dates[3], dates)
            
        try:
            boarder3_ = dates.index(self.border_dates[4])
        except ValueError:
            print(f"警告: 日期 {self.border_dates[4]} 不在数据中，使用最接近的日期")
            boarder3_ = find_closest_date(self.border_dates[4], dates)
            
        try:
            boarder3 = dates.index(self.border_dates[5])
        except ValueError:
            print(f"警告: 日期 {self.border_dates[5]} 不在数据中，使用最接近的日期")
            boarder3 = find_closest_date(self.border_dates[5], dates)

        self.boarder_end = [boarder1, boarder2, boarder3]
        self.boarder_start = [boarder1_, boarder2_, boarder3_]
        
        df_data = df[self.attr]
        df_data = df_data.replace([np.inf], config.INF)
        df_data = df_data.replace([-np.inf], config.INF*(-1))
        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values

        cov_list = np.array(df['cov_list'].values.tolist()) # [stock_num*len, stock_num]
        feature_list = np.array(df[self.temporal_feature].values.tolist()) # [stock_num*len, 10]
        close_list = np.array(df['price'].values.tolist())

        # 打印形状信息以便调试
        print(f"cov_list shape: {cov_list.shape}")
        print(f"feature_list shape: {feature_list.shape}")
        print(f"close_list shape: {close_list.shape}")
        
        # 计算天数
        days = len(df) // actual_stock_num
        print(f"计算得到的天数: {days}")
        
        # 使用实际股票数量而不是配置的股票数量
        data_cov = cov_list.reshape(-1, actual_stock_num, cov_list.shape[1], cov_list.shape[2]) # [day, num_stocks, num_stocks, num_stocks]
        data_technical = data.reshape(-1, actual_stock_num, len(self.attr)) # [day, stock_num, technical_len]
        data_feature = feature_list.reshape(-1, actual_stock_num, len(self.temporal_feature)) # [day, stock_num, temporal_feature_len=10]
        data_close = close_list.reshape(-1, actual_stock_num)

        # 这里也需要使用 actual_stock_num 而不是 stock_num
        label_short_term = np.array(df['label_short_term'].values.tolist()).reshape(-1, actual_stock_num)
        label_long_term = np.array(df['label_long_term'].values.tolist()).reshape(-1, actual_stock_num)

        self.data_all = np.concatenate((data_cov[:, 0, :, :], data_technical, data_feature), axis=-1) # [days, num_stocks, cov+technical_len+feature_len]
        self.label_all = np.stack((label_short_term, label_long_term), axis=0) # [2, days, num_stocks, 1]
        self.dates = np.array(dates)
        self.data_close = data_close

        print("data shape: ",self.data_all.shape)
        print("label shape: ",self.label_all.shape)
        


class DatasetStock_MAE(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE, pred_type=None):
        super().__init__()
        assert type in ['train', 'test', 'valid']
        pos = stock.type_map[type]
        self.feature_len = len(feature)

        self.data = stock.data_all[stock.boarder_start[pos]: stock.boarder_end[pos]+1]
        self.label = stock.label_all[:, stock.boarder_start[pos]: stock.boarder_end[pos]+1]

        # pdb.set_trace()

    def __getitem__(self, index):
        seq_x = self.data[index, :, :-self.feature_len]
        return seq_x
    
    def __len__(self):
        return len(self.data)


class DatasetStock_PRED(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE, pred_type='label_short_term'):
        super().__init__()
        assert type in ['train', 'test', 'valid']
        
        # 修改断言，增加对 'long_term_len' 和 'short_term_len' 的支持
        if pred_type == 'long_term_len':
            pred_type = 'label_long_term'
        elif pred_type == 'short_term_len':
            pred_type = 'label_short_term'
            
        assert pred_type in ['label_short_term', 'label_long_term']
        print(pred_type)
        pos = stock.type_map[type]

        self.label_type = stock.pred_type_map[pred_type]
        self.start_pos = stock.boarder_start[pos]
        self.end_pos = stock.boarder_end[pos]+1
        print(self.start_pos, self.end_pos)

        self.feature_len = len(feature)
        self.feature_day_len = stock.seq_len
        self.data = stock.data_all
        self.label = stock.label_all
        
        self.dates = stock.dates[self.start_pos: self.end_pos]
        self.data_close = stock.data_close[self.start_pos: self.end_pos]
        
        # 添加有效索引列表，排除序列长度不足的样本
        self.valid_indices = []
        for i in range(self.end_pos - self.start_pos):
            position = self.start_pos + i
            # 检查是否有足够的历史数据
            if position >= self.feature_day_len - 1:
                self.valid_indices.append(i)
        
        print(f"有效样本数量: {len(self.valid_indices)}/{self.end_pos - self.start_pos}")

    def __getitem__(self, index):
        # 使用有效索引列表
        valid_idx = self.valid_indices[index]
        position = self.start_pos + valid_idx
        
        # 只使用时间特征，与模型期望的通道数匹配
        seq_x = self.data[position-self.feature_day_len+1:position+1, :, -self.feature_len:].transpose(1,0,2) #[days, num_stocks, feature]-> [num_stocks, days, feature]
        seq_x_dec = seq_x[:, -1:, :]

        seq_y = self.label[self.label_type, valid_idx, :]
        return seq_x, seq_x_dec, seq_y
    
    def __len__(self):
        # 返回有效样本数量
        return len(self.valid_indices)


        

class DatasetStock(Dataset):
    def __init__(self, stock: Stock_Data, type='train', feature=config.TEMPORAL_FEATURE):
        super().__init__()
        assert type in ['train', 'test', 'valid']
        pos = stock.type_map[type]

        self.start_pos = stock.boarder_start[pos]
        self.end_pos = stock.boarder_end[pos]+1
        print(self.start_pos, self.end_pos)

        self.feature_len = len(feature)
        self.feature_day_len = stock.seq_len
        self.data = stock.data_all
        self.label = stock.label_all

        # pdb.set_trace()

    def __getitem__(self, index):
        position = self.start_pos+index
        data1 = self.data[position, :, :-self.feature_len] #[num_stocks, cov+technical]
        data2 = self.data[position-self.feature_day_len+1:position+1, :, -self.feature_len:].transpose(1,0,2) #[days, num_stocks, feature]-> [num_stocks, days, feature]
        
        label1 = self.label[0, index, :]
        label2 = self.label[1, index, :]
        return data1, data2, label1, label2
    
    def __len__(self):
        return self.end_pos-self.start_pos


# 修改这个类的名称，避免与上面的类冲突
class Stock_Data_Simple:
    def __init__(self, root_path, dataset_name, full_stock_path, size, prediction_len):
        # 初始化参数
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.full_stock_path = full_stock_path
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.short_term_len = prediction_len[0]
        self.long_term_len = prediction_len[1]
        
        self.__read_data__()
        
    def __read_data__(self):
        # 读取所有处理过的股票数据文件
        self.file_list = glob.glob(os.path.join(self.full_stock_path, "processed_*.csv"))
        print(f"找到 {len(self.file_list)} 个股票数据文件")
        
        # 读取第一个文件以确定特征数量
        if len(self.file_list) > 0:
            sample_df = pd.read_csv(self.file_list[0], header=None)
            # 假设第一列是日期，第二列是价格，其余是特征
            self.feature_dim = sample_df.shape[1] - 1  # 减去日期列
            print(f"特征维度: {self.feature_dim}")
        else:
            raise ValueError(f"在 {self.full_stock_path} 中没有找到处理过的股票数据文件")
    
    def _get_data(self, file_path):
        # 读取单个股票数据文件
        df = pd.read_csv(file_path, header=None)
        # 移除空行
        df = df.dropna()
        
        # 提取特征列（假设第一列是日期，其余是特征）
        data = df.iloc[:, 1:].values
        
        # 提取股票代码（从文件名中）
        stock_code = os.path.basename(file_path).replace("processed_", "").replace(".csv", "")
        
        return data, stock_code
    
    def _prepare_data(self, data):
        """
        准备训练和测试数据
        """
        # 数据分割: 70% 训练, 20% 验证, 10% 测试
        train_ratio = 0.7
        valid_ratio = 0.2
        
        num_samples = data.shape[0]
        train_size = int(num_samples * train_ratio)
        valid_size = int(num_samples * valid_ratio)
        
        train_data = data[:train_size]
        valid_data = data[train_size:train_size+valid_size]
        test_data = data[train_size+valid_size:]
        
        return train_data, valid_data, test_data
    
    def get_data_loaders(self, batch_size=32, num_workers=10):
        """
        创建训练、验证和测试数据加载器
        """
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        
        # 处理每个股票文件
        for file_path in self.file_list:
            data, stock_code = self._get_data(file_path)
            train_data, valid_data, test_data = self._prepare_data(data)
            
            # 创建数据集
            train_dataset = DatasetStock(train_data, self.seq_len, self.label_len, self.pred_len)
            valid_dataset = DatasetStock(valid_data, self.seq_len, self.label_len, self.pred_len)
            test_dataset = DatasetStock(test_data, self.seq_len, self.label_len, self.pred_len)
            
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)
        
        # 合并所有股票的数据集
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, valid_loader, test_loader


# 修改这个类的名称，避免与上面的类冲突
class DatasetStock_Simple(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        return seq_x, seq_y


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)
        
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, index):
        dataset_idx = np.searchsorted(self.cumulative_lengths, index, side='right')
        if dataset_idx > 0:
            sample_idx = index - self.cumulative_lengths[dataset_idx - 1]
        else:
            sample_idx = index
        
        return self.datasets[dataset_idx][sample_idx]