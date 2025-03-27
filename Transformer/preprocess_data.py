import os
import pandas as pd
import numpy as np
import glob
from yahoodownloader import YahooDownloader
import argparse

def preprocess_cn_stocks(input_dir, output_dir):
    """
    预处理中国股票数据
    
    Parameters
    ----------
    input_dir : str
        原始数据目录
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        stock_code = os.path.basename(file_path).replace(".csv", "")
        print(f"处理 {stock_code}...")
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 确保日期列是日期类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 按日期排序
        df = df.sort_values('date')
        
        # 计算技术指标
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        # 相对强弱指标 (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 布林带
        df['middle_band'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['upper_band'] = df['middle_band'] + (std * 2)
        df['lower_band'] = df['middle_band'] - (std * 2)
        
        # 计算收益率
        df['return'] = df['close'].pct_change()
        
        # 计算波动率
        df['volatility'] = df['return'].rolling(window=20).std()
        
        # 标准化价格和交易量
        df['norm_close'] = (df['close'] - df['close'].mean()) / df['close'].std()
        if 'volume' in df.columns:
            df['norm_volume'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
        
        # 删除NaN值
        df = df.dropna()
        
        # 保存处理后的数据
        output_path = os.path.join(output_dir, f"processed_{stock_code}.csv")
        
        # 选择要保存的列
        if 'volume' in df.columns:
            cols_to_save = ['date', 'close', 'norm_close', 'open', 'high', 'low', 
                           'norm_volume', 'ma5', 'ma10', 'ma20', 'rsi', 
                           'upper_band', 'middle_band', 'lower_band', 'return', 'volatility']
        else:
            cols_to_save = ['date', 'close', 'norm_close', 'open', 'high', 'low', 
                           'ma5', 'ma10', 'ma20', 'rsi', 
                           'upper_band', 'middle_band', 'lower_band', 'return', 'volatility']
        
        df[cols_to_save].to_csv(output_path, index=False)
        print(f"已保存处理后的 {stock_code} 数据到 {output_path}")

def download_and_preprocess_us_stocks(tickers, start_date, end_date, output_dir):
    """
    下载并预处理美国股票数据
    
    Parameters
    ----------
    tickers : list
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
    output_dir : str
        输出目录
    """
    # 创建下载器
    downloader = YahooDownloader(start_date=start_date, end_date=end_date, ticker_list=tickers)
    
    # 下载数据
    raw_data_dir = os.path.join(output_dir, 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    df = downloader.fetch_data(save_path=raw_data_dir)
    
    # 预处理数据
    processed_data_dir = os.path.join(output_dir, 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    
    downloader.preprocess_data(df, output_path=processed_data_dir)
    
    print(f"数据处理完成，已保存到 {processed_data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预处理股票数据')
    parser.add_argument('--mode', type=str, choices=['cn', 'us'], default='cn',
                        help='处理中国股票(cn)或美国股票(us)')
    parser.add_argument('--input_dir', type=str, default='data/cn_stocks/hs300',
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='data/cn_stocks/hs300/preprocessed',
                        help='输出目录')
    parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                        help='美国股票代码列表')
    parser.add_argument('--start_date', type=str, default='2010-01-01',
                        help='开始日期')
    parser.add_argument('--end_date', type=str, default='2023-01-01',
                        help='结束日期')
    
    args = parser.parse_args()
    
    if args.mode == 'cn':
        preprocess_cn_stocks(args.input_dir, args.output_dir)
    else:
        download_and_preprocess_us_stocks(args.tickers, args.start_date, args.end_date, args.output_dir)