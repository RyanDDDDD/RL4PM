"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import yfinance as yf
import os


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, save_path=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------
        proxy : str, optional
            proxy server, by default None
        save_path : str, optional
            path to save individual stock data, by default None

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            print(f"下载 {tic} 的数据...")
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date, proxy=proxy)
            temp_df["tic"] = tic
            
            # 如果指定了保存路径，则保存单个股票数据
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, f"raw_{tic}.csv")
                temp_df.to_csv(file_path)
                print(f"已保存 {tic} 的数据到 {file_path}")
            
            data_df = data_df.append(temp_df)
            
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        """选择具有相同或更多行数的股票
        
        Parameters
        ----------
        df : pd.DataFrame
            包含多个股票数据的DataFrame
            
        Returns
        -------
        pd.DataFrame
            筛选后的DataFrame
        """
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
        
    def preprocess_data(self, df, output_path=None):
        """预处理数据并可选择保存到指定路径
        
        Parameters
        ----------
        df : pd.DataFrame
            原始数据
        output_path : str, optional
            输出路径, by default None
            
        Returns
        -------
        pd.DataFrame
            预处理后的数据
        """
        # 按股票代码分组处理
        processed_dfs = []
        
        for tic, group in df.groupby('tic'):
            # 按日期排序
            group = group.sort_values('date')
            
            # 计算技术指标
            # 这里可以添加您需要的技术指标计算
            # 例如: 移动平均线, RSI, MACD等
            
            # 示例: 计算5日和10日移动平均线
            group['ma5'] = group['close'].rolling(window=5).mean()
            group['ma10'] = group['close'].rolling(window=10).mean()
            
            # 计算收益率
            group['return'] = group['close'].pct_change()
            
            # 计算波动率 (20日滚动标准差)
            group['volatility'] = group['return'].rolling(window=20).std()
            
            # 删除NaN值
            group = group.dropna()
            
            # 标准化价格和交易量
            group['norm_close'] = (group['close'] - group['close'].mean()) / group['close'].std()
            group['norm_volume'] = (group['volume'] - group['volume'].mean()) / group['volume'].std()
            
            # 如果需要保存处理后的数据
            if output_path:
                os.makedirs(output_path, exist_ok=True)
                file_path = os.path.join(output_path, f"processed_{tic}.csv")
                
                # 选择要保存的列
                cols_to_save = ['date', 'close', 'norm_close', 'open', 'high', 'low', 
                               'norm_volume', 'tic', 'ma5', 'ma10', 'return', 'volatility']
                
                group[cols_to_save].to_csv(file_path, index=False)
                print(f"已保存处理后的 {tic} 数据到 {file_path}")
            
            processed_dfs.append(group)
        
        # 合并所有处理后的数据
        processed_df = pd.concat(processed_dfs, axis=0)
        return processed_df
