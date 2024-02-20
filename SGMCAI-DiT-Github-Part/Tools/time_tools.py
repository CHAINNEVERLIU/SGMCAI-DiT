import numpy as np
import pandas as pd


def time_to_hours(df):
    """
    将时间DataFrame转换为等长整数序列，其中整数序列第一个为0，而后每个位置是相应时间间隔的小时数

    参数：
    df -- 时间DataFrame，包含一个或多个时间列

    返回：
    hours -- 等长整数序列，每个位置是相应时间间隔的小时数
    """
    # 将所有时间列转换为datetime类型
    df = pd.to_datetime(df)

    # 计算每个时间列与第一个时间的时间差，单位为小时
    hours = (df - df.iloc[0]).astype('timedelta64[h]')

    # 将第一个位置设置为0
    hours.iloc[0] = 0

    hours = np.array(hours, dtype=np.int)

    return hours



