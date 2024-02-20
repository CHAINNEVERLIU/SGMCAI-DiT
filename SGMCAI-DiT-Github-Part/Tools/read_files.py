import pandas as pd
import numpy as np
import os
import torch
import yaml


class TableReader:
    """
    用于读取各种类型的表格数据
    """
    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.std = None
        self.mean = None

    def read_txt(self, file_path, delimiter='\t', normalize=False, method='minmax'):
        data = pd.read_csv(file_path, delimiter=delimiter)
        return self._process_data(data, normalize, method)

    def read_csv(self, file_path, normalize=False, method='minmax'):
        data = pd.read_csv(file_path)
        return self._process_data(data, normalize, method)

    def read_xlsx(self, file_path, sheet_name=None, normalize=False, method='minmax'):
        if sheet_name is None:
            data = pd.read_excel(file_path)
        else:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        return self._process_data(data, normalize, method)

    def _process_data(self, data, normalize=False, method='minmax'):
        time_col = None
        data_cols = []

        for col in data.columns:
            if data[col].dtype == np.dtype('datetime64[ns]'):
                time_col = data[col]
            else:
                data_cols.append(data[col].to_numpy(dtype=np.float32))

        data_array = np.column_stack(data_cols).astype(np.float32)

        if normalize:
            data_array = self._normalize_data(data_array, method=method)  # 归一化或标准化数据

        return time_col, data_array

    def read_file(self, file_path, normalize=False, method='minmax'):
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'txt':
            return self.read_txt(file_path, normalize=normalize, method=method)
        elif file_extension == 'csv':
            return self.read_csv(file_path, normalize=normalize, method=method)
        elif file_extension == 'xlsx':
            return self.read_xlsx(file_path, normalize=normalize, method=method)
        else:
            raise ValueError("Unsupported file format.")

    def _normalize_data(self, data, method='minmax'):
        if method == 'minmax':
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
            normalized_data = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        elif method == 'standard':
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            normalized_data = (data - self.mean) / (self.std + 1e-8)
        else:
            raise ValueError("Invalid normalization method.Only support minmax or standard!")

        return normalized_data

    def inverse_normalize(self, data, method='inverse_minmax'):
        if method == 'inverse_minmax':
            if self.min_val is None or self.max_val is None:
                raise ValueError("请先利用TableReader类归一化数据")
            """将归一化数据反归一化为原始数据"""
            normalized_data = np.asarray(data)
            min_val = np.asarray(self.min_val)
            max_val = np.asarray(self.max_val)

            original_data = normalized_data * (max_val - min_val) + min_val
            return original_data
        elif method == 'inverse_standard':
            if self.mean is None or self.std is None:
                raise ValueError("请先利用TableReader类标准化数据")
            """将标准化数据反标准化为原始数据"""
            standardized_data = np.asarray(data)
            mean = np.asarray(self.mean)
            std = np.asarray(self.std)

            original_data = standardized_data * std + mean
            return original_data
        else:
            raise ValueError("Invalid inverse normalization method.Only support inverse_minmax or inverse_standard!")

    @staticmethod
    def sliding_window(data, window_length, stride):
        """
        对二维tensor进行滑窗处理
        :param data: 输入的二维张量
        :param window_length: 滑窗长度
        :param stride: 滑窗移动步长
        :return: 滑窗后的数据
        """
        data_vars = data.shape[1]  # 变量个数

        data = data.unsqueeze(0)  # [batch, channels, number, vars]
        unfolded_data = data.unfold(dimension=1, size=window_length, step=stride)
        unfolded_data = unfolded_data.reshape((-1, data_vars, window_length)).transpose(1, 2)

        return unfolded_data

    @staticmethod
    def save_numpy_array_as_excel(array, path, sheet_name='Sheet1'):
        # 检查目录是否存在，如果不存在则创建它
        os.makedirs(os.path.dirname(path), exist_ok=True)

        df = pd.DataFrame(array)
        df.to_excel(path, sheet_name=sheet_name, index=False)
        print(f"NumPy array saved as Excel file at {path}")


def save_dict_to_yaml(data, filename):
    """
    Save a dictionary to a YAML file.

    Parameters:
    - data: The dictionary to be saved.
    - filename: The name of the YAML file.
    """
    with open(filename, 'w') as yaml_file:
        for key, value in data.items():
            yaml.dump({key: value}, yaml_file, default_flow_style=False)
            yaml_file.write('\n')  # Add a blank line between dictionaries
    print(f'Data has been saved to {filename}')


def load_dict_from_yaml(filename):
    """
    Load a dictionary from a YAML file.

    Parameters:
    - filename: The name of the YAML file.

    Returns:
    - The loaded dictionary.
    """
    with open(filename, 'r') as yaml_file:
        loaded_data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return loaded_data





