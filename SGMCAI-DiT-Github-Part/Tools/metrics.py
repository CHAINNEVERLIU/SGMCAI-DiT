import numpy as np


class EvaluationMetrics:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def set_targets(self, y_true, y_pred):
        """
        设置真实值和预测值

        Args:
            y_true: 真实值，numpy数组
            y_pred: 预测值，numpy数组
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def rmse(self):
        """
        计算均方根误差（RMSE）

        Returns:
            均方根误差值
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Targets have not been set.")
        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))

    def mae(self):
        """
        计算平均绝对误差（MAE）

        Returns:
            平均绝对误差值
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Targets have not been set.")
        return np.mean(np.abs(self.y_true - self.y_pred))

    def mape(self):
        """
        计算平均绝对百分比误差（MAPE）

        Returns:
            平均绝对百分比误差值
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Targets have not been set.")
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100

    def mspe(self):
        """
        计算平均平方百分比误差（MSPE）

        Returns:
            平均平方百分比误差值
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Targets have not been set.")
        return np.mean(((self.y_true - self.y_pred) / self.y_true) ** 2) * 100

    def r2(self):
        """
        计算决定系数（R2）

        Returns:
            决定系数值
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Targets have not been set.")
        y_mean = np.mean(self.y_true)
        ss_total = np.sum((self.y_true - y_mean) ** 2)
        ss_residual = np.sum((self.y_true - self.y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def evaluate(self, metrics):
        """
        一次性计算多个指标

        Args:
            metrics: 要计算的指标列表,如metrics = ['rmse', 'mae', 'mape', 'mspe', 'r2']

        Returns:
            包含每个指标计算结果的字典
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Targets have not been set.")
        results = {}
        for metric in metrics:
            if not hasattr(self, metric):
                raise ValueError(f"Invalid metric: {metric}")
            results[metric] = getattr(self, metric)()
        return results




