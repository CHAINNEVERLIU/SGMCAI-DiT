import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


class MultiCurvePlotter:
    def __init__(self):
        self.colors = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        self.markers = ['o', 's', 'D', '^', 'v', 'p', 'h']
        self.font = fm.FontProperties(family='Times New Roman', size=20)  # 自定义字体样式
        self.ticks_font = fm.FontProperties(family='Times New Roman', size=15)  # 自定义字体样式

    def plot(self, x, curves, labels=None, x_label='Sample Number', y_label='Output Value', title=None, save_path=None):
        """
        在同一个图中绘制多条曲线

        参数:
            x: array, x坐标数据
            curves: list of arrays, 多组y坐标数据
            labels: list of str, 曲线标签，可选参数
            x_label: str, x轴标签，可选参数
            y_label: str, y轴标签，可选参数
            title: str, 图标题，可选参数
            save_path: str, 图像保存路径，可选参数，当传入路径时则会将画出的图像保存至制定路径，需注意的是路径需制定最后文件名
        """
        plt.figure(figsize=(10, 5), dpi=200)
        for i, curve in enumerate(curves):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            label = None if labels is None else labels[i]
            plt.plot(x, curve, color=color, marker=marker, label=label, linewidth=2.0, markersize=3)
            plt.xticks(fontproperties=self.ticks_font)
            plt.yticks(fontproperties=self.ticks_font)

        if labels is not None:
            plt.legend(facecolor='gainsboro', prop={'family': 'Times New Roman', 'size': 15})

        if x_label is not None:
            plt.xlabel(x_label, fontproperties=self.font)

        if y_label is not None:
            plt.ylabel(y_label, fontproperties=self.font)

        if title is not None:
            plt.title(title, fontproperties=self.font)

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()

    def subplot(self, x_list, y_list, labels=None, x_labels=None, y_labels=None, titles=None, save_path=None):
        """
        在多个子图中绘制多组曲线

        参数:
            x_list: list of arrays, 多组x坐标数据
            y_list: list of arrays, 多组y坐标数据
            labels: list of str, 曲线标签，可选参数
            x_labels: list of str, x轴标签，可选参数
            y_labels: list of str, y轴标签，可选参数
            titles: list of str, 子图标题，可选参数
            save_path: str, 图像保存路径，可选参数，当传入路径时则会将画出的图像保存至制定路径，需注意的是路径需制定最后文件名
        """
        num_plots = len(x_list)
        nums = int(num_plots / 2)
        if num_plots % 2:
            nums = nums + 1
        else:
            pass

        fig, axs = plt.subplots(nums, 2, figsize=(5 * num_plots, 5), dpi=200)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            color = self.colors[i % len(self.colors)]
            marker = self.markers[i % len(self.markers)]
            label = None if labels is None else labels[i]
            x_label = None if x_labels is None else x_labels[i]
            y_label = None if y_labels is None else y_labels[i]
            title = None if titles is None else titles[i]
            if nums == 1:
                current_axs = axs[i]
            else:
                row = int(i / 2)
                col = i % 2
                current_axs = axs[row, col]
            current_axs.plot(x, y, color=color, marker=marker, label=label, linewidth=2.0, markersize=3)

            if labels is not None:
                current_axs.legend(facecolor='gainsboro', prop={'family': 'Times New Roman', 'size': 15})

            current_axs.tick_params(axis='both', which='major', labelsize=15)

            if x_label is not None:
                current_axs.set_xlabel(x_labels[i], fontproperties=self.font)

            if y_label is not None:
                current_axs.set_ylabel(y_labels[i], fontproperties=self.font)

            if title is not None:
                current_axs.set_title(title, fontsize=20, fontproperties=self.font)

        if save_path is not None:
            plt.savefig(save_path)

        plt.show()
