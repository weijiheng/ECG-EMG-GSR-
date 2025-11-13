串口通信类和储存类基本不需要修改尽量不要动已经足够完善。



#GUI界面
添加一个控件QLabel用于显示心率，以及一个属性`self.heart_rate`。
```python
self.heart_rate = 0
self.heart_rate_label = QLabel(f"心率：{self.heart_rate} bpm")
```
编写一个`self.heart_rate_label`的回调函数用于计算实时心率，心率数据储存于数据结构中`ECG_Data`中。可以尝试直接调用一些现有库函数。

#重构绘图函数`update_plot(self)`
当前绘图函数已有原型但是无法满足实时显示数据，可以通过降采样显示数据，避免失真，每个plot显示当前5-10s内的数据图像，定时更新图像。
根据选择的模块绘制对应的图像

#当前的控件回调
## _数据保存函数`save_intermediate_data(self)`_
基本绝大部分的回调函数已经完成，保存数据的回调函数有两个但是只需要用`save_intermediate_data(self):`。
该代码是需要注意的：
保存文件命名不在需要当前阶段`self.current_phase`
保存文件根据选择的模块适配保存数据，例如
两个模块都选择ecg和emg保存文件名为：实验编号_实验时间_emg_ecg.csv
一个模块选择ecg保存文件名为：实验编号_实验时间_ecg.csv
一个模块选择emg保存文件名为：实验编号_实验时间_emg.csv
不可两个模块都不选择！！！


一些代码存在实验人数`self.subject_count`的判断注意修复，