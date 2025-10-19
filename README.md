# 生物信号采集

## 硬件电路
该硬件电路采用ECG,EMG,GSR模块作为硬件采集端，具体模块在网上购买，具体型号可自行选择。

该简易采集设备具有两个GSR,两个EMG,两个ECG信号采集通道，可通过配套上位机预选采集信号人数（1-2人）。
该设计以一个GSR，一个ECG,一个ECG为一组，共两组。

供电：采用可反复充电使用的18650电池(3.7v)作为电源，共三节。后续通过稳压电路，将11.1v转为3.3v为采集系统供电。GSR,EMG,ECG采集输入电压均为3.3v。

Arduino通过软件开发编写实现采样频率1KHZ，在采样上述三种生物信号时满足Nyquist Sampling Theorem，避免采样信号在频域出现频率混叠影响采样质量。ADC分辨率为10bit，能有效量化上述三种生物信号并满足使用需求。Arduino软件实现对每次采样数据进行逐字节拆分整型数据并按小端序（Little-Endian）发送，该方式节约传输带宽，接收端能快速解读数据。使用包头包尾作为数据检验位。Arduino通过串口通信传输采集数据，并以此作为供电。

该系统设计了双层PCB硬件电路，顶层电路为信号传输电路，电源电路。底层电路为接地铺铜连接所有硬件电路GND。两组采集通道置于arduino两侧，便于区分, 每组配备有独立开关。Arduino采用倒扣的方式安装在PCB电路板上，易于安装。

<img width="1127" height="777" alt="image" src="https://github.com/user-attachments/assets/00744586-a836-47dc-8f01-e9e3086082ec" />

## 软件

### arduino

### 上位机
<img width="1064" height="760" alt="image" src="https://github.com/user-attachments/assets/2bee3630-89bb-40a5-b40b-5865b7dbda2a" />


