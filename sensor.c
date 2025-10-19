/* 
GSR connection pins to Arduino microcontroller

Arduino           GSR
GND               GND
5V                VCC
A0                SIG
D13             RED LED
*/

/*
GSR, standing for galvanic skin response, is a method of 
measuring the electrical conductance of the skin. Strong 
emotion can cause stimulus to your sympathetic nervous 
system, resulting more sweat being secreted by the sweat 
glands. Grove – GSR allows you to spot such strong emotions 
by simple attaching two electrodes to two fingers on one hand,
an interesting gear to create emotion related projects, like 
sleep quality monitor. http://www.seeedstudio.com/wiki/Grove_-_GSR_Sensor
*/

const byte PACKET_HEADER = 0XAA;
const byte PACKET_TAIL = 0X55;
const int GSR1_pin = A0;
const int ECG1_pin = A1;
const int EMG1_pin = A2;
const int GSR2_pin = A3;
const int ECG2_pin = A4;
const int EMG2_pin = A5;

// 传感器值变量
volatile int GSR1;
volatile int ECG1;
volatile int EMG1;
volatile int GSR2;
volatile int ECG2;
volatile int EMG2;

void setup() {
  Serial.begin(500000);
  
  // 初始化定时器1，设置1ms中断
  initTimer1();
  
  delay(1000);
}

// 定时器1初始化函数，设置为1ms中断一次
void initTimer1() {
  noInterrupts();  // 关闭全局中断
  
  // 重置定时器1控制寄存器
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1  = 0;
  
  // 设置比较值，实现1ms中断
  // 16MHz时钟，预分频64，计数250次 = 1ms
  OCR1A = 249;
  
  // CTC模式
  TCCR1B |= (1 << WGM12);
  
  // 预分频64
  TCCR1B |= (1 << CS11) | (1 << CS10);
  
  // 允许定时器比较匹配中断
  TIMSK1 |= (1 << OCIE1A);
  
  interrupts();  // 开启全局中断
}

// 定时器1比较匹配中断服务程序
ISR(TIMER1_COMPA_vect) {
  // 读取传感器值
  GSR1 = analogRead(GSR1_pin);
  EMG1 = analogRead(EMG1_pin);
  ECG1 = analogRead(ECG1_pin);
  GSR2 = analogRead(GSR2_pin);
  EMG2 = analogRead(EMG2_pin);
  ECG2 = analogRead(ECG2_pin);

  // Serial.println(GSR1);
  // Serial.println(EMG1);
  //  Serial.println(ECG1);
  //  Serial.println(GSR2);
  //  Serial.println(EMG2);
  //  Serial.println(ECG2);
  
  // 打包发送数据
  int sensorData[6] = {GSR1, EMG1, ECG1, GSR2, EMG2, ECG2};

  sendPacket(sensorData);
}

// 发送数据包函数
void sendPacket(int* data) {
  Serial.write(PACKET_HEADER);
  for(int i = 0; i < 6; i++) {
    Serial.write(lowByte(data[i]));  // 发送低字节
    Serial.write(highByte(data[i])); // 发送高字节
  }
  Serial.write(PACKET_TAIL);
}

void loop() {

}