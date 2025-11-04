# ProyectoMicro2
## Características a implementar: 
### 1. Sensor magnético MC-39
Estado: Completo
### 2. Sensor ultra sónico 
Estado: incompleto 
### 3. Sensor BH1750 (sensor de luz)
Estado: incompleto 
### 4. Sensor humedad FC-28 
Estado: incompleto
### 5. Sensor de temperatura LM-75
Estado: incompleto
### 6. LED rgb y servomotor mostrar resultados 
Estado: incompleto
### 7. Pantalla OLED mostrar resultados
Estado: incompleto
### 8. Salida de datos a google sheets
Estado: incompleto
## Resumen de los pines pensados--
[!NOTE]
Pin  Componente Función
D1  I2C SCL     OLED + LM75 + BH1750
D2  I2C         SDAOLED + LM75 + BH1750
D3  HC-SR04     Echo
D4  Servo       Señal PWM
D5  MC-38       Sensor magnético
D6  LED         RGBVerde
D7  LED         RGBAzul
D8  HC-SR04     Trigger
A0  FC-28Humedad (analógico)
