# ProyectoMicro2
## Componentes Utilizados

- **Microcontrolador**: ESP8266 (NodeMCU)
- **Display**: LCD OLED 128x64 (I2C)
- **Actuadores**:
  - LED RGB
  - Servo Motor
  - Pantalla OLED
- **Sensores**:
  - HC-SR04 (Ultrasonido)
  - LM75 (Temperatura)
  - BH1750 (Luz ambiental)
  - FC-28 (Humedad de suelo)
  - MC-38 (Magnético/Proximidad)
---
## Características a implementar: 
### 1. Sensor magnético MC-39
Estado: Completo
### 2. Sensor ultra sónico 
Estado: Completo
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
## Resumen de Pines

| Pin ESP8266 | Componente | Función |
|-------------|------------|---------|
| **D0** | LED RGB | Rojo |
| **D1** | I2C SCL | OLED + LM75 + BH1750 |
| **D2** | I2C SDA | OLED + LM75 + BH1750 |
| **D3** | HC-SR04 | Echo |
| **D4** | Servo | Señal PWM |
| **D5** | MC-38 | Sensor magnético |
| **D6** | LED RGB | Verde |
| **D7** | LED RGB | Azul |
| **D8** | HC-SR04 | Trigger |
| **A0** | FC-28 | Humedad (analógico) |

---
