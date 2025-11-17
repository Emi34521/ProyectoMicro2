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
  - BMP180 (Presión)
  - FC-28 (Humedad de suelo)
  - MC-38 (Magnético/Proximidad)
---
## Características a implementar: 
### 1. Sensor magnético MC-39
Estado: Completo
### 2. Sensor ultra sónico 
Estado: Completo
### 3. Sensor BMP180 (sensor presión)
Estado: Completo
Se cambió el sensor BH1750 por el BMP180 debido a diversos problemas con el sensor.
Ahora la funcionalidad de este es determinar cambios de altura, se conectará un
motor para que esté variando sus mediciones. Se implementará después.
### 4. Sensor humedad FC-28 
Estado: Completo
Calibración: 
valores de 0 a 24 --ambiente seco 5%
valores de 50 a 120 --Húmedo, ideal para muchas plantas
valores de 315 a 450 --Muy húmedo, sumergido en agua 
### 5. Sensor de temperatura LM-75
Estado: Completo. Añadí algunas medidas de ejemplo. Están sujetas a cambios.
### 6. Motor para variar altura
Estado: espera (cancelado)
La placa no tiene la suficiente intensidad ni voltaje para alimentar al servomotor y al motor. Por lo que decidió eliminarlos
### 7. LED rgb y servomotor mostrar resultados 
Estado: completo
### 8. Pantalla OLED mostrar resultados
Estado: completo
### 9. Salida de datos a google sheets
Estado: completo
## Resumen de Pines

| Pin ESP8266 | Componente | Función |
|-------------|------------|---------|
| **D0** | LED RGB | Rojo |
| **D1** | I2C SCL | OLED + LM75 + BMP180 |
| **D2** | I2C SDA | OLED + LM75 + BMP180 |
| **D3** | HC-SR04 | Echo |
| **D4** | Servo | Señal PWM |
| **D5** | MC-38 | Sensor magnético |
| **D6** | LED RGB | Verde |
| **D7** | LED RGB | Azul |
| **D8** | HC-SR04 | Trigger |
| **A0** | FC-28 | Humedad (analógico) |

---
