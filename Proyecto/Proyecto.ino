// ==================== LIBRERÍAS ====================
#include <Wire.h>
#include <Adafruit_BMP085.h>
#include <Servo.h>
#include <U8g2lib.h>

// ==================== PINES ====================
// Sensor Magnético MC-38
const int PIN_MC38 = D5;

// Sensor Ultrasónico HC-SR04
const int PIN_TRIGGER = D8;
const int PIN_ECHO = D3;

// Sensor de Humedad FC-28
const int PIN_FC28 = A0;

// LED RGB
const int PIN_LED_ROJO = D0;
const int PIN_LED_VERDE = D6;
const int PIN_LED_AZUL = D7;

// Servo Motor
const int PIN_SERVO = D4;

// I2C (BMP180, LM75, OLED)
// SCL: D1
// SDA: D2

// ==================== OBJETOS ====================
Adafruit_BMP085 bmp;
Servo servoMotor;
// Probar con SH1106 primero (más común en pantallas 0.96")
U8G2_SH1106_128X64_NONAME_F_HW_I2C display(U8G2_R0, U8X8_PIN_NONE);

// ==================== CONSTANTES LM75 ====================
#define LM75_ADDRESS 0x48  // Dirección I2C del LM75
#define LM75_TEMP_REGISTER 0x00  // Registro de temperatura

// ==================== VARIABLES LED RGB Y SERVO ====================
int anguloServo = 90; // Posición inicial del servo (centro)
bool oledDetectado = false;
unsigned long ultimaActualizacionOLED = 0;
const unsigned long INTERVALO_OLED = 1000; // Actualizar OLED cada 1 segundo (reduce parpadeo)

// ==================== VARIABLES MC-38 ====================
int contadorCerrado = 0;
bool estadoAnterior = HIGH;

// ==================== VARIABLES HC-SR04 ====================
const float RADIO_DETECCION = 80.0; // cm
const float UMBRAL_MOVIMIENTO = 5.0; // cm - cambio mínimo para detectar movimiento
const int LECTURAS_ESTABLE = 5; // Número de lecturas estables para ignorar objeto

float distanciaActual = 0;
float distanciaPrevia = 0;
int contadorEstable = 0;
bool objetoEstable = false;
bool movimientoDetectado = false;

// ==================== VARIABLES BMP180 ====================
#define N 10              // Número de lecturas para promediar
float buffer[N];          // Arreglo para almacenar lecturas
int indexBuffer = 0;      // Índice del buffer
bool bufferLleno = false; // Para saber si ya se llenó el buffer una vez

float presion = 0;
float altitudBase = 0.0;
float altitudActual = 0.0;
float deltaAltura = 0.0;
bool bmp180Detectado = false;

// Sistema de pisos del edificio
int pisoActual = 0; // 0=PB, 1-4=Pisos
unsigned long ultimoCambioPiso = 0;
const unsigned long INTERVALO_CAMBIO_PISO = 5000; // Cambiar de piso cada 5 segundos (simulación)

// ==================== VARIABLES FC-28 ====================
int valorAnalogico = 0;
int porcentajeHumedad = 0;
String nivelHumedad = "";

// Valores de calibración basados en TU sensor
const int VALOR_SECO = 20;     // Valor en aire seco
const int VALOR_MOJADO = 400;  // Valor en agua
const bool MODO_CALIBRACION = false; // Calibración completada

// ==================== VARIABLES LM75 ====================
float temperaturaLM75 = 0;
String estadoTemp = "";
bool lm75Detectado = false;

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  delay(500); // Delay más largo para estabilidad
  
  Serial.println("\n\n\n=== INICIANDO SISTEMA ===");
  Serial.println("Esperando estabilización...");
  delay(1000);
  
  // Configurar MC-38
  pinMode(PIN_MC38, INPUT_PULLUP);
  Serial.println("✓ MC-38 configurado");
  
  // Configurar HC-SR04
  pinMode(PIN_TRIGGER, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  Serial.println("✓ HC-SR04 configurado");
  
  // Configurar FC-28
  pinMode(PIN_FC28, INPUT);
  Serial.println("✓ FC-28 configurado");
  
  // Configurar LED RGB
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AZUL, OUTPUT);
  // Apagar LED al inicio
  digitalWrite(PIN_LED_ROJO, LOW);
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_AZUL, LOW);
  Serial.println("✓ LED RGB configurado");
  
  // Configurar Servo
  servoMotor.attach(PIN_SERVO);
  servoMotor.write(anguloServo); // Posición inicial
  Serial.println("✓ Servo configurado en posición inicial (90°)");
  
  // Configurar I2C
  Wire.begin(D2, D1); // SDA, SCL
  Serial.println("✓ I2C configurado");
  
  Serial.println("\n=== Sistema Iniciado ===");
  Serial.println("Escaneando bus I2C...");
  
  // Escanear dispositivos I2C
  escanearI2C();
  
  // Inicializar BMP180
  Serial.println("\nIntentando iniciar BMP180...");
  
  if (bmp.begin()) {
    Serial.println("✓ BMP180 iniciado correctamente");
    bmp180Detectado = true;
    
    delay(1000); // Estabilización de sensor
    
    // Inicializar el buffer con la altitud inicial
    float primera = bmp.readAltitude(101325);
    for (int i = 0; i < N; i++) {
      buffer[i] = primera;
    }
    
    altitudBase = primera;
    Serial.print("Altitud base registrada: ");
    Serial.print(altitudBase);
    Serial.println(" m");
    
  } else {
    Serial.println("✗ BMP180 no responde - Verifica conexiones:");
    Serial.println("  VCC -> 3.3V");
    Serial.println("  GND -> GND");
    Serial.println("  SCL -> D1");
    Serial.println("  SDA -> D2");
    bmp180Detectado = false;
  }
  
  // Inicializar LM75
  Serial.println("\nIntentando iniciar LM75...");
  
  Wire.beginTransmission(0x48); // Dirección por defecto del LM75
  if (Wire.endTransmission() == 0) {
    Serial.println("✓ LM75 detectado en dirección 0x48");
    lm75Detectado = true;
  } else {
    Serial.println("✗ LM75 no responde - Verifica conexiones:");
    Serial.println("  VCC -> 3.3V");
    Serial.println("  GND -> GND");
    Serial.println("  SCL -> D1");
    Serial.println("  SDA -> D2");
    Serial.println("  A0, A1, A2 -> GND (dirección 0x48)");
    lm75Detectado = false;
  }
  
  // Inicializar OLED
  Serial.println("\nIntentando iniciar OLED...");
  
  display.begin();
  Wire.setClock(400000); // Velocidad I2C rápida
  
  // Test simple
  display.clearBuffer();
  display.setFont(u8g2_font_ncenB14_tr);
  display.drawStr(0, 20, "SISTEMA");
  display.setFont(u8g2_font_ncenB08_tr);
  display.drawStr(0, 40, "Iniciando...");
  display.sendBuffer();
  
  Serial.println("✓ OLED iniciado (SH1106)");
  oledDetectado = true;
  delay(2000);

  Serial.println("\nSensores: MC-38 + HC-SR04 + BMP180 + FC-28 + LM75");
  Serial.println("Actuadores: LED RGB + Servo + OLED");
  Serial.println("Radio de detección: 80cm | Umbral: 5cm");
  
  // Delay adicional para estabilizar las primeras lecturas
  Serial.println("\nEstabilizando sensores...");
  delay(2000);
}

// ==================== LOOP ====================
void loop() {
  leerMC38();
  leerHCSR04();
  leerBMP180();
  leerFC28();
  leerLM75();
  actualizarActuadores();
  actualizarOLED();
  mostrarDatos();
  
  delay(500); // Delay general del sistema
}

// ==================== FUNCIONES MC-38 ====================
void leerMC38() {
  bool estadoActual = digitalRead(PIN_MC38);
  
  // Detectar flanco de bajada (de abierto a cerrado)
  if (estadoActual == LOW && estadoAnterior == HIGH) {
    contadorCerrado++;
  }
  
  estadoAnterior = estadoActual;
}

// ==================== FUNCIONES HC-SR04 ====================
void leerHCSR04() {
  // Generar pulso de trigger
  digitalWrite(PIN_TRIGGER, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIGGER, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIGGER, LOW);
  
  // Medir el tiempo del pulso Echo
  long duracion = pulseIn(PIN_ECHO, HIGH, 30000);
  
  if (duracion > 0) {
    distanciaActual = duracion * 0.0343 / 2;
    
    // Solo procesar si está dentro del radio de detección
    if (distanciaActual <= RADIO_DETECCION) {
      detectarMovimiento();
    } else {
      // Fuera del radio: resetear
      resetearDeteccion();
    }
  } else {
    // Sin lectura válida: resetear
    resetearDeteccion();
  }
}

void detectarMovimiento() {
  // Calcular cambio de distancia
  float cambio = abs(distanciaActual - distanciaPrevia);
  
  // Si el cambio es significativo: MOVIMIENTO
  if (cambio >= UMBRAL_MOVIMIENTO) {
    movimientoDetectado = true;
    contadorEstable = 0;
    objetoEstable = false;
  } 
  // Si el cambio es pequeño: objeto posiblemente estable
  else {
    contadorEstable++;
    
    // Si se mantiene estable por varias lecturas: ignorar
    if (contadorEstable >= LECTURAS_ESTABLE) {
      objetoEstable = true;
      movimientoDetectado = false;
    }
  }
  
  distanciaPrevia = distanciaActual;
}

void resetearDeteccion() {
  distanciaActual = -1;
  distanciaPrevia = 0;
  contadorEstable = 0;
  objetoEstable = false;
  movimientoDetectado = false;
}

// ==================== FUNCIONES BMP180 ====================
float leerAltitudFiltrada() {
  buffer[indexBuffer] = bmp.readAltitude(101325); 
  indexBuffer = (indexBuffer + 1) % N;

  if (indexBuffer == 0) bufferLleno = true;

  float suma = 0;
  int count = bufferLleno ? N : indexBuffer;

  for (int i = 0; i < count; i++) {
    suma += buffer[i];
  }
  
  return suma / count;
}

void leerBMP180() {
  if (!bmp180Detectado) {
    presion = -999;
    deltaAltura = -999;
    return;
  }
  
  presion = bmp.readPressure() / 100.0; // Convertir a hPa
  altitudActual = leerAltitudFiltrada();
  deltaAltura = altitudActual - altitudBase; // Sin multiplicador
  
  // Simulación de cambio de piso automático cada 5 segundos
  unsigned long tiempoActual = millis();
  if (tiempoActual - ultimoCambioPiso >= INTERVALO_CAMBIO_PISO) {
    // Cambiar aleatoriamente de piso
    int cambio = random(-1, 2); // -1, 0, o 1
    pisoActual += cambio;
    pisoActual = constrain(pisoActual, 0, 4); // Limitar entre PB y Piso 4
    ultimoCambioPiso = tiempoActual;
  }
}

// ==================== FUNCIONES FC-28 ====================
void leerFC28() {
  valorAnalogico = analogRead(PIN_FC28);
  
  if (MODO_CALIBRACION) {
    // Modo calibración: solo mostrar valor RAW
    porcentajeHumedad = -1;
    nivelHumedad = "CALIBRANDO";
    return;
  }
  
  // Mapear el valor a porcentaje (0-100%)
  // Este sensor: a MAYOR valor = MAYOR humedad
  porcentajeHumedad = map(valorAnalogico, VALOR_SECO, VALOR_MOJADO, 0, 100);
  porcentajeHumedad = constrain(porcentajeHumedad, 0, 100);
  
  // Clasificar nivel de humedad
  if (porcentajeHumedad < 20) {
    nivelHumedad = "MUY SECO";
  } else if (porcentajeHumedad < 40) {
    nivelHumedad = "SECO";
  } else if (porcentajeHumedad < 60) {
    nivelHumedad = "HÚMEDO";
  } else if (porcentajeHumedad < 80) {
    nivelHumedad = "MUY HÚMEDO";
  } else {
    nivelHumedad = "SATURADO";
  }
}

// ==================== FUNCIONES LM75 ====================
float leerTemperaturaLM75Raw() {
  Wire.beginTransmission(LM75_ADDRESS);
  Wire.write(LM75_TEMP_REGISTER);
  if (Wire.endTransmission() != 0) {
    return -999; // Error de comunicación
  }
  
  Wire.requestFrom(LM75_ADDRESS, 2);
  if (Wire.available() < 2) {
    return -999; // No hay datos disponibles
  }
  
  byte msb = Wire.read();
  byte lsb = Wire.read();
  
  // Convertir los dos bytes a temperatura
  int16_t temp = (msb << 8) | lsb;
  temp >>= 5; // Los 11 bits más significativos son la temperatura
  
  float celsius = temp * 0.125; // Cada bit representa 0.125°C
  return celsius;
}

void leerLM75() {
  if (!lm75Detectado) {
    temperaturaLM75 = -999;
    estadoTemp = "NO CONECTADO";
    return;
  }
  
  temperaturaLM75 = leerTemperaturaLM75Raw();
  
  if (temperaturaLM75 == -999) {
    estadoTemp = "ERROR LECTURA";
    return;
  }
  
  // Clasificar temperatura ambiente
  if (temperaturaLM75 < 10) {
    estadoTemp = "FRÍO";
  } else if (temperaturaLM75 < 20) {
    estadoTemp = "FRESCO";
  } else if (temperaturaLM75 < 26) {
    estadoTemp = "CONFORTABLE";
  } else if (temperaturaLM75 < 30) {
    estadoTemp = "CÁLIDO";
  } else {
    estadoTemp = "CALIENTE";
  }
}

// ==================== FUNCIONES LED RGB Y SERVO ====================
void actualizarActuadores() {
  // ===== LED RGB: Indicador de estado combinado =====
  // Apagar todos primero
  digitalWrite(PIN_LED_ROJO, LOW);
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_AZUL, LOW);
  
  // PRIORIDAD 1: Movimiento detectado (ROJO)
  if (movimientoDetectado) {
    digitalWrite(PIN_LED_ROJO, HIGH);
  }
  // PRIORIDAD 2: Puerta cerrada (AZUL)
  else if (digitalRead(PIN_MC38) == LOW) {
    digitalWrite(PIN_LED_AZUL, HIGH);
  }
  // PRIORIDAD 3: Todo normal (VERDE)
  else {
    digitalWrite(PIN_LED_VERDE, HIGH);
  }
  
  // ===== SERVO: Indicador de humedad del suelo =====
  // Mapear porcentaje de humedad (0-100%) a ángulo del servo (0-180°)
  anguloServo = map(porcentajeHumedad, 0, 100, 0, 180);
  anguloServo = constrain(anguloServo, 0, 180);
  servoMotor.write(anguloServo);
}

// ==================== FUNCIONES OLED ====================
void actualizarOLED() {
  if (!oledDetectado) return;
  
  // Actualizar solo cada INTERVALO_OLED para reducir parpadeo
  unsigned long tiempoActual = millis();
  if (tiempoActual - ultimaActualizacionOLED < INTERVALO_OLED) {
    return;
  }
  ultimaActualizacionOLED = tiempoActual;
  
  display.clearBuffer();
  display.setFont(u8g2_font_6x10_tr);
  
  // === LÍNEA 1: Puerta Automática ===
  display.drawStr(0, 10, "Puerta:");
  if (digitalRead(PIN_MC38) == LOW) {
    display.drawStr(50, 10, "Cerrada");
  } else {
    display.drawStr(50, 10, "Abierta");
  }
  
  // === LÍNEA 2: Sistema de Iluminación ===
  display.drawStr(0, 21, "Luces:");
  if (movimientoDetectado) {
    display.drawStr(50, 21, "ENCENDIDAS");
  } else {
    display.drawStr(50, 21, "Apagadas");
  }
  
  // === LÍNEA 3: Aire Acondicionado ===
  display.drawStr(0, 32, "A/C:");
  if (lm75Detectado && temperaturaLM75 != -999) {
    if (temperaturaLM75 > 26) {
      display.drawStr(50, 32, "ACTIVO");
    } else {
      display.drawStr(50, 32, "Apagado");
    }
    // Mostrar temperatura al lado
    char tempBuf[10];
    sprintf(tempBuf, "%.1fC", temperaturaLM75);
    display.drawStr(95, 32, tempBuf);
  } else {
    display.drawStr(50, 32, "N/A");
  }
  
  // === LÍNEA 4: Sistema de Riego ===
  display.drawStr(0, 43, "Riego:");
  char buffer[20];
  if (porcentajeHumedad < 40) {
    display.drawStr(50, 43, "ACTIVO");
    sprintf(buffer, "%d%%", porcentajeHumedad);
    display.drawStr(95, 43, buffer);
  } else {
    display.drawStr(50, 43, "Apagado");
    sprintf(buffer, "%d%%", porcentajeHumedad);
    display.drawStr(95, 43, buffer);
  }
  
  // === LÍNEA 5: Ascensor (altura) ===
  display.drawStr(0, 54, "Ascensor:");
  if (bmp180Detectado) {
    // Calcular planta actual basada en altura
    float alturaTotal = deltaAltura;
    int planta = 0;
    
    if (alturaTotal < 2.0) {
      // Planta baja (0-2m)
      planta = 0;
      display.drawStr(55, 54, "PB");
    } else if (alturaTotal < 6.0) {
      // Planta 1 (2-6m)
      planta = 1;
      display.drawStr(55, 54, "Piso 1");
    } else if (alturaTotal < 10.0) {
      // Planta 2 (6-10m)
      planta = 2;
      display.drawStr(55, 54, "Piso 2");
    } else if (alturaTotal < 14.0) {
      // Planta 3 (10-14m)
      planta = 3;
      display.drawStr(55, 54, "Piso 3");
    } else {
      // Planta 4 o superior (14m+)
      planta = 4;
      display.drawStr(55, 54, "Piso 4");
    }
    
    // Mostrar altura en metros al final
    sprintf(buffer, "%.1fm", alturaTotal);
    display.drawStr(95, 54, buffer);
  } else {
    display.drawStr(55, 54, "N/A");
  }
  
  display.sendBuffer();
}

// ==================== FUNCIÓN DIAGNÓSTICO I2C ====================
void escanearI2C() {
  byte error, address;
  int nDevices = 0;
  
  Serial.println("Escaneando direcciones 0x01-0x7F...");
  
  for(address = 1; address < 127; address++ ) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.print("✓ Dispositivo I2C encontrado en 0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);
      
      // Identificar dispositivos comunes
      if (address == 0x77) Serial.print(" (BMP180)");
      if (address == 0x48) Serial.print(" (LM75)");
      if (address == 0x3C) Serial.print(" (OLED)");
      
      Serial.println();
      nDevices++;
    }
  }
  
  if (nDevices == 0) {
    Serial.println("✗ No se encontraron dispositivos I2C");
    Serial.println("  Verifica las conexiones SDA y SCL");
  } else {
    Serial.print("Total de dispositivos encontrados: ");
    Serial.println(nDevices);
  }
}

// ==================== MOSTRAR DATOS ====================
void mostrarDatos() {
  Serial.println("═════════════════════════════════════════");
  
  // MC-38
  Serial.print("MC-38: ");
  if (digitalRead(PIN_MC38) == LOW) {
    Serial.print("CERRADO");
  } else {
    Serial.print("ABIERTO ");
  }
  Serial.print("  | Contador: ");
  Serial.println(contadorCerrado);
  
  // HC-SR04
  Serial.print("HC-SR04: ");
  if (distanciaActual > 0 && distanciaActual <= RADIO_DETECCION) {
    Serial.print(distanciaActual, 1);
    Serial.print(" cm");
    
    // Estado de detección
    if (objetoEstable) {
      Serial.print(" [OBJETO ESTABLE]");
    } else if (movimientoDetectado) {
      Serial.print(" [*** MOVIMIENTO ***]");
    } else {
      Serial.print(" [Analizando...]");
    }
    Serial.println();
  } else {
    Serial.println("Sin detección en 80cm");
  }
  
  // BMP180
  Serial.print("BMP180: ");
  if (bmp180Detectado) {
    Serial.print(presion, 1);
    Serial.print(" hPa  |  Δh: ");
    Serial.print(deltaAltura, 2);
    Serial.println(" m");
  } else {
    Serial.println("[NO CONECTADO - Ver diagnóstico arriba]");
  }
  
  // FC-28
  Serial.print("FC-28: ");
  
  if (MODO_CALIBRACION) {
    Serial.print("*** MODO CALIBRACIÓN ***  ADC RAW: ");
    Serial.println(valorAnalogico);
    Serial.println("  -> Coloca el sensor en AIRE SECO y anota el valor");
    Serial.println("  -> Coloca el sensor en AGUA y anota el valor");
    Serial.println("  -> Actualiza VALOR_SECO y VALOR_MOJADO en el código");
  } else {
    Serial.print(porcentajeHumedad);
    Serial.print("% [");
    Serial.print(nivelHumedad);
    Serial.print("]  (ADC: ");
    Serial.print(valorAnalogico);
    Serial.println(")");
  }
  
  // LM75
  Serial.print("LM75: ");
  if (lm75Detectado) {
    Serial.print(temperaturaLM75, 1);
    Serial.print(" °C  [");
    Serial.print(estadoTemp);
    Serial.println("]");
  } else {
    Serial.println("[NO CONECTADO - Ver diagnóstico arriba]");
  }
  
  // Actuadores
  Serial.print("LED RGB: ");
  if (movimientoDetectado) {
    Serial.print("🔴 ROJO (Movimiento detectado)");
  } else if (digitalRead(PIN_MC38) == LOW) {
    Serial.print("🔵 AZUL (Puerta cerrada)");
  } else {
    Serial.print("🟢 VERDE (Normal)");
  }
  Serial.println();
  
  Serial.print("SERVO: ");
  Serial.print(anguloServo);
  Serial.print("° (Humedad: ");
  Serial.print(porcentajeHumedad);
  Serial.println("%)");
  
  Serial.print("OLED: ");
  if (oledDetectado) {
    Serial.println("✓ Mostrando datos");
  } else {
    Serial.println("[NO CONECTADO]");
  }
}