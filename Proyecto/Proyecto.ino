// ==================== PINES ====================
// Sensor Magnético MC-38
const int PIN_MC38 = D5;

// Sensor Ultrasónico HC-SR04
const int PIN_TRIGGER = D8;
const int PIN_ECHO = D3;

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

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  
  // Configurar MC-38
  pinMode(PIN_MC38, INPUT_PULLUP);
  
  // Configurar HC-SR04
  pinMode(PIN_TRIGGER, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  Serial.println("\n=== Sistema Iniciado ===");
  Serial.println("Sensores: MC-38 + HC-SR04 (Detector de Movimiento)");
  Serial.println("Radio de detección: 80cm | Umbral: 5cm");
  delay(1000);
}

// ==================== LOOP ====================
void loop() {
  leerMC38();
  leerHCSR04();
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

// ==================== MOSTRAR DATOS ====================
void mostrarDatos() {
  Serial.println("─────────────────────────────");
  
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
      Serial.print(" [OBJETO ESTABLE - IGNORADO]");
    } else if (movimientoDetectado) {
      Serial.print(" [*** MOVIMIENTO DETECTADO ***]");
    } else {
      Serial.print(" [Analizando...]");
    }
    Serial.println();
  } else {
    Serial.println("Sin detección en radio de 80cm");
  }
}