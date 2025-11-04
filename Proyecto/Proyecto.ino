// Sensor Magnético MC-38
const int PIN_MC38 = D5;  // Pin digital 5

int contadorCerrado = 0;
bool estadoAnterior = HIGH;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_MC38, INPUT_PULLUP);

  Serial.println("\n=== Sensor MC-38 Iniciado ===");
  delay(1000);
}

void loop() {
  bool estadoActual = digitalRead(PIN_MC38);

  // Detectar flanco de bajada (de abierto a cerrado)
  if (estadoActual == LOW && estadoAnterior == HIGH) {
    contadorCerrado++;
  }

  estadoAnterior = estadoActual;

  if (estadoActual == LOW) {
    Serial.print("Estado: CERRADO  | Contador: ");
  } else {
    Serial.print("Estado: ABIERTO  | Contador: ");
  }
  Serial.println(contadorCerrado);

  delay(500);
}
