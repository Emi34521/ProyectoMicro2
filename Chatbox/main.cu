// CC3086 - Lab 9: Smart Home Chat-Box con IA (CUDA + CSV + ESP8266)

#include "common.h"
#include "csv_handler.h"
#include "nlu_engine.h"
#include "data_analyzer.h"
#include "user_interface.h"
#include "command_executor.h"

int main() {
    showWelcome();

    // Cargar datos CSV
    const std::string csv_path = "ConsumoCasaInteligente.csv";
    std::vector<SensorData> csvData = loadCSV(csv_path);

    if (csvData.empty()) {
        printf("  No se encontró el archivo CSV o está vacío.\n");
        printf("   Coloca el archivo en el mismo directorio que el ejecutable.\n");
        printf("   Nombre esperado: ConsumoCasaInteligente.csv\n\n");
        printf("   El sistema funcionará solo con control en tiempo real.\n\n");
    }

    // Inicializar motor NLU
    NLUEngine nluEngine;
    nluEngine.initialize();

    // Inicializar ejecutor de comandos
    CommandExecutor executor;
    
    // Solicitar IP del ESP8266
    printf("\n=== CONFIGURACIÓN DE CONEXIÓN ===\n");
    printf("Ingresa la IP del ESP8266 (ejemplo: 192.168.1.100)\n");
    printf("Presiona Enter para omitir y usar solo análisis de datos\n");
    printf("IP: ");
    
    std::string esp_ip;
    std::getline(std::cin, esp_ip);
    
    if (!esp_ip.empty() && esp_ip.find('.') != std::string::npos) {
        executor.setESP8266IP(esp_ip);
    } else {
        printf("\nModo solo análisis de datos activado.\n");
    }

    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Sistema listo. Escribe 'ayuda' para ver comandos.\n");
    printf("═══════════════════════════════════════════════════════\n");

    int queryCount = 0;
    std::string userQuery;

    // Bucle principal
    while (true) {
        if (!getUserInput(userQuery)) {
            printf("\n🔌 Cerrando sistema...\n");
            break;
        }

        queryCount++;

        // Mostrar ayuda si se solicita
        if (userQuery == "ayuda" || userQuery == "help") {
            showHelp();
            showControlHelp();
            continue;
        }
        
        // Mostrar estado actual
        if (userQuery == "estado" || userQuery == "status") {
            executor.showCurrentStatus();
            if (!csvData.empty()) {
                analyzeData(csvData, ESTADO, userQuery);
            }
            continue;
        }

        // Procesar query con NLU
        int topIntent = 0;
        float confidence = 0.0f;
        float processingTime = 0.0f;

        nluEngine.processQuery(userQuery, topIntent, confidence, processingTime);

        printf("\nIntent: %s (%.1f%% confianza)\n",
               intentNames[topIntent], confidence * 100);

        // Ejecutar comando si es tipo CONTROL
        bool commandExecuted = false;
        if (topIntent == CONTROL || topIntent == LUCES || 
            topIntent == AC || topIntent == PUERTA) {
            commandExecuted = executor.executeControl(userQuery);
        }

        // Analizar datos CSV si están disponibles
        if (!csvData.empty()) {
            analyzeData(csvData, topIntent, userQuery);
        }

        printf("\nTiempo de procesamiento NLU: %.2f ms\n", processingTime);
        
        if (commandExecuted) {
            printf("Comando ejecutado exitosamente\n");
        }
    }

    // Limpieza
    nluEngine.cleanup();

    printf("\nSesión finalizada: %d consultas procesadas\n", queryCount);
    return 0;
}

void showControlHelp() {
    printf("\n COMANDOS DE CONTROL EN TIEMPO REAL:\n\n");
    printf(" LUCES:\n");
    printf("   • encender luces / prender luces\n");
    printf("   • apagar luces\n\n");
    printf(" AIRE ACONDICIONADO:\n");
    printf("   • encender ac / activar aire acondicionado\n");
    printf("   • apagar ac\n\n");
    printf(" RIEGO:\n");
    printf("   • activar riego\n");
    printf("   • desactivar riego\n\n");
    printf(" PUERTA:\n");
    printf("   • cerrar puerta\n");
    printf("   • abrir puerta\n\n");
    printf(" ASCENSOR:\n");
    printf("   • ir a piso 1 / mover ascensor a piso 2\n");
    printf("   • bajar a pb / ir a planta baja\n\n");
    printf(" ESTADO:\n");
    printf("   • estado / status (ver todo el sistema)\n\n");
    printf("═══════════════════════════════════════════════════════\n");
}