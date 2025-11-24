#include "command_executor.h"
#include <iostream>
#include <algorithm>
#include <cctype>
#include <sstream>

CommandExecutor::CommandExecutor() : esp8266Connected(false) {}

void CommandExecutor::setESP8266IP(const std::string& ip) {
    webClient.setESP8266IP(ip);
    esp8266Connected = testConnection();
}

bool CommandExecutor::testConnection() {
    std::string response;
    bool success = webClient.getStatus(response);
    
    if (success) {
        printf("\n✓ Conexión establecida con ESP8266\n");
    } else {
        printf("\n✗ No se pudo conectar con ESP8266\n");
        printf("  Verifica que el dispositivo esté encendido y conectado\n");
    }
    
    return success;
}

bool CommandExecutor::parseAction(const std::string& query, std::string& action) {
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
    
    if (lowerQuery.find("encender") != std::string::npos ||
        lowerQuery.find("activar") != std::string::npos ||
        lowerQuery.find("prender") != std::string::npos) {
        action = "on";
        return true;
    }
    
    if (lowerQuery.find("apagar") != std::string::npos ||
        lowerQuery.find("desactivar") != std::string::npos) {
        action = "off";
        return true;
    }
    
    if (lowerQuery.find("cerrar") != std::string::npos) {
        action = "close";
        return true;
    }
    
    if (lowerQuery.find("abrir") != std::string::npos) {
        action = "open";
        return true;
    }
    
    return false;
}

bool CommandExecutor::parseFloor(const std::string& query, int& floor) {
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
    
    // Buscar "pb" o "planta baja"
    if (lowerQuery.find("pb") != std::string::npos ||
        lowerQuery.find("planta baja") != std::string::npos) {
        floor = 0;
        return true;
    }
    
    // Buscar números del 1-4
    for (int i = 1; i <= 4; i++) {
        std::string pisoStr = "piso " + std::to_string(i);
        if (lowerQuery.find(pisoStr) != std::string::npos) {
            floor = i;
            return true;
        }
    }
    
    return false;
}

bool CommandExecutor::executeCommand(int intent, const std::string& query) {
    if (!esp8266Connected) {
        printf("\n⚠️  ESP8266 no conectado. Mostrando solo análisis de datos.\n");
        return false;
    }
    
    std::string action;
    int floor;
    bool success = false;
    
    switch(intent) {
        case LUCES:
            if (parseAction(query, action)) {
                success = webClient.toggleLights(action == "on");
                if (success) {
                    printf("\n✓ Luces %s\n", action == "on" ? "encendidas" : "apagadas");
                }
            }
            break;
            
        case AC:
            if (parseAction(query, action)) {
                success = webClient.toggleAC(action == "on");
                if (success) {
                    printf("\n✓ Aire acondicionado %s\n", action == "on" ? "activado" : "desactivado");
                }
            }
            break;
            
        case PUERTA:
            if (parseAction(query, action)) {
                success = webClient.toggleDoor(action == "close");
                if (success) {
                    printf("\n✓ Puerta %s\n", action == "close" ? "cerrada" : "abierta");
                }
            }
            break;
            
        case ESTADO:
            showCurrentStatus();
            success = true;
            break;
            
        default:
            // No es un comando de control, solo análisis
            break;
    }
    
    return success;
}

bool CommandExecutor::executeControl(const std::string& query) {
    if (!esp8266Connected) {
        printf("\n⚠️  ESP8266 no conectado\n");
        return false;
    }
    
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
    std::string action;
    int floor;
    bool success = false;
    
    // Detectar dispositivo y acción
    if (lowerQuery.find("luz") != std::string::npos || 
        lowerQuery.find("luces") != std::string::npos) {
        if (parseAction(query, action)) {
            success = webClient.toggleLights(action == "on");
            if (success) {
                printf("\n✓ Luces %s\n", action == "on" ? "encendidas" : "apagadas");
            }
        }
    }
    else if (lowerQuery.find("ac") != std::string::npos ||
             lowerQuery.find("aire") != std::string::npos ||
             lowerQuery.find("acondicionado") != std::string::npos) {
        if (parseAction(query, action)) {
            success = webClient.toggleAC(action == "on");
            if (success) {
                printf("\n✓ A/C %s\n", action == "on" ? "activado" : "desactivado");
            }
        }
    }
    else if (lowerQuery.find("riego") != std::string::npos) {
        if (parseAction(query, action)) {
            success = webClient.toggleIrrigation(action == "on");
            if (success) {
                printf("\n✓ Riego %s\n", action == "on" ? "activado" : "desactivado");
            }
        }
    }
    else if (lowerQuery.find("puerta") != std::string::npos) {
        if (parseAction(query, action)) {
            success = webClient.toggleDoor(action == "close");
            if (success) {
                printf("\n✓ Puerta %s\n", action == "close" ? "cerrada" : "abierta");
            }
        }
    }
    else if (lowerQuery.find("ascensor") != std::string::npos ||
             lowerQuery.find("elevador") != std::string::npos ||
             lowerQuery.find("piso") != std::string::npos) {
        if (parseFloor(query, floor)) {
            success = webClient.setElevatorFloor(floor);
            if (success) {
                printf("\n✓ Ascensor movido a %s\n", 
                       floor == 0 ? "PB" : ("Piso " + std::to_string(floor)).c_str());
            }
        }
    }
    
    if (!success && !action.empty()) {
        printf("\n✗ No se pudo ejecutar el comando\n");
    }
    
    return success;
}

void CommandExecutor::showCurrentStatus() {
    if (!esp8266Connected) {
        printf("\n⚠️  ESP8266 no conectado\n");
        return;
    }
    
    std::string response;
    if (!webClient.getStatus(response)) {
        printf("\n✗ Error obteniendo estado\n");
        return;
    }
    
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│ ESTADO ACTUAL DEL SISTEMA                           │\n");
    printf("└─────────────────────────────────────────────────────┘\n");
    
    // Parsear JSON simple (búsqueda de strings)
    auto findValue = [&](const std::string& key) -> std::string {
        size_t pos = response.find("\"" + key + "\":");
        if (pos == std::string::npos) return "N/A";
        
        pos += key.length() + 3;
        size_t endPos = response.find_first_of(",}", pos);
        if (endPos == std::string::npos) return "N/A";
        
        std::string value = response.substr(pos, endPos - pos);
        // Remover comillas si existen
        if (value.front() == '"') value = value.substr(1, value.length() - 2);
        
        return value;
    };
    
    printf(" 💡 Luces: %s\n", findValue("luces") == "true" ? "ENCENDIDAS" : "Apagadas");
    printf(" ❄️  A/C: %s (Temp: %s°C)\n", 
           findValue("ac") == "true" ? "ACTIVO" : "Apagado",
           findValue("temp").c_str());
    printf(" 💧 Riego: %s (Hum: %s%%)\n", 
           findValue("riego") == "true" ? "ACTIVO" : "Apagado",
           findValue("hum").c_str());
    printf(" 🚪 Puerta: %s\n", findValue("puerta") == "true" ? "Cerrada" : "Abierta");
    printf(" 🛗 Ascensor: %s\n", findValue("piso").c_str());
    printf("\n ⚡ Consumo Total: %s Wh\n", findValue("consumo_total").c_str());
    printf("─────────────────────────────────────────────────────\n");
}