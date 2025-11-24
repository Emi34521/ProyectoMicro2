#ifndef COMMAND_EXECUTOR_H
#define COMMAND_EXECUTOR_H

#include "common.h"
#include "web_client.h"
#include <string>
#include <vector>

// Clase para ejecutar comandos basados en la intención detectada
class CommandExecutor {
private:
    WebClient webClient;
    bool esp8266Connected;
    
    // Parsear comando específico del query
    bool parseAction(const std::string& query, std::string& action);
    bool parseDevice(const std::string& query, std::string& device);
    bool parseFloor(const std::string& query, int& floor);
    
public:
    CommandExecutor();
    
    // Configurar conexión con ESP8266
    void setESP8266IP(const std::string& ip);
    bool testConnection();
    
    // Ejecutar comando basado en intención
    bool executeCommand(int intent, const std::string& query);
    
    // Comandos directos
    bool executeControl(const std::string& query);
    
    // Mostrar estado actual
    void showCurrentStatus();
};

#endif // COMMAND_EXECUTOR_H