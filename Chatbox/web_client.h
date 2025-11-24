#ifndef WEB_CLIENT_H
#define WEB_CLIENT_H

#include <string>
#include <curl/curl.h>

// Clase para comunicación HTTP con el ESP8266
class WebClient {
private:
    std::string esp8266_ip;
    CURL* curl;
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);
    
public:
    WebClient();
    ~WebClient();
    
    // Configurar IP del ESP8266
    void setESP8266IP(const std::string& ip);
    
    // Obtener estado actual de la casa
    bool getStatus(std::string& response);
    
    // Enviar comando de control
    bool sendControl(const std::string& device, int value);
    
    // Comandos específicos
    bool toggleLights(bool on);
    bool toggleAC(bool on);
    bool toggleIrrigation(bool on);
    bool toggleDoor(bool closed);
    bool setElevatorFloor(int floor);
    bool enableManualMode(const std::string& device, bool enable);
};

#endif // WEB_CLIENT_H