#include "web_client.h"
#include <iostream>

size_t WebClient::WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

WebClient::WebClient() : curl(nullptr) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
}

WebClient::~WebClient() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();
}

void WebClient::setESP8266IP(const std::string& ip) {
    esp8266_ip = ip;
}

bool WebClient::getStatus(std::string& response) {
    if (!curl) return false;
    
    std::string url = "http://" + esp8266_ip + "/api/status";
    response.clear();
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "Error HTTP: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    
    return true;
}

bool WebClient::sendControl(const std::string& device, int value) {
    if (!curl) return false;
    
    std::string url = "http://" + esp8266_ip + "/api/control?dev=" + 
                      device + "&val=" + std::to_string(value);
    std::string response;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        std::cerr << "Error enviando comando: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    
    return true;
}

bool WebClient::toggleLights(bool on) {
    return sendControl("luces", on ? 1 : 0);
}

bool WebClient::toggleAC(bool on) {
    return sendControl("ac", on ? 1 : 0);
}

bool WebClient::toggleIrrigation(bool on) {
    return sendControl("riego", on ? 1 : 0);
}

bool WebClient::toggleDoor(bool closed) {
    return sendControl("puerta", closed ? 1 : 0);
}

bool WebClient::setElevatorFloor(int floor) {
    if (floor < 0 || floor > 4) return false;
    return sendControl("ascensor", floor);
}

bool WebClient::enableManualMode(const std::string& device, bool enable) {
    // Activar modo manual enviando cualquier comando
    // El ESP8266 ya maneja el timeout automáticamente
    return true;
}