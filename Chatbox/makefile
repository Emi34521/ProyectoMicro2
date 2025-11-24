# Makefile para Smart Home CSV Analyzer + ESP8266 Control

# Compilador
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_75 --extended-lambda
CXX_FLAGS = -std=c++14 -O3

# Librerías
LIBS = -lcurl

# Archivos objeto
OBJS = main.o nlu_engine.o nlu_kernels.o csv_handler.o data_analyzer.o \
       user_interface.o web_client.o command_executor.o

# Ejecutable
TARGET = smart_home_analyzer

# Regla principal
all: $(TARGET)

# Linkear todos los objetos
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compilar main
main.o: main.cu common.h csv_handler.h nlu_engine.h data_analyzer.h \
        user_interface.h command_executor.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu

# Compilar motor NLU
nlu_engine.o: nlu_engine.cu nlu_engine.h nlu_kernels.cuh common.h
	$(NVCC) $(NVCC_FLAGS) -c nlu_engine.cu

# Compilar kernels
nlu_kernels.o: nlu_kernels.cu nlu_kernels.cuh common.h
	$(NVCC) $(NVCC_FLAGS) -c nlu_kernels.cu

# Compilar módulos C++
csv_handler.o: csv_handler.cpp csv_handler.h
	$(NVCC) $(CXX_FLAGS) -c csv_handler.cpp

data_analyzer.o: data_analyzer.cpp data_analyzer.h csv_handler.h common.h
	$(NVCC) $(CXX_FLAGS) -c data_analyzer.cpp

user_interface.o: user_interface.cpp user_interface.h common.h
	$(NVCC) $(CXX_FLAGS) -c user_interface.cpp

# Nuevos módulos para control web
web_client.o: web_client.cpp web_client.h
	$(NVCC) $(CXX_FLAGS) -c web_client.cpp

command_executor.o: command_executor.cpp command_executor.h web_client.h common.h
	$(NVCC) $(CXX_FLAGS) -c command_executor.cpp

# Limpiar archivos generados
clean:
	rm -f $(OBJS) $(TARGET)

# Limpiar y recompilar
rebuild: clean all

# Ejecutar
run: $(TARGET)
	./$(TARGET)

# Profile con nvprof (legacy) o ncu (moderno)
profile: $(TARGET)
	@echo "Perfilando con nvprof..."
	nvprof --print-gpu-trace ./$(TARGET) 2>&1 || \
	ncu --target-processes all ./$(TARGET)

# Verificar dependencias
check-deps:
	@echo "Verificando dependencias..."
	@which nvcc > /dev/null || echo "✗ CUDA Toolkit no encontrado"
	@pkg-config --exists libcurl || echo "✗ libcurl no encontrada"
	@echo "✓ Verificación completa"

.PHONY: all clean rebuild run profile check-deps