ROOT = ${CWD}

BIN_NAME = vector_sum
NVCC = nvcc
BUILD_FLAGS = -rdc=true

run: build
	@./out/${BIN_NAME}

b: build
build: clean prepare
	@${NVCC} ${BUILD_FLAGS} -o ./out/${BIN_NAME} ./src/main.cu

prepare:
	@mkdir -p ./out

clean:
	@rm -rf ./out/*
