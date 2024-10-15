ROOT = ${CWD}

BIN_NAME = mat_mul
NVCC = nvcc
BUILD_FLAGS = 

run: build
	@./out/mat_mul

b: build
build: clean prepare
	@${NVCC} ${BUILD_FLAGS} -o ./out/${BIN_NAME} ./src/main.cu

prepare:
	@mkdir -p ./out

clean:
	@rm -rf ./out/*
