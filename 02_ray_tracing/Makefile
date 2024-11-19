ROOT = ${CWD}

BIN_NAME = ray_tracing
NVCC = nvcc
BUILD_FLAGS = 

# img: build
# 	@./out/${BIN_NAME} > image.ppm

run: build
	@./out/${BIN_NAME}

b: build
build: clean prepare
	@${NVCC} ${BUILD_FLAGS} -o ./out/${BIN_NAME} ./src/main.cu ./src/*/*.cpp

prepare:
	@mkdir -p ./out

clean:
	@rm -rf ./out/*
