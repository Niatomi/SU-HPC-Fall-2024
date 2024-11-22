ROOT = ${CWD}

BIN_NAME = ray_tracing
NVCC = nvcc
BUILD_FLAGS = --diag-suppress 177

SRCS = ./src/main.cu
INCS = ./src/vec3.h ./src/ray.h ./src/hitable.h ./src/hitable_list.h ./src/sphere.h ./src/camera.h ./src/material.h ./src/image.h ./src/EasyBMP/*.cpp ./src/light.h

run: build
	@./out/${BIN_NAME}

b: build
build: clean prepare  compile $(INCS)
	@$(NVCC) ${BUILD_FLAGS} -o ./out/${BIN_NAME} ./out/${BIN_NAME}.o

compile: $(SRCS)
	@$(NVCC) ${BUILD_FLAGS} -o ./out/${BIN_NAME}.o -c $(SRCS)

draft: clean prepare
	@${NVCC} ${BUILD_FLAGS} -o ./out/${BIN_NAME} ./draft/main.cu ./draft/*/*.cpp  ./src/*.cpp &&\
	./out/${BIN_NAME}

prepare:
	@mkdir -p ./out

clean:
	@rm -rf ./out/* &&\
	rm -rf ./assets

