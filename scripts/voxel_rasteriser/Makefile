NVCC ?= nvcc
ARCH ?= 89
TARGET := siren_voxel
NVCCFLAGS := -O3 --use_fast_math -std=c++17 -arch=sm_$(ARCH)

.PHONY: all run visualize clean

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) 2000 0.001 120 1000

visualize:
	python3 visualize.py --make-video --show

clean:
	rm -f $(TARGET)
	rm -rf output
