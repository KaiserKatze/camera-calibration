# 编译器和编译选项
CXX = g++
CXXFLAGS = -std=gnu++17 -O2 -lgsl -lgslcblas -lm -I/usr/local/include/eigen3
TARGET = SimulateCamera

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): SimulateCamera.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 清理生成的文件
clean:
	rm -f $(TARGET)

# 重新编译
rebuild: clean all

# 伪目标声明
.PHONY: all clean rebuild
