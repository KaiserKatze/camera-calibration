# 编译器
CXX = g++

# 编译选项 (包含路径、优化级别、C++标准)
CXXFLAGS = -std=gnu++17 -O2 -I/usr/local/include/eigen3

# 链接选项 (库文件)
LDLIBS = -lgsl -lgslcblas -lm

TARGET = SimulateCamera

# 默认目标
all: $(TARGET)

# 编译目标
$(TARGET): SimulateCamera.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

# 清理生成的文件
clean:
	rm -f $(TARGET)

# 重新编译
rebuild: clean all

# 伪目标声明
.PHONY: all clean rebuild
