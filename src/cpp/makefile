# 编译器
CXX = g++

# 公共编译选项 (C++标准, Include路径)
COMMON_FLAGS = -std=gnu++17 -I/usr/local/include/eigen3

# 链接选项 (库文件)
LDLIBS = -lgsl -lgslcblas -lm

# 目标文件名
TARGET_RELEASE = SimulateCamera
TARGET_DEBUG = $(TARGET_RELEASE)_debug

# Release 版编译参数 (开启 O2 优化)
CXXFLAGS_RELEASE = $(COMMON_FLAGS) -O2

# Debug 版编译参数 (关闭优化 -O0, 开启调试符号 -g)
CXXFLAGS_DEBUG = $(COMMON_FLAGS) -O0 -g

# 默认编译目标 (Release)
all: $(TARGET_RELEASE)

# Release 编译规则
$(TARGET_RELEASE): SimulateCamera.cpp
	$(CXX) $(CXXFLAGS_RELEASE) $< -o $@ $(LDLIBS)

# Debug 编译目标
debug: $(TARGET_DEBUG)

# Debug 编译规则
$(TARGET_DEBUG): SimulateCamera.cpp
	$(CXX) $(CXXFLAGS_DEBUG) $< -o $@ $(LDLIBS)

# 清理生成的文件 (同时清理 release 和 debug 版本)
clean:
	rm -f $(TARGET_RELEASE) $(TARGET_DEBUG)

# 重新编译
rebuild: clean all

# 伪目标声明
.PHONY: all clean rebuild debug
