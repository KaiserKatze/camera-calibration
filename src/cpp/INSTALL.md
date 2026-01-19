# 安装依赖

```bash
EIGEN_VERSION=5.0.1

apt-get update -y
apt-get upgrade -y
apt-get install -y \
    cmake build-essentials \
    libgsl-dev \
    python3 python3-pip python3-dev python3-venv \
    python3-numpy python3-pandas python3-matplotlib \
    python3-opencv

cd /opt
wget https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.bz2
tar -xjvf eigen-$EIGEN_VERSION.tar.bz2
cd eigen-$EIGEN_VERSION
mkdir build
cd build
cmake ..
make install

#===========================================================
# 安装 ROS

# 新建 ros 用户
sudo useradd -m -d "/home/ros" -s "/bin/bash" --comment "pseudo-user" "ros"
sudo passwd ros
sudo chown -R ros:ros /home/ros
sudo chmod -aG sudo ros
sudo su - ros

# 修改编码方式
sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo export LANG=en_US.UTF-8
echo "export LANG=en_US.UTF-8">> /home/ros/.profile

# 添加镜像源
apt-get install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS2
sudo apt-get update -y
sudo apt-get upgrade -y
UBUNTU_CODENAME=$(lsb_release -c | awk '{NR>1}{print $2}')
ROS_VERSION=""
if [ "$UBUNTU_CODENAME" -eq "noble" ]; then
    ROS_VERSION="jazzy"
elif [ "$UBUNTU_CODENAME" -eq "jammy" ]; then
    ROS_VERSION="humble"
else
    echo "No suitable ros desktop version found!"
    exit 1
fi
echo "Install ros-$ROS_VERSION-desktop for Ubuntu $UBUNTU_CODENAME ..."
sudo apt-get install -y ros-$ROS_VERSION-desktop

# 设置环境变量
source /opt/ros/$ROS_VERSION/setup.bash
echo "source /opt/ros/$ROS_VERSION/setup.bash" >> /home/ros/.profile
# 测试1（消息的发布和订阅）
ros2 run demo_nodes_cpp talker          # 启动一个数据发布者节点
ros2 run demo_nodes_py listener         # 启动一个数据订阅者节点
# 如果“Hello World”字符串在两个终端中正常传输，说明通信系统没有问题

# 测试2（小海龟仿真器）
ros2 run turtlesim turtlesim_node       # 启动一个蓝色背景的海龟仿真器
ros2 run turtlesim turtle_teleop_key    # 启动一个键盘控制节点
# 在终端中点击键盘上的“上下左右”按键，就可以控制小海龟运动啦
```
