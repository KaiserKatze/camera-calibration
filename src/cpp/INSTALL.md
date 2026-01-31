# 安装依赖

```bash
EIGEN_VERSION=5.0.1

LINUX_DISTROS=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
UBUNTU_CODENAME=$(lsb_release -c | awk '{NR>1}{print $2}' | tr '[:upper:]' '[:lower:]')

if [[ "$LINUX_DISTROS" == "ubuntu" ]]; then
    echo "Unsupported OS!"
    exit 1
fi

#===========================================================
# 安装 Eigen 矩阵运算 C++ 函数库

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
# @see: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

# 新建 ros 用户
sudo useradd -m -d "/home/ros" -s "/bin/bash" -c "ros" "ros"
sudo passwd ros
sudo chown -R ros:ros /home/ros
sudo chmod -aG sudo ros
sudo su - ros

# 修改编码方式
sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo export LANG=en_US.UTF-8
echo "export LANG=en_US.UTF-8" >> /home/ros/.profile

# 添加镜像源
sudo apt-get install -y curl gnupg lsb-release
if [[ $UBUNTU_CODENAME == "noble" ]]; then

    sudo apt-get install -y software-properties-common
    sudo add-apt-repository universe
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb

else

    # 清华镜像源 https://mirror.tuna.tsinghua.edu.cn/rosdistro/ros.key
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | \
        sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

fi

# 安装 ROS2
sudo apt-get update -y
sudo apt-get upgrade -y
ROS_CODENAME=""
if [ "$UBUNTU_CODENAME" -eq "noble" ]; then
    ROS_CODENAME="jazzy"
elif [ "$UBUNTU_CODENAME" -eq "jammy" ]; then
    ROS_CODENAME="humble"
else
    echo "No suitable ros desktop version found!"
    exit 1
fi
echo "Install ros-$ROS_CODENAME-desktop for Ubuntu $UBUNTU_CODENAME ..."
sudo apt-get install -y ros-$ROS_CODENAME-desktop

# 设置环境变量
echo "export ROS_CODENAME=$ROS_CODENAME" >> /home/ros/.profile
source /opt/ros/$ROS_CODENAME/setup.bash
echo "source /opt/ros/$ROS_CODENAME/setup.bash" >> /home/ros/.profile

#===========================================================
# 安装 OpenCV
sudo apt-get install -y libopencv-dev

#===========================================================
# 安装 ROS-TF2 工具（用于查看坐标变换）
sudo apt-get install -y ros-$ROS_CODENAME-tf2-tools transforms3d

#===========================================================
# 安装 USB 摄像头驱动
sudo apt-get install -y ros-$ROS_CODENAME-usb-cam

#===========================================================
# 安装 rosdep (用于自动拉取依赖)
# @see: https://docs.ros.org/en/independent/api/rosdep/html/overview.html
# @see: https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Rosdep.html
# @see: https://docs.ros.org/en/jazzy/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html#resolve-dependencies
if [[ "$LINUX_DISTROS" == "ubuntu" ]]; then
    sudo apt-get install -y python3-rosdep ||\
    { echo "Fail to install rosdep"; exit 1; }
elif [[ "$LINUX_DISTROS" == "debian" ]]; then
    sudo apt-get install -y python3-rosdep2 ||\
    { echo "Fail to install rosdep"; exit 1; }
else
    sudo pip3 install -U rosdep ||\
    sudo pip install -U rosdep ||\
    sudo easy_install -U rosdep rospkg ||\
    { echo "Fail to install rosdep"; exit 1; }
fi

ROSDEP_SOURCES_LIST_D=/etc/ros/rosdep/sources.list.d
sudo rosdep init || mkdir -p $ROSDEP_SOURCES_LIST_D/

cat > /tmp/20-default.list << EOF
# os-specific listings first
yaml https://mirror.tuna.tsinghua.edu.cn/rosdistro/rosdep/osx-homebrew.yaml osx

# generic
yaml https://mirror.tuna.tsinghua.edu.cn/rosdistro/rosdep/base.yaml
yaml https://mirror.tuna.tsinghua.edu.cn/rosdistro/rosdep/python.yaml
yaml https://mirror.tuna.tsinghua.edu.cn/rosdistro/rosdep/ruby.yaml
gbpdistro https://raw.githubusercontent.com/ros/rosdistro/master/releases/fuerte.yaml fuerte

# newer distributions (Groovy, Hydro, ...) must not be listed anymore, they are being fetched from the rosdistro index.yaml instead
EOF
sudo mv /tmp/20-default.list $ROSDEP_SOURCES_LIST_D/20-default.list

rosdep update
# 真要使用 rosdep 时，在工作空间中执行
#       rosdep install --from-paths src -y --ignore-src
#       rosdep install -i --from-path src --rosdistro $ROS_CODENAME -y

#===========================================================
# 安装 colcon (用于构建)
sudo apt-get install -y python3-colcon-common-extensions
# 真要使用 colcon 时，在工作空间中执行
#       colcon build
```


```bash
# 在执行 ros2 run 命令或 ros2 launch 命令之前，必须要准备运行环境，在 workspace 中执行以下命令
source ./install/local_setup.sh
```
