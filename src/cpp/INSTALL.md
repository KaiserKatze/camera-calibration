# 安装依赖

```bash
EIGEN_VERSION=5.0.1

apt-get update -y
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
```
