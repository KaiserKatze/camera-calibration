# 安装依赖

```bash
EIGEN_VERSION=5.0.1

apt-get update -y
apt-get install -y cmake build-essentials libgsl-dev

cd /opt
wget https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VERSION/eigen-$EIGEN_VERSION.tar.bz2
tar -xjvf eigen-$EIGEN_VERSION.tar.bz2
cd eigen-$EIGEN_VERSION
mkdir build
cd build
cmake ..
make install
```
