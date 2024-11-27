# 基礎映像使用 nvcr.io/nvidia/pytorch:23.11-py3
FROM nvcr.io/nvidia/pytorch:23.11-py3

# 設置工作目錄
WORKDIR /AIups

# 安裝必要的系統套件
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    binutils

# 更新 pip 並安裝必要的 Python 套件
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 曝露 Jupyter Lab 的預設埠
EXPOSE 8888

# 啟動 Jupyter Lab 並設置預設運行參數
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
