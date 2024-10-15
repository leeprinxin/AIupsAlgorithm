# 基礎映像使用 python:3.9
FROM python:3.9

# 設置工作目錄
WORKDIR /AIups

# 更新 pip 並安裝必要的 Python 套件
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 曝露 Jupyter Lab 的預設埠
EXPOSE 8888

# 啟動 Jupyter Lab 並設置預設運行參數
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
