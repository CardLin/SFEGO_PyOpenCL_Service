# Spatial Frequency Extraction using Gradient-liked Operator (SFEGO) Service
## PyOpenCL Version
### Introduction
- This work is based on https://github.com/CardLin/SFEGO_PyOpenCL

- Now, I write a Server and Client that can runs on different GPU Server

### Hardware Requirement
- Require GPU on Server to execute OpenCL Kernel Code

- Recommend to use NVIDIA GPU with 1GB+ VRAM (VRAM usage is depend on Image Size)

- AMD Integrated GPU and Intel Integrated GPU can also run this project

- Although It can also run OpenCL on CPU mode but even the Intel Integrated GPU is faster than high-end CPU

### Execution
- Modify PLATFORMS = [(0,0,4),(0,1,8)] in Server.py which is (Platform_ID, Device_ID, Worker_Count)

- I have two AMD GPU on this server. [(0,0,4),(0,1,8)] means run 4 thread on (Platform_ID=0, Device_ID=0) and 8 thread on (Platform_ID=0, Device_ID=1)

- python Server.py


- Modify ServerList = [ ("127.0.0.1", 8888, 12), ("192.168.1.33", 8888, 8) ] in Client.py which is ((IP, port, ExecuteCount))

- ExecuteCount is concurrent thread that how many socket connect to specific Server, you can set this number as Worker_Count on server

- Client.py support send image to different server to increase throughput

- Modify IN_Folder for read image and OUT_Folder for save Spatial Frame

- python Client.py