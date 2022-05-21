# Service
import socket
import queue
import threading

# Calculation
import pyopencl as cl
import numpy as np
import cv2
import math

# Misc
from io import BytesIO
import time
import sys

M_PI=3.14159265358979323846

def build_list(radius):
    ar_len=0
    x_list=[]
    y_list=[]
    deg_list=[]
    radius_list=[]
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if ((np.sqrt(i*i+j*j) < radius+1.0) and not (i==0 and j==0)):
                x_list.append(j)
                y_list.append(i)
                deg=math.atan2(j,i)
                if deg<0.0:
                    deg+=M_PI*2
                deg_list.append(deg)
                radius_list.append(np.sqrt(i*i+j*j))
    zipped=zip(x_list, y_list, deg_list, radius_list)
    zipped=sorted(zipped, key = lambda x: (x[2], x[3]))
    return zipped

def SFEGO(cl_info, resize_ratio, execute_radius, image_data):
    ctx, queue, prg, knl_gradient, knl_integral, mf = cl_info
    
    #Resize
    height = image_data.shape[0]
    width = image_data.shape[1]
    target_height = int(height/resize_ratio)
    target_width = int(width/resize_ratio)
    input_data = cv2.resize(image_data, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    #Setup Radius
    ar_list=build_list(execute_radius)
    x_list, y_list, deg_list, radius_list=zip(*ar_list)
    list_len=len(ar_list)

    #Convert to Numpy
    np_data = np.asarray(input_data).flatten().astype(np.float32)
    np_x_list = np.asarray(x_list).astype(np.int32)
    np_y_list = np.asarray(y_list).astype(np.int32)
    np_deg_list = np.asarray(deg_list).astype(np.float32)

    #OpenCL Buffer 
    list_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_x_list)
    list_y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_y_list)
    list_deg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_deg_list)
    data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_data)
    diff = cl.Buffer(ctx, mf.READ_WRITE, np_data.nbytes)
    direct = cl.Buffer(ctx, mf.READ_WRITE, np_data.nbytes)
    result = cl.Buffer(ctx, mf.READ_WRITE, np_data.nbytes)

    #OpenCL Execution
    knl_gradient(queue, (target_width, target_height), None, data, diff, direct, list_x, list_y, list_deg, np.int32(list_len), np.int32(target_width), np.int32(target_height))
    knl_integral(queue, (target_width, target_height), None, result, diff, direct, list_x, list_y, list_deg, np.int32(list_len), np.int32(target_width), np.int32(target_height))

    #Get OpenCL Result
    np_result = np.empty_like(np_data)
    cl.enqueue_copy(queue, np_result, result)
    np_result = np_result / list_len

    list_x.release()
    list_y.release()
    list_deg.release()
    data.release()
    diff.release()
    direct.release()
    result.release()

    #Reshape to correct width, height
    SpatialFrame_result=np_result.reshape((target_height, target_width))
    
    #Resize 
    SpatialFrame_result=cv2.resize(SpatialFrame_result, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return SpatialFrame_result


def sendall(sock, data):
    BUFF_SIZE = 1048576
    length=len(data)
    #SEND_LEN=length
    length=str(length)
    sock.sendall(length.encode())
    init=sock.recv(BUFF_SIZE).decode()
    if init == "OK":
        #SEND_SIZE=0
        #OFFSET=0
        #while SEND_SIZE < SEND_LEN:
        #    if OFFSET+BUFF_SIZE > SEND_LEN:
        #        sock.send(data[OFFSET:])
        #        SEND_SIZE+=SEND_LEN-OFFSET
        #    else:
        #        sock.send(data[OFFSET:OFFSET+BUFF_SIZE])
        #        SEND_SIZE+=BUFF_SIZE
        #    OFFSET+=BUFF_SIZE
        sock.sendall(data)
    

def recvall(sock):
    BUFF_SIZE = 1048576 # 1024 KB = 1 MB
    length=sock.recv(BUFF_SIZE).decode()
    length=int(length)
    sock.sendall("OK".encode())
    data = b''
    while True:
        part = sock.recv(BUFF_SIZE)
        data += part
        #if len(part) < BUFF_SIZE:
        if len(data) >= length:
            # either 0 or end of data
            break
    return data

SFEGO_queue = queue.Queue()
SEND_queue = queue.Queue()

def SFEGO_worker(Platform_ID, Device_ID):
    print("Initial SFEGO_worker on Platfrom_ID=", Platform_ID, "Device_ID=", Device_ID)
    ctx = cl.Context([cl.get_platforms()[Platform_ID].get_devices()[Device_ID]])
    queue = cl.CommandQueue(ctx)
    prg = cl.Program(ctx, open('kernel.cl').read()).build()
    knl_gradient = prg.GMEMD_gradient
    knl_integral = prg.GMEMD_integral
    mf = cl.mem_flags
    cl_info = (ctx, queue, prg, knl_gradient, knl_integral, mf)
    while True:
        try:
            #print("SFEGO get")
            client, addr, resize_ratio, execute_radius, data = SFEGO_queue.get()
            #print("SFEGO execute")
            result = SFEGO(cl_info, resize_ratio, execute_radius, data)
            #print("SFEGO done")
            SEND_queue.put((client, addr, resize_ratio, execute_radius, result))
        except:
            print("Addr:", addr, "Failure on SFEGO_worker")
            client.close()
        finally:
            print("Addr:", addr, "SFEGO:", resize_ratio, execute_radius, "Processed on", (Platform_ID, Device_ID))

def SENDER_worker():
    while True:
        try:
            client, addr, resize_ratio, execute_radius, result = SEND_queue.get()
            f = BytesIO()
            np.savez_compressed(f, result=result)
            f.seek(0)
            out = f.read()
            #client.sendall(out)
            sendall(client, out)
            client.shutdown(1)
            client.close()
        except:
            #write error code to file
            print("Addr:", addr, "Failure on SENDER_Worker")
            client.close()
        finally:
            client.close()
            #print("Addr:", addr, "SFEGO:", resize_ratio, execute_radius, "Result Sended")
    
def Session_handler(client, addr):
    #print("Session handler started by ", addr)
    try:
        command = recvall(client)
        command = command.decode()
        #print("Addr:", addr, "Command:", command)
        if command == "SFEGO":
            #time.sleep(1.0)
            sendall(client, "WAIT".encode())
            buffer = recvall(client)
            np_data=np.load(BytesIO(buffer))
            data=np_data['data']
            resize_ratio=np_data['resize_ratio']
            execute_radius=np_data['execute_radius']
            SFEGO_queue.put((client, addr, resize_ratio, execute_radius, data))
            #print("Addr:", addr, "Queued")
        else:
            # Don't care other client.
            print("Addr:", addr, "Unknown Command:", command)
            client.close()
    except:
        print("Addr:", addr, "Disconnected")
        client.close()
        

sys.setswitchinterval(0.005)

# Variable for Socket
bind_ip = ""
bind_port = 8888

server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind((bind_ip,bind_port))
server.listen(1000)

# Variable for Worker
PLATFORMS = [(0,0,4),(0,1,8)] #(Platform_ID, Device_ID, Worker_Count)
Sender_Count = 8
Worker_thread_list = []
Sender_thread_list = []
Session_thread_list = []

# Initial Worker Thread
for PLATFROM in PLATFORMS:
    Platform_ID, Device_ID, Worker_Count = PLATFROM
    for index in range(Worker_Count):
        thread = threading.Thread(target = SFEGO_worker, args = (Platform_ID, Device_ID))
        thread.start()
        Worker_thread_list.append(thread)

# Initial Sender Thread
for index in range(Sender_Count):
    thread = threading.Thread(target = SENDER_worker)
    thread.start()
    Sender_thread_list.append(thread)

print("[*] Listening on %s:%d " % (bind_ip,bind_port))

while True:
    client, addr = server.accept()
    #print("Addr:", addr, "Connected")
    thread = threading.Thread(target = Session_handler, args = (client, addr))
    thread.start()
    Session_thread_list.append(thread)