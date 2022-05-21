# Service
import socket
import queue
import threading

# Image Library
import cv2
import numpy as np

# Misc
from io import BytesIO
import time
import sys
import os

# Input Folder
IN_Folder="Input"
OUT_Folder="Output"

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

IMAGE_queue = queue.Queue()
FRAME_queue = queue.Queue()
SPATIAL_FRAME_queue = queue.Queue()
ANSWER_queue = queue.Queue()

def ServiceWorker():
    #print("ServiceWorker")
    while True:
        try:
            filename = IMAGE_queue.get()
            image=cv2.imread(filename)
            height = image.shape[0]
            width = image.shape[1]
            channel = image.shape[2]
            for ch in range(channel):
                frame=image[:,:,ch]
                FRAME_queue.put((filename, ch, frame))
        except:
            print("ServiceWorker Error on", filename)

def JobWorker():
    #print("JobWorker")
    resize_ratios=[]
    execute_radiuss=[]
    effective_radiuss=[]
    file = open('default_radius')
    for line in file:
        fields = line.strip().split()
        resize_ratio=float(fields[0])
        execute_radius=int(fields[1])
        resize_ratios.append(resize_ratio)
        execute_radiuss.append(execute_radius)
        
    while True:
        try:
            filename, ch, frame = FRAME_queue.get()
            for resize_ratio, execute_radius in zip(resize_ratios, execute_radiuss):
                SPATIAL_FRAME_queue.put((filename, resize_ratio, execute_radius, ch, frame))
        except:
            print("JobWorker Error on", filename, "with ch:", ch)
            

def ExecuteWorker(addr, port):
    #print("ExecuteWorker")
    #s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            index = 0
            filename, resize_ratio, execute_radius, ch, frame = SPATIAL_FRAME_queue.get()

            index = 1
            # Socket Connect
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((addr, port))
            
            start_time=time.time()
            
            index = 2
            # Send Command
            cmd = "SFEGO"
            sendall(s, cmd.encode())
            
            # Recv WAIT
            cmd=recvall(s).decode()
            
            index = 3
            # Send Frame and Radius
            f = BytesIO()
            np.savez_compressed(f, data=frame, resize_ratio=resize_ratio, execute_radius=execute_radius)
            f.seek(0)
            out = f.read()
            sendall(s, out)
            
            index = 4
            # Receive Result
            buffer = recvall(s)
            np_data=np.load(BytesIO(buffer))
            result=np_data['result']
            
            index = 5
            # Socket Close
            s.close()

            end_time=time.time()
            used_time=end_time-start_time

            index = 6
            ANSWER_queue.put((filename, resize_ratio, execute_radius, ch, result))
            print("Addr:", (addr, port), filename, "SFEGO:",resize_ratio, execute_radius, "on Channel:", ch, "Used Time:", used_time)
        except:
            print("ExecuteWorker Error on addr:", addr, "port:", port, "index:", index)
            SPATIAL_FRAME_queue.put((filename, resize_ratio, execute_radius, ch, frame))
            time.sleep(0.01)

Answer = [{}, {}, {}]
SaveCounts = []
def SaveWorker(index):
    #print("SaveWorker")
    while True:
        try:
            filename, resize_ratio, execute_radius, ch, result = ANSWER_queue.get()
            effective_radius=resize_ratio*execute_radius
            target_filename = filename+"_SFEGO_SpatialFrame_"+str(round(effective_radius, 2))+"("+str(resize_ratio)+"x"+str(execute_radius)+").png"
            
            result_min=np.min(result)
            result_max=np.max(result)
            result=255*(result-result_min)/(result_max-result_min)
            result=result.astype(np.uint8)
            
            Answer[ch][target_filename] = result
            End=1
            for ch in range(3):
                if target_filename not in Answer[ch]:
                    End=0
            if End:
                B=Answer[0][target_filename]
                G=Answer[1][target_filename]
                R=Answer[2][target_filename]
                ColorSpatialFrame = cv2.merge((B,G,R))
                filename = os.path.split(filename)[-1]
                output_filename = filename+"_SFEGO_ColorSpatialFrame_"+str(round(effective_radius, 2))+"("+str(resize_ratio)+"x"+str(execute_radius)+").png"
                output_filepath = os.path.join(OUT_Folder, output_filename)
                cv2.imwrite(output_filepath, ColorSpatialFrame)
                #print(output_filepath)
                SaveCounts[index]+=1
        except:
            print("SaveWorker Error on", filename)


# Set Global Interpreter Lock (GIL) switching frequency
sys.setswitchinterval(0.005)

# Set Server list (ExecuteCount = Number of concurrent client thread created)
ServerList = [ ("127.0.0.1", 8888, 12), ("192.168.1.33", 8888, 8) ] #(IP, port, ExecuteCount)

# Service Thread: concurrent thread that open the image and submit job
ServiceCount = 4

# Job Thread: concurrent thread that submit job for each frame (resize_ratio, execute_radius)
JobCount = 4

# Save Thread: concurrent thread that save the result to disk
SaveCount = 4

Service_thread_list = []
for index in range(ServiceCount):
    thread = threading.Thread(target = ServiceWorker)
    thread.start()
    Service_thread_list.append(thread)

Job_thread_list = []
for index in range(JobCount):
    thread = threading.Thread(target = JobWorker)
    thread.start()
    Job_thread_list.append(thread)

Execute_thread_list = []
for Server in ServerList:
    addr, port, ExecuteCount = Server
    for index in range(ExecuteCount):
        thread = threading.Thread(target = ExecuteWorker, args = (addr, port))
        thread.start()
        Execute_thread_list.append(thread)

Save_thread_list = []
for index in range(SaveCount):
    SaveCounts.append(0)
    thread = threading.Thread(target = SaveWorker, args = (index,))
    thread.start()
    Save_thread_list.append(thread)

#for i in range(1):
#IMAGE_queue.put("lena.png")

f = []
for (dirpath, dirnames, filenames) in os.walk(IN_Folder):
    f.extend(filenames)
    break

for filename in f:
    filepath=os.path.join(IN_Folder, filename)
    IMAGE_queue.put(filepath)

FileCount=len(f)
SpatialFrame_Count=0
file = open('default_radius')
for line in file:
    SpatialFrame_Count+=1
TotalCount=SpatialFrame_Count*FileCount

print("FileCount:", FileCount)
print("TotalCount:", TotalCount)

start_time = time.time()

time.sleep(0.001)

while True:
    #print(SaveCounts)
    SaveCount= sum(SaveCounts)
    end_time = time.time()
    used_time=end_time-start_time
    avg_fps=SaveCount/used_time
    print("Spatial Frame Progress:", SaveCount, "/", TotalCount, "Avg FPS:", avg_fps)
    if SaveCount == TotalCount:
        break
    time.sleep(1.0) 

print("All Calculation Finised... kill python manually.")


'''
HOST = '127.0.0.1'
PORT = 8888

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

cmd = "SFEGO"
sendall(s, cmd.encode())

cmd=recvall(s).decode()

filename = sys.argv[1]
img=cv2.imread(filename)
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = gray.astype(np.float32)
resize_ratio=float(0.25)
execute_radius=int(4)

f = BytesIO()
np.savez_compressed(f, data=gray, resize_ratio=resize_ratio, execute_radius=execute_radius)
f.seek(0)
out = f.read()
sendall(s, out)

buffer = recvall(s)

np_data=np.load(BytesIO(buffer))
result_gray=np_data['result']
result_min=np.min(result_gray)
result_max=np.max(result_gray)
print(result_min, result_max)

#Real Amplitude
output_gray=255*(result_gray-result_min)/(result_max-result_min)
output_gray=output_gray.astype(np.uint8)
cv2.imshow('Result ', output_gray)
cv2.waitKey(0)
'''