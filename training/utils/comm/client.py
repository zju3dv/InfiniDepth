import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import socket
import time


class Client:
    def __init__(self, servername: str, port: int, encode_type="utf-8"):
        self.encode_type = encode_type
        self.servername = servername
        self.port = port

    def connect(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.client.connect((self.servername, self.port))
                print(f"Successfully connect to {self.servername}:{self.port}")
                break
            except ConnectionRefusedError:
                print(f"Faild to connect {self.servername}:{self.port} !")
                time.sleep(1)
                pass

    def receive_data(self):
        l = self.client.recv(4)
        l = int(np.frombuffer(l, dtype=np.float32)[0])
        print(f"Receive {l} data!")
        if l * 4 > 1024:
            total_data = self.client.recv(1024)
            left = l * 4 - 1024
            while True:
                if left > 1024:
                    total_data += self.client.recv(1024)
                else:
                    total_data += self.client.recv(left)
                if len(total_data) == l * 4:
                    print(f"Get {len(total_data)} now!")
                    break
                left -= 1024
        else:
            total_data = self.client.recv(l * 4)
            print(f"Get {len(total_data)} now!")
        total_data = np.frombuffer(total_data, dtype=np.float32)
        return total_data

    def send_data(self, pose):
        self.connect()
        if isinstance(pose, torch.Tensor):
            pose = pose.detach().cpu().numpy()
        msg = pose.reshape(-1)
        msg = msg.astype(np.float32)
        l = msg.shape[0]
        print(l)
        l = np.array(l, dtype=np.int32)
        l = l.tobytes()
        self.client.send(l)
        self.client.send(msg.tobytes())

        refine_pose = self.receive_data()
        self.client.close()
        return refine_pose
