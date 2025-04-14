import socket

def receive_files_over_wifi(port, save_folder='./received'):
    import os
    os.makedirs(save_folder, exist_ok=True)
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', port))
        s.listen()
        print(f"Listening on port {port}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            
            while True:
                # Receive file info
                data = conn.recv(1024).decode()
                if not data:
                    break
                    
                file_name, file_size = data.split(',')
                file_size = int(file_size)
                
                # Acknowledge
                conn.sendall(b"OK")
                
                # Receive file content
                file_path = os.path.join(save_folder, file_name)
                received = 0
                with open(file_path, 'wb') as f:
                    while received < file_size:
                        data = conn.recv(min(4096, file_size - received))
                        if not data:
                            break
                        f.write(data)
                        received += len(data)
                
                # Acknowledge file received
                conn.sendall(b"File received")

# Usage
receive_files_over_wifi(5000)