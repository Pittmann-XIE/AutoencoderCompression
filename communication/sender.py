import os
import socket
import time
from pathlib import Path

def send_files_over_wifi(host, port, folder_path):
    """Send .pth files with robust acknowledgment protocol"""
    files = sorted(Path(folder_path).glob('*.pth'))
    if not files:
        print("No .pth files found in the directory")
        return

    total_time = 0
    successful_transfers = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            print(f"Connected to {host}:{port}")

            for file in files:
                file_size = os.path.getsize(file)
                file_name = file.name
                print(f"\nPreparing to send {file_name} ({file_size/1024:.2f} KB)")

                # Start timer
                start_time = time.perf_counter()

                # 1. Send metadata (filename, size)
                metadata = f"{file_name},{file_size}"
                s.sendall(metadata.encode())
                print("Sent metadata, waiting for ACK...")

                # 2. Wait for metadata acknowledgment
                ack = s.recv(1024).decode().strip()
                if ack != "METADATA_ACK":
                    print(f"Error: Bad metadata ACK ({ack}), skipping file")
                    continue

                # 3. Send file data in chunks
                bytes_sent = 0
                with open(file, 'rb') as f:
                    while bytes_sent < file_size:
                        chunk = f.read(4096)
                        s.sendall(chunk)
                        bytes_sent += len(chunk)
                        print(f"\rProgress: {bytes_sent/file_size:.1%}", end='')

                # 4. Wait for final acknowledgment
                final_ack = s.recv(1024).decode().strip()
                end_time = time.perf_counter()

                if final_ack != "FILE_RECEIVED":
                    print(f"\nError: Bad final ACK ({final_ack}), transfer may have failed")
                    continue

                transfer_time = end_time - start_time
                total_time += transfer_time
                successful_transfers += 1
                print(f"\nTransfer successful in {transfer_time:.4f} seconds")

        except ConnectionError as e:
            print(f"\nConnection failed: {e}")
        except Exception as e:
            print(f"\nError occurred: {e}")

    print("\n=== Transfer Summary ===")
    print(f"Files attempted: {len(files)}")
    print(f"Files successfully sent: {successful_transfers}")
    if successful_transfers > 0:
        print(f"Total transfer time: {total_time:.4f} seconds")
        print(f"Average time per file: {total_time/successful_transfers:.4f} seconds")

# Usage
send_files_over_wifi('192.168.1.100', 5000, './binary')