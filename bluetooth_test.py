import bluetooth, time

PI_MAC = "B8:27:EB:C2:A5:45"   # ← Pi’s address
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((PI_MAC, 1))      # same channel as server

for i in range(3):
    msg = f"ping {i}\n"
    sock.send(msg.encode())
    print("PC sent:", msg.strip())
    reply = sock.recv(1024)
    print("PC got :", reply.decode().strip())
    time.sleep(1)
sock.close()
