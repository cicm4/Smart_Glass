import bluetooth

_SPP_UUID = "00001101-0000-1000-8000-00805F9B34FB"

def open_rfcomm(advertise_name: str = "SmartGlassService", backlog: int = 1):
    server = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server.bind(("", bluetooth.PORT_ANY))
    server.listen(backlog)

    channel = server.getsockname()[1]
    bluetooth.advertise_service(
        server, advertise_name,
        service_id=_SPP_UUID,
        service_classes=[_SPP_UUID, bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE],
    )
    print(f"[BT] RFCOMM server on channel {channel} – waiting for host…")

    client, info = server.accept()   # blocks here
    print(f"[BT] Connected: {info[0]}")
    return client
