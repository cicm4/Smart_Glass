#!/usr/bin/env python3
"""
bt_pair_auto.py – tiny Bluetooth agent for head-less pairing
• Press GPIO-17 momentary button → Pi becomes discoverable & pairable for 2 min
• Any device that pairs is automatically marked Trusted so it can reconnect
"""
import dbus, dbus.mainloop.glib, gi, RPi.GPIO as GPIO, subprocess, time, threading
gi.require_version('GLib', '2.0')
from gi.repository import GLib

BUTTON = 17          # BCM pin number
PAIR_WINDOW = 120    # seconds

# ─── Agent that auto-trusts every new bond ──────────────────────────────
AGENT_PATH = "/auto/agent"

class NoIOAgent(dbus.service.Object):
    @dbus.service.method('org.bluez.Agent1', in_signature='o', out_signature='')
    def AuthorizeService(self, dev, uuid):  self._trust(dev)

    @dbus.service.method('org.bluez.Agent1', in_signature='o', out_signature='')
    def RequestAuthorization(self, dev):    self._trust(dev)

    @dbus.service.method('org.bluez.Agent1', in_signature='o', out_signature='')
    def RequestConfirmation(self, dev, passkey):  self._trust(dev)

    def _trust(self, dev):
        props = dbus.Interface(bus.get_object('org.bluez', dev),
                               'org.freedesktop.DBus.Properties')
        props.Set('org.bluez.Device1', 'Trusted', True)
        print('[BT] Trusted', dev)

dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
bus = dbus.SystemBus()
manager = dbus.Interface(bus.get_object('org.bluez', '/org/bluez'),
                         'org.bluez.AgentManager1')
agent = NoIOAgent(bus, AGENT_PATH)
manager.RegisterAgent(AGENT_PATH, 'NoInputNoOutput')
manager.RequestDefaultAgent(AGENT_PATH)
print('[BT] Agent ready – press the button to pair')

# ─── Helper that toggles discoverable/pairable ──────────────────────────
def _adapter_ctl(*btctl_args):
    subprocess.run(['bluetoothctl', *btctl_args], check=True)

def _enter_pair_mode():
    print('[BT] Discoverable for', PAIR_WINDOW, 's')
    _adapter_ctl('discoverable', 'on')
    _adapter_ctl('pairable', 'on')
    time.sleep(PAIR_WINDOW)
    _adapter_ctl('discoverable', 'off')
    _adapter_ctl('pairable', 'off')
    print('[BT] Pair window closed')

# ─── GPIO button IRQ ────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(BUTTON, GPIO.FALLING, callback=lambda ch:
    threading.Thread(target=_enter_pair_mode, daemon=True).start(), bouncetime=300)

try:
    GLib.MainLoop().run()
finally:
    GPIO.cleanup()
