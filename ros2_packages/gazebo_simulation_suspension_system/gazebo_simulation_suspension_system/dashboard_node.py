#!/bin/env python3
"""
dashboard_node.py
=================
Live ASCII dashboard in the terminal showing:
  - Coil current (bar graph)
  - Damping coefficient C(t)
  - Piston position (visual indicator)
  - Sprung mass acceleration
  - Road input
  - Mode indicator

Subscribes: /damper/status  (Float32MultiArray)
  index 0: current I [A]
  index 1: C(t) [N·s/m]
  index 2: F_lorentz [N]
  index 3: zs [m]
  index 4: ddz_s [m/s²]
  index 5: zr [m]
  index 6: piston_pos [m]
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import os, sys, math


def bar(val, maxval, width=30, fill='█', empty='░'):
    frac = max(0.0, min(1.0, val / maxval))
    filled = int(frac * width)
    return fill * filled + empty * (width - filled)


def signed_bar(val, maxval, width=20):
    half = width // 2
    frac = max(-1.0, min(1.0, val / maxval))
    if frac >= 0:
        filled = int(frac * half)
        return '░' * half + '█' * filled + '░' * (half - filled)
    else:
        filled = int(-frac * half)
        return '░' * (half - filled) + '█' * filled + '░' * half


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        self.data = [0.0] * 7
        self.frame = 0
        self.create_subscription(
            Float32MultiArray, '/damper/status', self.status_cb, 10)
        self.create_timer(0.1, self.render)   # 10 Hz redraw

    def status_cb(self, msg):
        self.data = list(msg.data) + [0.0] * (7 - len(msg.data))

    def render(self):
        self.frame += 1
        I     = self.data[0]   # current A
        C     = self.data[1]   # damping N·s/m
        F_L   = self.data[2]   # Lorentz force N
        zs    = self.data[3]   # sprung mass pos m
        ddz_s = self.data[4]   # sprung accel m/s²
        zr    = self.data[5]   # road input m
        pos   = self.data[6]   # piston pos m

        # Colour codes
        R  = '\033[91m'; G = '\033[92m'; Y = '\033[93m'
        B  = '\033[94m'; M = '\033[95m'; C_= '\033[96m'
        W  = '\033[97m'; DIM = '\033[2m'; RST = '\033[0m'; BOLD = '\033[1m'

        # Current colour (green→yellow→red)
        if I < 5:   ci = G
        elif I < 10: ci = Y
        else:        ci = R

        # Piston position visual (vertical strip)
        pct = (pos + 0.09) / 0.18   # normalize 0..1
        prow = int((1.0 - pct) * 18)

        os.system('clear')
        print(f"\n{BOLD}{W}  ╔══════════════════════════════════════════════════════╗{RST}")
        print(f"{BOLD}{W}  ║    ELECTROMAGNETIC DAMPER — LIVE DASHBOARD           ║{RST}")
        print(f"{BOLD}{W}  ║    NIT Srinagar — Adaptive Suspension Project        ║{RST}")
        print(f"{BOLD}{W}  ╚══════════════════════════════════════════════════════╝{RST}\n")

        print(f"  {DIM}{'─'*56}{RST}")
        print(f"  {BOLD}COIL CURRENT  I(t){RST}")
        print(f"  {ci}{bar(I, 15.0, 40)}{RST}  {ci}{BOLD}{I:6.2f} A{RST}  / 15.0 A max")
        print()

        print(f"  {BOLD}DAMPING COEFF  C(t) = C₀ + k·I{RST}")
        print(f"  {B}{bar(C, 2000.0, 40)}{RST}  {B}{BOLD}{C:7.0f} N·s/m{RST}")
        print(f"  {DIM}  Passive C₀ = 800 N·s/m   Max = ~2000 N·s/m{RST}")
        print()

        print(f"  {BOLD}LORENTZ FORCE  F = B·I·L{RST}")
        print(f"  {M}{bar(abs(F_L), 20.0, 40)}{RST}  {M}{BOLD}{F_L:6.1f} N{RST}")
        print()

        print(f"  {BOLD}PISTON POSITION{RST}  [{DIM}−90mm{RST} ←——— 0 ———→ {DIM}+90mm{RST}]")
        print(f"  {C_}{signed_bar(pos, 0.09, 40)}{RST}  {C_}{BOLD}{pos*1000:+6.1f} mm{RST}")
        print()

        print(f"  {BOLD}ROAD EXCITATION  zr(t){RST}")
        print(f"  {Y}{signed_bar(zr, 0.05, 40)}{RST}  {Y}{BOLD}{zr*1000:+6.1f} mm{RST}")
        print()

        print(f"  {BOLD}SPRUNG MASS ACCEL  z̈s(t)  {DIM}(target = 0){RST}")
        accel_col = G if abs(ddz_s) < 1.0 else (Y if abs(ddz_s) < 3.0 else R)
        print(f"  {accel_col}{signed_bar(ddz_s, 5.0, 40)}{RST}  {accel_col}{BOLD}{ddz_s:+6.3f} m/s²{RST}")
        print()

        print(f"  {DIM}{'─'*56}{RST}")
        print(f"  {BOLD}QUARTER-CAR MODEL{RST}   {DIM}(NIT Srinagar Section 7){RST}")
        print(f"  {DIM}  ms·z̈s = −ks(zs−zu) − c(t)(żs−żu){RST}")
        print(f"  {DIM}  mu·z̈u =  ks(zs−zu) + c(t)(żs−żu) − kt(zu−zr){RST}")
        print(f"  {DIM}{'─'*56}{RST}")
        print(f"\n  {W}Topics:{RST}  {DIM}/damper/status   /damper/debug   /damper/piston_effort{RST}")
        print(f"  {W}Frame:{RST}   {self.frame}   {W}Ctrl+C to stop{RST}\n")


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
