#!/bin/env python3
"""
damper_controller_node.py
=========================
Implements the PID-controlled electromagnetic damper from the NIT Srinagar
Group 2 project report.

Quarter-car model equations (from Section 7 of report):
  Sprung mass:   ms * z̈s = -ks(zs - zu) - c(t)(żs - żu)
  Unsprung mass: mu * z̈u = ks(zs - zu) + c(t)(żs - żu) - kt(zu - zr)

Electromagnetic model:
  F_lorentz = B * I * L          (Lorentz force on piston)
  c(t) = c_passive + k_em * I    (variable damping coefficient)
  I = clip(PID_output, 0, I_max) (coil current)

PID error:
  setpoint = desired sprung mass acceleration (target = 0, i.e. smooth ride)
  measured = actual sprung mass acceleration from IMU / position diff
  error    = setpoint - measured

Subscribes:
  /damper/joint_states     — piston position & velocity

Publishes:
  /damper/joint_effort     — force applied to piston joint
  /damper/status           — DamperStatus (current, damping, force, position)
  /damper/excitation       — road input (simulated sinusoidal bump)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import math
import time


# ═══════════════════════════════════════════════════════
#  PID controller
# ═══════════════════════════════════════════════════════
class PID:
    def __init__(self, kp, ki, kd, out_min, out_max, i_limit=200.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_limit = i_limit
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_t   = None

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_t   = None

    def step(self, error):
        now = time.monotonic()
        dt  = (now - self._prev_t) if self._prev_t else 0.01
        dt  = max(dt, 1e-5)
        self._prev_t = now

        self._integral += error * dt
        self._integral  = max(-self.i_limit, min(self.i_limit, self._integral))
        deriv = (error - self._prev_err) / dt
        self._prev_err = error

        out = self.kp * error + self.ki * self._integral + self.kd * deriv
        return max(self.out_min, min(self.out_max, out))


# ═══════════════════════════════════════════════════════
#  Quarter-car state integrator
# ═══════════════════════════════════════════════════════
class QuarterCarModel:
    """
    Numerically integrates the quarter-car ODEs in real-time
    alongside the Gazebo joint simulation to provide sprung
    mass acceleration for PID feedback.

    Parameters match NIT Srinagar project values.
    """
    def __init__(self):
        # ── Masses ────────────────────────────────────────
        self.ms  = 240.0   # kg  sprung mass (quarter of ~960 kg car)
        self.mu  =  35.0   # kg  unsprung (wheel + carrier)
        # ── Stiffness ─────────────────────────────────────
        self.ks  = 16000.0  # N/m  suspension spring
        self.kt  = 160000.0 # N/m  tyre stiffness
        # ── Damping (updated by PID) ─────────────────────
        self.c   = 1500.0   # N·s/m  starts at passive value
        # ── States ────────────────────────────────────────
        self.zs  = 0.0;  self.dzs = 0.0
        self.zu  = 0.0;  self.dzu = 0.0
        self.zr  = 0.0   # road input

    def step(self, dt, c_t, zr):
        """Euler integrate one timestep."""
        self.zr = zr
        # Relative displacements
        dz_su = self.zs - self.zu
        dv_su = self.dzs - self.dzu
        dz_ur = self.zu - zr

        # Accelerations from report equations
        ddz_s = (-self.ks * dz_su - c_t * dv_su) / self.ms
        ddz_u = ( self.ks * dz_su + c_t * dv_su - self.kt * dz_ur) / self.mu

        # Euler integration
        self.dzs += ddz_s * dt
        self.dzu += ddz_u * dt
        self.zs  += self.dzs * dt
        self.zu  += self.dzu * dt

        return ddz_s  # sprung mass acceleration (comfort metric)


# ═══════════════════════════════════════════════════════
#  ROS2 Node
# ═══════════════════════════════════════════════════════
class DamperControllerNode(Node):

    # ── Electromagnetic constants ──────────────────────
    B       = 0.80   # T    — flux density
    L_wire  = 2.50   # m    — active coil wire
    I_MAX   = 15.0   # A    — max current
    C_PASS  = 800.0  # N·s/m passive damping
    K_EM    = 80.0   # extra damping per Ampere

    def __init__(self):
        super().__init__('damper_controller')

        # ── Parameters ────────────────────────────────────
        self.declare_parameter('kp',            22.0)
        self.declare_parameter('ki',             1.2)
        self.declare_parameter('kd',             5.0)
        self.declare_parameter('excitation_amp', 0.04)   # m  bump amplitude
        self.declare_parameter('excitation_freq',0.8)    # Hz
        self.declare_parameter('mode', 'pid')            # 'pid' | 'passive' | 'max'

        kp   = self.get_parameter('kp').value
        ki   = self.get_parameter('ki').value
        kd   = self.get_parameter('kd').value
        self.amp  = self.get_parameter('excitation_amp').value
        self.freq = self.get_parameter('excitation_freq').value
        self.mode = self.get_parameter('mode').value

        # ── PID: drives chassis acceleration → 0 ──────────
        self.pid = PID(kp, ki, kd, out_min=0.0, out_max=self.I_MAX * self.B * self.L_wire)

        # ── Quarter-car model ─────────────────────────────
        self.qc = QuarterCarModel()

        # ── State ─────────────────────────────────────────
        self.piston_pos = 0.0
        self.piston_vel = 0.0
        self.current    = 0.0
        self.c_t        = self.C_PASS
        self.t_start    = time.monotonic()
        self.prev_loop  = time.monotonic()

        # ── Pub / Sub ─────────────────────────────────────
        self.sub_js = self.create_subscription(
            JointState, '/damper/joint_states', self.joint_cb, 10)

        self.pub_effort  = self.create_publisher(Float64,           '/damper/piston_effort',  10)
        self.pub_js_cmd  = self.create_publisher(JointState,        '/damper/joint_command',   10)
        self.pub_status  = self.create_publisher(Float32MultiArray, '/damper/status',          10)
        self.pub_debug   = self.create_publisher(String,            '/damper/debug',           10)

        # Control loop 100 Hz
        self.create_timer(0.01, self.control_loop)
        self.get_logger().info(
            f'Damper controller started | mode={self.mode} '
            f'Kp={kp} Ki={ki} Kd={kd}'
        )

    def joint_cb(self, msg: JointState):
        if 'piston_travel' in msg.name:
            idx = msg.name.index('piston_travel')
            self.piston_pos = msg.position[idx] if msg.position else 0.0
            self.piston_vel = msg.velocity[idx]  if msg.velocity  else 0.0
            

    def control_loop(self):
        now = time.monotonic()
        dt  = now - self.prev_loop
        dt  = max(dt, 1e-4)
        self.prev_loop = now
        t   = now - self.t_start

        # ── Road excitation (sinusoidal bump input) ────────
        zr = self.amp * math.sin(2 * math.pi * self.freq * t)

        # ── Quarter-car model step ─────────────────────────
        ddz_s = self.qc.step(dt, self.c_t, zr)

        # ── PID: target sprung acceleration = 0 ───────────
        if self.mode == 'pid':
            error    = 0.0 - ddz_s          # want zero chassis accel
            f_out    = self.pid.step(error)
            I_cmd    = f_out / (self.B * self.L_wire)
            I_cmd    = max(0.0, min(self.I_MAX, I_cmd))

        elif self.mode == 'passive':
            I_cmd = 0.0                      # no coil — pure passive

        elif self.mode == 'max':
            I_cmd = self.I_MAX               # full stiffening

        else:
            I_cmd = 0.0

        self.current = I_cmd

        # ── Variable damping coefficient ───────────────────
        self.c_t = self.C_PASS + self.K_EM * I_cmd

        # ── Lorentz force on piston ────────────────────────
        F_lorentz = self.B * I_cmd * self.L_wire

        # ── Total damper force = damping + Lorentz ─────────
        F_damp    = self.c_t * self.piston_vel
        F_total   = -(F_damp + F_lorentz * math.copysign(1, self.piston_vel))

        # ── Publish effort to Gazebo joint ─────────────────
        eff      = Float64()
        eff.data = float(F_total)
        self.pub_effort.publish(eff)

        # Publish JointState so RViz2 animates piston position
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = ['piston_travel']
        js.position = [float(self.qc.zs * 0.8)]   # scale zs to piston travel
        js.velocity = [float(self.qc.dzs * 0.8)]
        js.effort   = [float(F_total)]
        self.pub_js_cmd.publish(js)

        # ── Status array: [I, C_t, F_lorentz, zs, ddz_s, zr] ──
        status      = Float32MultiArray()
        status.data = [
            float(I_cmd),
            float(self.c_t),
            float(F_lorentz),
            float(self.qc.zs),
            float(ddz_s),
            float(zr),
            float(self.piston_pos),
        ]
        self.pub_status.publish(status)

        # ── Human-readable debug ───────────────────────────
        dbg = String()
        dbg.data = (
            f"t={t:.2f}s | zr={zr*100:.1f}cm | "
            f"I={I_cmd:.2f}A | C={self.c_t:.0f}N·s/m | "
            f"F_L={F_lorentz:.1f}N | "
            f"zs={self.qc.zs*100:.2f}cm | "
            f"z̈s={ddz_s:.3f}m/s² | "
            f"piston={self.piston_pos*100:.1f}cm"
        )
        self.pub_debug.publish(dbg)

        if int(t * 10) % 10 == 0:   # log every 1 s
            self.get_logger().info(dbg.data)


def main(args=None):
    rclpy.init(args=args)
    node = DamperControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()