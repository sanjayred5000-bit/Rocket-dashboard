import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


class RocketSimulation:

    def __init__(
        self,
        m_wet,
        m_dry,
        thrust,
        burn_time,
        cd,
        radius,
        launch_angle_deg,
        dt=0.01,
    ):
        self.m_wet = m_wet
        self.m_dry = m_dry
        self.thrust = thrust
        self.burn_time = burn_time

        self.cd = cd
        self.area = math.pi * (radius**2)
        self.rho = 1.225
        self.g = 9.81

        self.dt = dt
        self.initial_angle = math.radians(launch_angle_deg)
        self.mass_flow_rate = (self.m_wet - self.m_dry) / self.burn_time

    def run_simulation(self, max_steps=2_000_000):
        times, x_vals, y_vals = [0.0], [0.0], [0.0]
        vx_vals, vy_vals, mass_vals = [0.0], [0.0], [self.m_wet]

        t = 0.0
        x, y = 0.0, 0.0
        vx, vy = 0.0, 0.0
        m = self.m_wet

        steps = 0
        while y >= 0:
            if t < self.burn_time:
                current_thrust = self.thrust
                m -= self.mass_flow_rate * self.dt
            else:
                current_thrust = 0.0
                m = self.m_dry

            # FIX: velocity magnitude must use squares, not *2
            v = math.sqrt(vx**2 + vy**2)
            if v != 0:
                cos_theta = vx / v
                sin_theta = vy / v
            else:
                cos_theta = math.cos(self.initial_angle)
                sin_theta = math.sin(self.initial_angle)

            F_tx = current_thrust * cos_theta
            F_ty = current_thrust * sin_theta

            drag_magnitude = 0.5 * self.rho * (v**2) * self.cd * self.area
            F_dx = -drag_magnitude * cos_theta
            F_dy = -drag_magnitude * sin_theta

            F_gx = 0.0
            F_gy = -m * self.g

            F_net_x = F_tx + F_dx + F_gx
            F_net_y = F_ty + F_dy + F_gy

            if y <= 0.001 and F_net_y < 0:
                F_net_y = 0.0
                vx, vy = 0.0, 0.0

            ax = F_net_x / m
            ay = F_net_y / m

            vx += ax * self.dt
            vy += ay * self.dt

            x += vx * self.dt
            y += vy * self.dt
            t += self.dt

            times.append(t)
            x_vals.append(x)
            y_vals.append(max(y, 0))
            vx_vals.append(vx)
            vy_vals.append(vy)
            mass_vals.append(m)

            steps += 1
            if steps > max_steps:
                # safety valve so a bad parameter set can't hang the app
                break

        return {
            "time": np.array(times),
            "x": np.array(x_vals),
            "y": np.array(y_vals),
            "vx": np.array(vx_vals),
            "vy": np.array(vy_vals),
            "mass": np.array(mass_vals),
        }

    def plot_dashboard(self, results):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(
            "2D Rocket Flight Simulation Dashboard", fontsize=16, fontweight="bold"
        )

        axs[0, 0].plot(results["x"], results["y"], "b-", linewidth=2)
        axs[0, 0].set_title("Flight Trajectory (X vs Y)")
        axs[0, 0].set_xlabel("Downrange Distance (m)")
        axs[0, 0].set_ylabel("Altitude (m)")
        axs[0, 0].grid(True)

        # FIX: same *2 -> **2 correction here
        speed = np.sqrt(results["vx"] ** 2 + results["vy"] ** 2)
        axs[0, 1].plot(results["time"], speed, "r-", linewidth=2)
        axs[0, 1].set_title("Total Velocity vs Time")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Speed (m/s)")
        axs[0, 1].grid(True)

        axs[1, 0].plot(results["time"], results["mass"], "g-", linewidth=2)
        axs[1, 0].set_title("Mass Depletion")
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Mass (kg)")
        axs[1, 0].grid(True)

        axs[1, 1].plot(results["time"], results["y"], "m-", linewidth=2)
        axs[1, 1].set_title("Altitude vs Time")
        axs[1, 1].set_xlabel("Time (s)")
        axs[1, 1].set_ylabel("Altitude (m)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        return fig


def main():
    st.set_page_config(page_title="Rocket Flight Simulator", layout="wide")
    st.title("🚀 2D Rocket Flight Simulation Dashboard")
    st.caption(
        "Adjust the rocket parameters in the sidebar and click **Run Simulation**."
    )

    with st.sidebar:
        st.header("Rocket Parameters")
        m_wet = st.number_input("Wet mass (kg)", min_value=0.1, value=120.0, step=1.0)
        m_dry = st.number_input("Dry mass (kg)", min_value=0.1, value=50.0, step=1.0)
        thrust = st.number_input("Thrust (N)", min_value=0.0, value=3200.0, step=10.0)
        burn_time = st.number_input(
            "Burn time (s)", min_value=0.01, value=12.0, step=0.5
        )

        st.header("Aerodynamics")
        cd = st.number_input("Drag coefficient (Cd)", min_value=0.0, value=0.45, step=0.01)
        radius = st.number_input("Body radius (m)", min_value=0.001, value=0.15, step=0.01)

        st.header("Launch")
        launch_angle_deg = st.slider("Launch angle (deg)", 0.0, 90.0, 70.0, step=1.0)
        dt = st.number_input(
            "Time step dt (s)", min_value=0.0001, value=0.01, step=0.001, format="%.4f"
        )

        run_button = st.button("Run Simulation", type="primary", use_container_width=True)

    if m_dry >= m_wet:
        st.error("Dry mass must be less than wet mass.")
        return

    if run_button or "sim_data" not in st.session_state:
        sim = RocketSimulation(
            m_wet=m_wet,
            m_dry=m_dry,
            thrust=thrust,
            burn_time=burn_time,
            cd=cd,
            radius=radius,
            launch_angle_deg=launch_angle_deg,
            dt=dt,
        )
        with st.spinner("Running simulation..."):
            st.session_state["sim_data"] = sim.run_simulation()
            st.session_state["sim"] = sim

    sim_data = st.session_state["sim_data"]
    sim = st.session_state["sim"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Apogee (Max Altitude)", f"{np.max(sim_data['y']):.1f} m")
    col2.metric("Max Downrange Distance", f"{np.max(sim_data['x']):.1f} m")
    col3.metric("Total Flight Time", f"{np.max(sim_data['time']):.1f} s")

    fig = sim.plot_dashboard(sim_data)
    st.pyplot(fig)

    with st.expander("Show raw simulation data"):
        import pandas as pd

        df = pd.DataFrame(
            {
                "time": sim_data["time"],
                "x": sim_data["x"],
                "y": sim_data["y"],
                "vx": sim_data["vx"],
                "vy": sim_data["vy"],
                "mass": sim_data["mass"],
            }
        )
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()