import numpy as np
import gymnasium as gym
from gymnasium import spaces
import asyncio
import mavsdk
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityBodyYawspeed


class DroneEnv(gym.Env):
    """Custom Environment for PX4 drone in Gazebo"""
    metadata = {'render.modes': ['human']}

    def __init__(self, connection_string="udp://:14540"):
        super(DroneEnv, self).__init__()

        # Define action and observation space
        # Actions: [velocity_forward, velocity_right, velocity_down, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-2, -2, -2, -45]),
            high=np.array([2, 2, 2, 45]),
            dtype=np.float32
        )

        # Observations: [position_north, position_east, position_down,
        #                velocity_north, velocity_east, velocity_down,
        #                roll, pitch, yaw,
        #                + obstacle detections from YOLO (simplified for now)]
        self.observation_space = spaces.Box(
            low=np.array([-100, -100, -100, -10, -10, -10, -180, -90, -180] + [0] * 10),
            high=np.array([100, 100, 100, 10, 10, 10, 180, 90, 180] + [1] * 10),
            dtype=np.float32
        )

        # MAVSDK setup
        self.connection_string = connection_string
        self.drone = System()
        self.loop = asyncio.get_event_loop()
        self._is_connected = False

        # Episode parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        self.goal_position = np.array([50.0, 50.0, -10.0])  # Example goal

        # Initialize YOLO, RTAB-Map, and OMPL interfaces
        # (We'll add these later)
        self.yolo_detector = None  # Placeholder for YOLO
        self.rtab_map = None  # Placeholder for RTAB-Map
        self.path_planner = None  # Placeholder for OMPL

    async def _setup_drone(self):
        """Connect to the drone and prepare for offboard control"""
        print(f"Connecting to drone at {self.connection_string}...")
        await self.drone.connect(system_address=self.connection_string)

        # Wait for drone to connect
        print("Waiting for drone to connect...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Drone connected!")
                break

        # Wait a bit for system to stabilize
        await asyncio.sleep(2)

        # Check if the drone is ready
        print("Checking drone health...")
        health_ok = False
        for _ in range(10):  # Try a few times with a timeout
            try:
                health_ok = await self.drone.telemetry.health_all_ok()
                if health_ok:
                    break
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Health check error: {e}")
                await asyncio.sleep(1)

        if not health_ok:
            print("WARNING: Drone not reporting all health checks OK")
            # Continue anyway - incase SITL just isn't reporting the health correctly

        try:
            print("-- Arming")
            await self.drone.action.arm()
        except Exception as e:
            print(f"Arming error: {e}")
            # Continue anyway - might already be armed

        # Set to offboard mode
        print("-- Setting initial setpoint")
        try:
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        except Exception as e:
            print(f"Setting velocity error: {e}")
            return False

        print("-- Starting offboard")
        try:
            await self.drone.offboard.start()
        except mavsdk.offboard.OffboardError as error:
            if "Offboard is already active" not in str(error):
                print(f"Starting offboard mode failed with error: {error}")
                try:
                    await self.drone.action.disarm()
                except:
                    pass
                return False

        print("Drone setup complete")
        return True

    def reset(self, **kwargs):
        """Reset the drone to initial state"""
        self.current_step = 0

        # Initialize connection if not already connected
        if not hasattr(self, '_is_connected') or not self._is_connected:
            success = self.loop.run_until_complete(self._setup_drone())
            if not success:
                raise RuntimeError("Failed to setup drone connection")
            self._is_connected = True

        # Reset drone position in simulator
        self.loop.run_until_complete(self._reset_drone())

        # Get initial observation
        observation = self.loop.run_until_complete(self._get_observation())
        info = {}

        return observation, info

    async def _reset_drone(self):
        """Reset the drone's position"""
        try:
            # Return to launch/home position
            await self.drone.action.return_to_launch()
            # Wait for landing
            await asyncio.sleep(5)
            # Rearm and prepare for a new episode
            await self.drone.action.arm()
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            try:
                await self.drone.offboard.start()
            except mavsdk.offboard.OffboardError as error:
                print(f"Starting offboard mode failed with error: {error}")
                if "Offboard is already active" not in str(error):
                    print("Failed to start offboard mode, attempting to continue anyway")
        except Exception as e:
            print(f"Error during drone reset: {e}")
            # Try to recover by at least setting velocity to zero
            try:
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            except:
                pass

    def step(self, action):
        """Execute action and return new state"""
        self.current_step += 1

        # Convert to proper action format and send to drone
        self.loop.run_until_complete(self._send_action(action))

        # Get new state, reward, etc.
        observation = self.loop.run_until_complete(self._get_observation())

        # Calculate reward based on distance to goal, collision avoidance, etc.
        reward = self._compute_reward(observation)

        # Check if episode is done
        done = self.current_step >= self.max_episode_steps

        # Collision detection (simplified)
        position = observation[:3]
        if self._check_collision(position):
            reward = -100  # Penalty for collision
            done = True

        # Goal reached check
        if np.linalg.norm(position - self.goal_position) < 3.0:  # 3m threshold
            reward += 100  # Bonus for reaching goal
            done = True

        info = {}
        return observation, reward, done, False, info

    async def _send_action(self, action):
        """Send velocity commands to the drone"""
        vx, vy, vz, yaw_rate = action
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(vx, vy, vz, yaw_rate))

    async def _get_observation(self):
        """Get the current drone state with consistent data types"""

        # Default values in case of error
        pos_ned = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vel_ned = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        att = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Get position with timeout
        try:
            async for position in self.drone.telemetry.position():
                try:
                    if hasattr(position, 'north_m'):
                        pos_ned = np.array([position.north_m, position.east_m, position.down_m], dtype=np.float32)
                    else:
                        # Fallback for older or different MAVSDK data.
                        # Just store zeros or do a lat/lon -> local NED transform.
                        pos_ned = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # or some default
                except Exception as e:
                    print(f"Error processing position attributes: {e}")
                break

        except Exception as e:
            print(f"Error getting position: {e}")

        # Get velocity with timeout
        try:
            async for velocity in self.drone.telemetry.velocity_ned():
                try:
                    if hasattr(velocity, 'north_m_s'):
                        vel_ned = np.array([velocity.north_m_s, velocity.east_m_s, velocity.down_m_s], dtype=np.float32)
                    elif hasattr(velocity, 'velocity_north_m_s'):
                        vel_ned = np.array(
                            [velocity.velocity_north_m_s, velocity.velocity_east_m_s, velocity.velocity_down_m_s],
                            dtype=np.float32
                        )
                    else:
                        print(f"Velocity object attributes: {dir(velocity)}")
                except Exception as e:
                    print(f"Error processing velocity attributes: {e}")
                break
        except Exception as e:
            print(f"Error getting velocity: {e}")

        # Get attitude with timeout
        try:
            async for attitude in self.drone.telemetry.attitude_euler():
                try:
                    if hasattr(attitude, 'roll_deg'):
                        att = np.array([attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg], dtype=np.float32)
                    elif hasattr(attitude, 'roll'):
                        # Convert radians to degrees if necessary
                        att = np.array([
                            np.degrees(attitude.roll),
                            np.degrees(attitude.pitch),
                            np.degrees(attitude.yaw)
                        ], dtype=np.float32)
                    else:
                        print(f"Attitude object attributes: {dir(attitude)}")
                except Exception as e:
                    print(f"Error processing attitude attributes: {e}")
                break
        except Exception as e:
            print(f"Error getting attitude: {e}")

        # Placeholder for YOLO obstacle detections (10 values representing detected objects)
        # In a real implementation, this would come from your YOLO model
        yolo_detections = np.zeros(10, dtype=np.float32)

        # Combine all observations
        observation = np.concatenate([pos_ned, vel_ned, att, yolo_detections])
        return observation.astype(np.float32)  # Ensure consistent float32 type

    def _compute_reward(self, observation):
        """Calculate reward based on current state"""
        position = observation[:3]
        velocity = observation[3:6]

        # Distance to goal reward - scaled to be less severely negative
        distance_to_goal = np.linalg.norm(position - self.goal_position)
        max_possible_distance = np.linalg.norm(
            np.array([100, 100, 100]) - self.goal_position)
        normalized_distance = distance_to_goal / max_possible_distance  # Between 0 and 1
        distance_reward = -10.0 * normalized_distance  # Scale between -10 and 0

        # Progress reward - reward for getting closer to the goal
        if hasattr(self, 'last_distance'):
            progress = self.last_distance - distance_to_goal
            progress_reward = 5.0 * progress  # Positive reward for getting closer
        else:
            progress_reward = 0.0
        self.last_distance = distance_to_goal

        # Penalize unnecessary movement or energy consumption (scaled to be less severe)
        velocity_magnitude = np.linalg.norm(velocity)
        energy_penalty = -0.1 * velocity_magnitude  # Linear penalty based on velocity magnitude

        # Stability reward - small reward for maintaining level flight
        attitude = observation[6:9]  # roll, pitch, yaw
        stability_reward = 0.1 * (1.0 - min(1.0, np.sqrt(attitude[0] ** 2 + attitude[1] ** 2) / 90.0))

        # Combine rewards
        total_reward = distance_reward + progress_reward + energy_penalty + stability_reward

        # Add small living penalty to encourage faster completion
        living_penalty = -0.1

        if self.current_step % 100 == 0:  # Print every 100 steps to avoid log spam
            print(f"Rewards: distance={distance_reward:.2f}, progress={progress_reward:.2f}, "
                  f"energy={energy_penalty:.2f}, stability={stability_reward:.2f}, "
                  f"total={total_reward + living_penalty:.2f}")

        return total_reward + living_penalty

    def _check_collision(self, position):
        """Check if drone has collided with obstacles"""
        # This is a placeholder for collision detection
        # In a real implementation, this would use data from your simulation or sensors

        # For now, simply define some no-fly zones as example obstacles
        obstacles = [
            {"center": np.array([20.0, 20.0, -5.0]), "radius": 5.0},
            {"center": np.array([30.0, 40.0, -5.0]), "radius": 5.0}
        ]

        for obstacle in obstacles:
            if np.linalg.norm(position - obstacle["center"]) < obstacle["radius"]:
                return True

        return False

    def render(self, mode='human'):
        """Render the environment (optional, as Gazebo provides visualization)"""
        pass

    def close(self):
        """Clean up resources"""
        self.loop.run_until_complete(self._cleanup())

    async def _cleanup(self):
        """Land the drone and disconnect"""
        if not self._is_connected:
            print("Drone wasn't connected, no cleanup needed")
            return

        try:
            print("-- Landing")
            await self.drone.action.land()
            await asyncio.sleep(5)  # Wait for landing
            self._is_connected = False
        except Exception as e:
            print(f"Error during cleanup: {e}")
