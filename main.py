import carla
import numpy as np
import cv2
import time
import random
import math
import pygame
import subprocess
import datetime
import os

from joblib import dump, load
import pandas as pd
from sklearn.linear_model import Ridge

from CDetectLane import CDetectLane

USE_ML_MODEL = False  # Toggle between ML model and PID

class CMachineLearningModel:
    def __init__(self, model_path='C:/Users/Michał/Desktop/Praca-magisterska/PracaMagisterska/ML_model/lane_steering_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.data = []

        # Ensure the model directory exists
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"✅ Created missing directory: {model_dir}")
        else:
            print(f"📁 Directory already exists: {model_dir}")

    def predict(self, error):
        if self.model is None:
            self.load_model()
        return float(self.model.predict([[error]]))

    def save_model(self):
        if self.data:
            df = pd.DataFrame(self.data, columns=['error', 'steer'])
            df.to_csv(self.model_path + ".csv", index=False)
            X = df[['error']].values
            y = df['steer'].values
            model = Ridge()
            model.fit(X, y)
            dump(model, self.model_path)
            print(f"✅ Model saved to {self.model_path}")

    def load_model(self):
        self.model = load(self.model_path)
        print(f"📥 Loaded model from {self.model_path}")

    def log_data(self, error, steer):
        self.data.append((error, steer))

class CFFmpeg:
    def __init__(self, disp_size):
        self.recording = False
        self.ffmpeg_process = None
        self.timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.output_dir = 'C:/Users/Michał/Desktop/Praca-magisterska/PracaMagisterska/out'
        os.makedirs(self.output_dir, exist_ok=True)

        csv_path = os.path.join(self.output_dir, f'{self.timestamp}.csv')
        self.lane_detector = CDetectLane(disp_size[0], disp_size[1], log_data=True, output_csv=csv_path)

    def start_recording(self, disp_size):
        if not self.recording:
            os.makedirs(self.output_dir, exist_ok=True)  # <-- Ensures directory before FFmpeg

            filename = os.path.normpath(os.path.join(self.output_dir, f'{self.timestamp}.avi'))

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{disp_size[0]}x{disp_size[1]}",
                "-r", "30",  # FPS
                "-i", "-",  # Input from stdin
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-crf", "23",
                "-loglevel", "error",
                filename
            ]

            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

            # Check if ffmpeg actually started
            time.sleep(0.2)  # Give it a moment to crash if it's going to
            if self.ffmpeg_process.poll() is not None:
                print("❌ FFmpeg failed to start.")
                self.ffmpeg_process = None
                self.recording = False
                return
            self.recording = True
            print(f"🔴 Recording started: {filename}")

    def stop_recording(self):
        if self.recording and self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
            except Exception as e:
                print(f"⚠️ Error closing stdin: {e}")
            self.ffmpeg_process = None
            self.recording = False
            print("🛑 Recording stopped.")

    def process_image(self, image):
        if not self.recording or self.ffmpeg_process is None:
            return

        if self.ffmpeg_process.poll() is not None:
            print("⚠️ FFmpeg not running.")
            self.recording = False
            return

        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            frame = array[:, :, :3]
            frame = self.lane_detector.detect_lanes(frame)
            self.ffmpeg_process.stdin.write(frame.tobytes())

        except (BrokenPipeError, ValueError) as e:
            print(f"❌ FFmpeg write error: {e}")
            self.recording = False
            if self.ffmpeg_process:
                try:
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.wait()
                except Exception as close_err:
                    print(f"⚠️ FFmpeg close error: {close_err}")
                self.ffmpeg_process = None


class CCamera:
    def __init__(self, world, bp_lib, disp_size, vehicle):
        self.camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(z=1.3,y=0, x=0.7))
        self.camera_bp.set_attribute("fov", "70")
        self.camera_bp.set_attribute('image_size_x', str(disp_size[0]))
        self.camera_bp.set_attribute('image_size_y', str(disp_size[1]))
        self.camera = world.spawn_actor(self.camera_bp, camera_init_trans, attach_to=vehicle)

    def start_listening(self, ffmpeg):
        self.camera.listen(lambda image: ffmpeg.process_image(image))

    def stop_listening(self):
        self.camera.stop()

    def destroy_camera(self):
        self.camera.destroy()


class CCar:
    def __init__(self, car_name='vehicle.ford.mustang'):
        self.car_name = car_name
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04_opt', map_layers=carla.MapLayer.Ground)
        self.bp_lib = self.world.get_blueprint_library()
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.spawn_vehicle()
        self.set_spectator_next_to_car()

    def spawn_vehicle(self):
        vehicle_bp = self.bp_lib.find(self.car_name)
        spawn_points = self.world.get_map().get_spawn_points()
        self.vehicle = self.world.spawn_actor(vehicle_bp, random.choice(spawn_points))

    def set_spectator_next_to_car(self):
        self.spectator = self.world.get_spectator()
        vehicle_transform = self.vehicle.get_transform()
        offset = carla.Location(y=-2,z=1.6)
        spectator_transform = carla.Transform(vehicle_transform.location + offset, vehicle_transform.rotation)
        self.spectator.set_transform(spectator_transform)

    def set_camera_on_a_car(self):
        self.scalar = 5
        self.disp_size = [x * self.scalar for x in [256, 256]]
        self.camera = CCamera(self.world, self.bp_lib, self.disp_size, self.vehicle)

    def start_processes(self):
        self.ffmpeg_process = CFFmpeg(self.disp_size)
        self.model = CMachineLearningModel()
        pygame.init()
        self.screen = pygame.display.set_mode((self.disp_size[0]//2, self.disp_size[1]//2))

    def destroy_camera(self):
        try:
            self.camera.stop_listening()
            self.camera.destroy_camera()
        except:
            try:
                self.camera.destroy_camera()
            except:
                return

    def destroy_vehicle(self):
        self.destroy_camera()
        self.vehicle.destroy()
        self.ffmpeg_process.lane_detector.close_file()
        if not USE_ML_MODEL:
            self.model.save_model()
        print("vehicle and camera successfully destroyed :D")

    def record_video(self):
        self.set_camera_on_a_car()
        self.start_processes()

        clock = pygame.time.Clock()
        control = carla.VehicleControl()
        control.throttle = 0.4
        control.steer = 0.0

        running = True
        driving = False
        lane_control_active = False

        lane_detector = self.ffmpeg_process.lane_detector
        self.camera.start_listening(self.ffmpeg_process)

        pid = PIDController(Kp=0.7, Ki=0.02, Kd=0.1)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        print("▶️ Starting...")
                        self.ffmpeg_process.start_recording(self.disp_size)
                        driving = True
                    elif event.key == pygame.K_k:
                        self.ffmpeg_process.stop_recording()
                        self.destroy_camera()
                        pygame.quit()
                        running = False
                    elif event.key == pygame.K_q:
                        running = False

            if driving:
                if not lane_control_active:
                    control.throttle = 0.4
                    control.steer = 0.0
                    self.vehicle.apply_control(control)
                    if lane_detector.last_position:
                        lane_control_active = True
                        print("✅ Lane detected.")
                else:
                    x_offset, _ = lane_detector.last_position
                    max_offset = self.disp_size[0] / 2
                    error = x_offset / max_offset

                    if USE_ML_MODEL:
                        steer = self.model.predict(error)
                    else:
                        steer = pid.compute(error)
                        self.model.log_data(error, steer)

                    control.steer = steer
                    self.vehicle.apply_control(control)

            clock.tick(30)


class PIDController:
    def __init__(self, Kp=0.5, Ki=0.01, Kd=0.2):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def reset(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_time = time.time()

    def compute(self, error):
        current_time = time.time()
        delta_time = current_time - self.prev_time
        delta_error = error - self.prev_error if delta_time > 0 else 0

        self.integral += error * delta_time
        derivative = delta_error / delta_time if delta_time > 0 else 0

        output = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

        self.prev_error = error
        self.prev_time = current_time

        return np.clip(output, -1.0, 1.0)


if __name__ == "__main__":
    car = CCar()
    car.record_video()
    car.destroy_vehicle()