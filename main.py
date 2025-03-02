import carla
import numpy as np
import cv2
import time
import random
import math
import ffmpeg
import pygame
import subprocess
import datetime
from CDetectLane import CDetectLane

class CFFmpeg:
    def __init__(self, disp_size):
        self.recording = False
        self.ffmpeg_process = None
        return

    def start_recording(self, disp_size):
        if not self.recording:
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            filename = f'C:/Users/Michał/Desktop/Praca-magisterska/out/{timestamp}.avi'

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Nadpisywanie plików
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{disp_size[0]}x{disp_size[1]}",
                "-r", "30",  # FPS
                "-i", "-",  # Input z `stdin`
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                filename
            ]

            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
            self.recording = True
            print(f"🔴 Nagrywanie rozpoczęte: {filename}")

    def stop_recording(self):
        if self.recording and self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            self.recording = False
            print("🛑 Nagrywanie zakończone.")

    def process_image(self, image):
        """ Przetwarza obraz z kamery i wysyła do ffmpeg. """
        if self.recording and self.ffmpeg_process:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # RGBA
            frame = array[:, :, :3]  # Konwersja do BGR (OpenCV)
            #CDetectLane(frame).detect_lines()
            self.ffmpeg_process.stdin.write(frame.tobytes())

class CCamera:
    def __init__(self, world, bp_lib, disp_size, vehicle):
        self.camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_init_trans = carla.Transform(carla.Location(z=1.3,y=0, x=0.5))
        self.camera_bp.set_attribute('image_size_x', str(disp_size[0]))
        self.camera_bp.set_attribute('image_size_y', str(disp_size[1]))
        self.camera = world.spawn_actor(self.camera_bp, camera_init_trans, attach_to=vehicle)

    def start_listening(self, CFFmpeg):
        self.camera.listen(lambda image: CFFmpeg.process_image(image)) #image.save_to_disk('C:/Users/Michał/Desktop/Praca-magisterska/out/%06d.png' % image.frame))

    def stop_listening(self):
        self.camera.stop()

    def destroy_camera(self):
        self.camera.destroy()

class CCar:
    def __init__(self, car_name = 'vehicle.ford.mustang'):
        self.car_name = car_name
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
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
        self.scalar = 3
        self.disp_size = [x * self.scalar for x in [256, 256]]
        self.camera = CCamera(self.world, self.bp_lib, self.disp_size, self.vehicle)

    def start_processes(self):
        self.ffmpeg_process = CFFmpeg(self.disp_size)
        pygame.init()
        self.screen=pygame.display.set_mode((self.disp_size[0], self.disp_size[1]))

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

    def car_go(self):
        self.vehicle.set_autopilot(True)

    def record_video(self):
        self.set_camera_on_a_car()
        self.car_go()
        self.start_processes()
        self.camera.start_listening(self.ffmpeg_process)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Naciśnięcie "R"
                        self.ffmpeg_process.start_recording(self.disp_size)
                    if event.key == pygame.K_k:  # Naciśnięcie "K"
                        self.ffmpeg_process.stop_recording()
                        self.destroy_camera()
                        pygame.quit()
                        running = False
                    if event.key == pygame.K_q:
                        running = False

if __name__ == "__main__":
    car = CCar("vehicle.tesla.model3")
    car.record_video()
    car.destroy_vehicle()

