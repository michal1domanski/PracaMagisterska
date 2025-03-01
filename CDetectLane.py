import cv2
import numpy as np
import carla

def detect_lanes(image):
    """Wykrywanie pasów ruchu za pomocą przetwarzania obrazu"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Skala szarości
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Rozmycie
    edges = cv2.Canny(blur, 50, 150)  # Detekcja krawędzi
    cv2.imshow(image)
    # Tworzenie maski dla regionu zainteresowania (ROI)
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(50, height), (width//2-50, height//2), (width//2+50, height//2), (width-50, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Wykrywanie linii metodą Hougha
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=200)
    return lines

def get_lane_center(lines, image_width):
    """Obliczanie środka pasa na podstawie wykrytych linii"""
    if lines is None:
        return image_width // 2  # Brak wykrytych linii – domyślnie środek ekranu

    left_lines, right_lines = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Unikanie dzielenia przez zero
        if slope < 0:
            left_lines.append((x1, x2))
        else:
            right_lines.append((x1, x2))

    if left_lines and right_lines:
        left_x = np.mean([x for x1, x2 in left_lines for x in (x1, x2)])
        right_x = np.mean([x for x1, x2 in right_lines for x in (x1, x2)])
        return (left_x + right_x) / 2

    return image_width // 2  # Domyślna wartość środka pasa


def get_vehicle_position(image_width):
    """Pozycja pojazdu w obrazie (zakładamy, że to środek dolnej krawędzi)"""
    return image_width // 2

class PIDController:
    """Prosty kontroler PID do sterowania pojazdem"""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def control(self, error, dt):
        """Obliczanie korekcji sterowania"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

pid = PIDController(Kp=0.2, Ki=0.05, Kd=0.1)

def process_image(image, vehicle):
    """Obsługa obrazu z kamery i sterowanie pojazdem"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]

    # Wykrywanie linii pasa ruchu
    lines = detect_lanes(array)
    print("lol")
    
    # Obliczenie błędu względem środka pasa
    desired_position = get_lane_center(lines, array.shape[1])
    current_position = get_vehicle_position(array.shape[1])
    error = desired_position - current_position

    # Obliczenie korekcji PID
    correction = pid.control(error, 0.05)  # dt = 0.05s

    # Sterowanie pojazdem
    control = carla.VehicleControl()
    control.steer = np.clip(correction * 0.01, -1.0, 1.0)  # Korekcja skrętu
    control.throttle = 0.3  # Stała prędkość

    vehicle.apply_control(control)

    # Podgląd obrazu z wykrytymi liniami
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(array, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    cv2.imshow("Lane Detection", array)
    cv2.waitKey(1)

# Start nasłuchiwania kamery
camera.listen(lambda image: process_image(image))

print("Autopilot uruchomiony!")

def stop_simulation(camera, vehicle):
    """Zatrzymuje symulację i niszczy aktorów"""
    camera.stop()
    vehicle.destroy()
    print("Symulacja zatrzymana.")

stop_simulation()  # Wykonaj tę funkcję, jeśli chcesz zatrzymać wszystko