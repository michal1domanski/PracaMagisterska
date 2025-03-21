import cv2
import numpy as np
import carla
import time

class CDetectLane:
    def __init__(self, width, height):
        self.bottomLeft = (width * 0.1, height * 0.71)
        self.topLeft = (width * 0.05, height * 0.48)
        self.topRight = (width * 0.95, height * 0.48)
        self.bottomRight = (width * 0.9, height * 0.71)
        pass

    def detect_lanes(self, image):
        # Konwersja do HSV
        # image = self.preprocess_image(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Zakresy kolorów dla białych i żółtych pasów
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([255, 30, 255], dtype=np.uint8)

        lower_yellow = np.array([15, 100, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)

        # Maskowanie białych i żółtych pasów
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Połączenie obu masek
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Zastosowanie maski do obrazu
        masked_image = cv2.bitwise_and(image, image, mask=lane_mask)
        
        # Konwersja na skale szarości i redukcja szumu
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detekcja krawędzi
        edges = cv2.Canny(blurred, 50, 150)

        # Ograniczenie obszaru do ROI
        edges = self.region_of_interest(edges)

        # height, width = image.shape[:2]
        # mask = np.zeros_like(edges)
        # polygon = np.array([[
        # (0, height), (width, height), (width//2, height//2)
        # ]], np.int32)
    
        # cv2.fillPoly(mask, polygon, 255)
        # masked_edges = cv2.bitwise_and(edges, mask)

        # Hough Transform do wykrywania linii
        rho = 1
        theta = np.pi / 180
        threshold = 30
        min_line_length = 30
        max_line_gap = 40

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), 
                                min_line_length, max_line_gap)

        # Rysowanie wykrytych linii
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Połączenie wykrytych pasów z obrazem
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        
        return result
    
    def region_of_interest(self, edges):
        # height, width = edges.shape
        mask = np.zeros_like(edges)

        # polygon = np.array([[
        #     (0, height),
        #     (width // 2 - 50, height // 2),
        #     (width // 2 + 50, height // 2),
        #     (width, height)
        # ]], np.int32)
        # Define a polygon mask to focus on the lane area
        polygon = np.array([[
            self.bottomLeft,  # Bottom-left
            self.topLeft,  # Top-left
            self.topRight,  # Top-right
            self.bottomRight  # Bottom-right
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)  # Fill ROI with white
        
        masked_edges = cv2.bitwise_and(edges, mask)
        cv2.polylines(masked_edges, polygon, True, (125, 125, 0))
        return masked_edges

    def enhance_contrast(self, image):
        """Enhances contrast using Adaptive Histogram Equalization (CLAHE)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_image

    def adjust_brightness(self, image):
        """Brightens shadowed areas using HSV adjustments."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)  # Equalize the brightness
        adjusted_hsv = cv2.merge((h, s, v))
        brightened_image = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
        return brightened_image

    def preprocess_image(self, image):
        """Applies contrast enhancement and brightness adjustment for better lane detection."""
        image = self.enhance_contrast(image)
        # image = self.adjust_brightness(image)
        return image

# def detect_lanes(image):
#     """Wykrywanie pasów ruchu za pomocą przetwarzania obrazu"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Skala szarości
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Rozmycie
#     edges = cv2.Canny(blur, 50, 150)  # Detekcja krawędzi
#     cv2.imshow(image)
#     # Tworzenie maski dla regionu zainteresowania (ROI)
#     height, width = edges.shape
#     mask = np.zeros_like(edges)
#     roi_vertices = np.array([[(50, height), (width//2-50, height//2), (width//2+50, height//2), (width-50, height)]], dtype=np.int32)
#     cv2.fillPoly(mask, roi_vertices, 255)
#     masked_edges = cv2.bitwise_and(edges, mask)

#     # Wykrywanie linii metodą Hougha
#     lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=200)
#     return lines

# def get_lane_center(lines, image_width):
#     """Obliczanie środka pasa na podstawie wykrytych linii"""
#     if lines is None:
#         return image_width // 2  # Brak wykrytych linii – domyślnie środek ekranu

#     left_lines, right_lines = [], []

#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Unikanie dzielenia przez zero
#         if slope < 0:
#             left_lines.append((x1, x2))
#         else:
#             right_lines.append((x1, x2))

#     if left_lines and right_lines:
#         left_x = np.mean([x for x1, x2 in left_lines for x in (x1, x2)])
#         right_x = np.mean([x for x1, x2 in right_lines for x in (x1, x2)])
#         return (left_x + right_x) / 2

#     return image_width // 2  # Domyślna wartość środka pasa


# def get_vehicle_position(image_width):
#     """Pozycja pojazdu w obrazie (zakładamy, że to środek dolnej krawędzi)"""
#     return image_width // 2

# class PIDController:
#     """Prosty kontroler PID do sterowania pojazdem"""
#     def __init__(self, Kp, Ki, Kd):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.integral = 0
#         self.prev_error = 0

#     def control(self, error, dt):
#         """Obliczanie korekcji sterowania"""
#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt
#         output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
#         self.prev_error = error
#         return output

# pid = PIDController(Kp=0.2, Ki=0.05, Kd=0.1)

# def process_image(image, vehicle):
#     """Obsługa obrazu z kamery i sterowanie pojazdem"""
#     array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]

#     # Wykrywanie linii pasa ruchu
#     lines = detect_lanes(array)
#     print("lol")
    
#     # Obliczenie błędu względem środka pasa
#     desired_position = get_lane_center(lines, array.shape[1])
#     current_position = get_vehicle_position(array.shape[1])
#     error = desired_position - current_position

#     # Obliczenie korekcji PID
#     correction = pid.control(error, 0.05)  # dt = 0.05s

#     # Sterowanie pojazdem
#     control = carla.VehicleControl()
#     control.steer = np.clip(correction * 0.01, -1.0, 1.0)  # Korekcja skrętu
#     control.throttle = 0.3  # Stała prędkość

#     vehicle.apply_control(control)

#     # Podgląd obrazu z wykrytymi liniami
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(array, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
#     cv2.imshow("Lane Detection", array)
#     cv2.waitKey(1)

# # Start nasłuchiwania kamery
# camera.listen(lambda image: process_image(image))

# print("Autopilot uruchomiony!")

# def stop_simulation(camera, vehicle):
#     """Zatrzymuje symulację i niszczy aktorów"""
#     camera.stop()
#     vehicle.destroy()
#     print("Symulacja zatrzymana.")

# stop_simulation()  # Wykonaj tę funkcję, jeśli chcesz zatrzymać wszystko