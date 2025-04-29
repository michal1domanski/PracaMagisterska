import cv2
import numpy as np
import csv
import os
import collections

class CDetectLane:
    def __init__(self, width, height, log_data=False, output_csv="lane_data.csv"):
        self.width = width
        self.height = height
        self.log_data = log_data
        self.output_csv = output_csv
        self.frame_count = 0
        self.file = None
        self.writer = None

        self.bottomLeft = (width * 0.1, height * 0.71)
        self.topLeft = (width * 0.1, height * 0.4)
        self.topRight = (width * 0.9, height * 0.4)
        self.bottomRight = (width * 0.9, height * 0.71)

        self.lower_yellow = np.array([15, 60, 100], dtype=np.uint8)
        self.upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        self.lower_white = np.array([0, 0, 220], dtype=np.uint8)
        self.upper_white = np.array([180, 25, 255], dtype=np.uint8)

        self.last_polygon = None
        self.last_position = None
        self.last_left_lines = []
        self.last_right_lines = []

        self.offset_history = collections.deque(maxlen=5)
        self.smoothed_offset = 0.0
        self.alpha = 0.2
        self.max_offset_change = 15.0

        if self.log_data:
            self._open_file()

    def detect_lanes(self, image):
        image = image.copy()
        self.frame_count += 1
        cv2.putText(image, f"Frame ID: {self.frame_count}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, self.lower_white, self.upper_white)
        yellow_mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

        kernel = np.ones((5, 5), np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)

        masked_image = cv2.bitwise_and(image, image, mask=lane_mask)
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = self.region_of_interest(edges)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, np.array([]), 30, 40)
        left_lines, right_lines = self.extract_lane_edges(lines)

        if not left_lines and self.last_left_lines:
            left_lines = self.last_left_lines
        if not right_lines and self.last_right_lines:
            right_lines = self.last_right_lines

        if left_lines: self.last_left_lines = left_lines
        if right_lines: self.last_right_lines = right_lines

        lane_width = self.compute_lane_width(left_lines, right_lines)
        if lane_width is None or lane_width < 100:
            return image

        position = self.calculate_lane_position(left_lines, right_lines)
        self.last_position = position 
        lane_polygon = self.create_lane_polygon(left_lines, right_lines)
        turn = "unknown"

        if lane_polygon is not None:
            self.last_polygon = lane_polygon
        elif self.last_polygon is not None:
            lane_polygon = self.last_polygon

        if lane_polygon is not None:
            cv2.fillPoly(image, [lane_polygon], (0, 255, 255))
            for point in lane_polygon:
                cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)

            if position is not None:
                raw_offset, _ = position
                delta = raw_offset - self.smoothed_offset
                delta = np.clip(delta, -self.max_offset_change, self.max_offset_change)
                self.smoothed_offset += self.alpha * delta
                x_offset = self.smoothed_offset

                cv2.putText(image, f"Offset X: {x_offset:.1f}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                turn = self.detect_turn_direction(left_lines, right_lines)

                if self.log_data:
                    self.save_data(self.frame_count, x_offset, turn)

        return image

    def region_of_interest(self, edges):
        mask = np.zeros_like(edges)
        polygon = np.array([[self.bottomLeft, self.topLeft, self.topRight, self.bottomRight]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def extract_lane_edges(self, lines):
        left, right = [], []
        MIN_LENGTH = 30
        if lines is None:
            return left, right

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                length = np.hypot(x2 - x1, y2 - y1)
                slope = (y2 - y1) / (x2 - x1)
                if length < MIN_LENGTH or abs(slope) < 0.3:
                    continue
                (left if slope < 0 else right).append((x1, y1, x2, y2))
        return left, right

    def interpolate_x_at_y(self, x1, y1, x2, y2, y):
        if y2 == y1:
            return None
        slope = (x2 - x1) / (y2 - y1)
        return x1 + slope * (y - y1)

    def calculate_lane_position(self, left_lines, right_lines):
        y = int(self.height * 0.95)
        left_x = [self.interpolate_x_at_y(x1, y1, x2, y2, y)
                  for x1, y1, x2, y2 in left_lines]
        right_x = [self.interpolate_x_at_y(x1, y1, x2, y2, y)
                   for x1, y1, x2, y2 in right_lines]

        left_x = [x for x in left_x if x is not None]
        right_x = [x for x in right_x if x is not None]

        if not left_x or not right_x:
            return None

        lane_center = (np.mean(left_x) + np.mean(right_x)) / 2
        x_offset = lane_center - (self.width / 2)
        return x_offset, y

    def compute_lane_width(self, left_lines, right_lines):
        y = int(self.height * 0.95)
        left_x = [self.interpolate_x_at_y(x1, y1, x2, y2, y)
                  for x1, y1, x2, y2 in left_lines]
        right_x = [self.interpolate_x_at_y(x1, y1, x2, y2, y)
                   for x1, y1, x2, y2 in right_lines]

        left_x = [x for x in left_x if x is not None]
        right_x = [x for x in right_x if x is not None]

        if not left_x or not right_x:
            return None
        return np.mean(right_x) - np.mean(left_x)

    def detect_turn_direction(self, left_lines, right_lines):
        def avg_slope(lines):
            slopes = [(y2 - y1) / (x2 - x1) for x1, y1, x2, y2 in lines if x2 != x1]
            return np.mean(slopes) if slopes else 0

        left_slope = avg_slope(left_lines)
        right_slope = avg_slope(right_lines)
        total_slope = left_slope + right_slope

        if total_slope > 0.4:
            return "right"
        elif total_slope < -0.4:
            return "left"
        return "straight"

    def create_lane_polygon(self, left_lines, right_lines):
        y1 = int(self.height * 0.95)
        y2 = int(self.height * 0.6)

        def average_line(lines):
            if not lines:
                return None
            xs, ys = [], []
            for x1, y1_, x2, y2_ in lines:
                xs.extend([x1, x2])
                ys.extend([y1_, y2_])
            if len(xs) < 2 or len(ys) < 2:
                return None
            fit = np.polyfit(ys, xs, 1)
            return lambda y: int(fit[0] * y + fit[1])

        left_fit = average_line(left_lines)
        right_fit = average_line(right_lines)
        if not left_fit or not right_fit:
            return None

        points = [
            (left_fit(y1), y1), (left_fit(y2), y2),
            (right_fit(y2), y2), (right_fit(y1), y1)
        ]
        return np.array(points, dtype=np.int32)

    def _open_file(self):
        file_exists = os.path.exists(self.output_csv)
        self.file = open(self.output_csv, mode='a', newline='')
        self.writer = csv.writer(self.file)
        if not file_exists or os.stat(self.output_csv).st_size == 0:
            self.writer.writerow(["frame", "x_offset", "turn"])

    def save_data(self, frame_id, x_offset, turn):
        if self.log_data and self.writer:
            self.writer.writerow([frame_id, x_offset, turn])
            self.file.flush()

    def close_file(self):
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None
