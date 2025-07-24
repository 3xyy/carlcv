import time

import pygame
import sys
import random
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import random as rand
from sympy import false
import mpVisualizers as mpVis

WIDTH = 1750
HEIGHT = 1000
MARGIN = 150
FPS = 240
TILE_SPEED = 10

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.SCALED)
pygame.display.set_caption("Testing")
pygame.init()

modelPath = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=modelPath)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

NON_PINKY_DIST = 0.3
PINKY_DIST = 0.25
POSSIBLE_FINGERS = {"RPo", "RM", "RR", "RPi", "LPo", "LM", "LR", "LPi"}
PRESS_THRESH = 500

fingers = set()
keys_to_press = set()
completed = False
total_points = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 60)

# Fonts
font = pygame.font.SysFont("Arial", 24)

# Rectangles represent the tiles present on the screen
rectangles = []

clock = pygame.time.Clock()

running = True


class Button:
    def __init__(self, x, y, img):
        self.top_left = (x, y)
        self.img = img
        self.rect = self.img.get_rect()
        self.rect.topleft = (x, y)
        self.clicked = False

    def draw(self, screen):
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                self.clicked = True
                print("CLICKED")
        screen.blit(self.img, (self.rect.x, self.rect.y))

    def is_clicked(self):
        return self.clicked

    def reset(self):
        self.clicked = False


# Class rectangle used to represent tiles on the screen
class Rectangle:
    def __init__(self):
        self.w = (WIDTH - 2 * MARGIN) / 8
        self.h = self.w * 3 / 2
        self.color = (0, 0, 0)
        self.lane = random.randint(1, 8)
        self.x = int((self.lane - 1) * self.w + MARGIN)
        self.y = int(-1 * self.h)
        self.last_update_time = pygame.time.get_ticks()

    def update(self):
        self.y += TILE_SPEED
        self.last_update_time = pygame.time.get_ticks()

        if PRESS_THRESH - 5 < self.y < PRESS_THRESH + 5:

            if self.lane == 1:
                keys_to_press.add(("LPi", self))
            elif self.lane == 2:
                keys_to_press.add(("LR", self))
            elif self.lane == 3:
                keys_to_press.add(("LM", self))
            elif self.lane == 4:
                keys_to_press.add(("LPo", self))
            elif self.lane == 5:
                keys_to_press.add(("RPo", self))
            elif self.lane == 6:
                keys_to_press.add(("RM", self))
            elif self.lane == 7:
                keys_to_press.add(("RR", self))
            elif self.lane == 8:
                keys_to_press.add(("RPi", self))

        if self.y > HEIGHT:
            if self.lane == 1:
                keys_to_press.discard(("LPi", self))
            elif self.lane == 2:
                keys_to_press.discard(("LR", self))
            elif self.lane == 3:
                keys_to_press.discard(("LM", self))
            elif self.lane == 4:
                keys_to_press.discard(("LPo", self))
            elif self.lane == 5:
                keys_to_press.discard(("RPo", self))
            elif self.lane == 6:
                keys_to_press.discard(("RM", self))
            elif self.lane == 7:
                keys_to_press.discard(("RR", self))
            elif self.lane == 8:
                keys_to_press.discard(("RPi", self))

    # Method to draw tiles and lines on the pygame screen
    def draw(self, surface):
        r = pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x), int(self.y)), (int(self.x + self.w), int(self.y)), 3)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x), int(self.y)), (int(self.x), int(self.y + self.h)), 3)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x + self.w), int(self.y)),
                         (int(self.x + self.w), int(self.y + self.h)), 3)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x), int(self.y + self.h)),
                         (int(self.x + self.w), int(self.y + self.h)), 3)

    # Get the location of the top left of the tile
    def get_point(self):
        return self.x, self.y

    def set_color(self, color):
        self.color = color


class Piano:
    def __init__(self, display, gameStateManager):
        self.screen = display
        self.gameStateManager = gameStateManager
        self.frames = 0
        self.back_image = pygame.image.load("back.png").convert_alpha()
        self.back_button = Button(5, 800, pygame.transform.scale(self.back_image,
                                                                 (self.back_image.get_width() / 4,
                                                                  self.back_image.get_height() / 2)))
        self.last_overlay = None
        self.last_overlay_frame = -1
        self.last_frame = None
        self.last_detect_result = None

        with open("tilemap.json", 'r') as file:
            self.data = json.load(file)

        self.times = []
        for d in self.data:
            self.times.append(round(d.get("time"), 1))
        print(self.times)

        self.start_time = time.time()
        self.cur_note = 0

    def update_hand_overlay(self):
        ret, frame = cap.read()
        if not ret:
            return
        fingers.clear()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detect_result = detector.detect(mp_image)
        height = frame.shape[0]
        width = frame.shape[1]
        mask = np.zeros((height, width), dtype=np.uint8)
        exclude_indices = set([1, 2, 3, 4])
        all_indices = set()
        for i in range(21):
            all_indices.add(i)
        mask_indices = list(all_indices - exclude_indices)
        mask_indices.sort()
        circle_radius = 25
        palm_circle_radius = 60
        handedness = detect_result.handedness
        leftHand = None
        rightHand = None
        for i in range(len(handedness)):
            if handedness[i][0].category_name == "Left":
                if leftHand is None:
                    leftHand = i
            elif handedness[i][0].category_name == "Right":
                if rightHand is None:
                    rightHand = i

        def mask_hand(hand_landmarks):
            wx = int(hand_landmarks[0].x * width)
            wy = int(hand_landmarks[0].y * height)
            bx = int(hand_landmarks[9].x * width)
            by = int(hand_landmarks[9].y * height)
            palm_x = int((wx + bx) / 2)
            palm_y = int((wy + by) / 2)
            cv2.circle(mask, (width - palm_x, palm_y), palm_circle_radius, 255, -1)
            for tip_idx in [8, 12, 16, 20]:
                tx = int(hand_landmarks[tip_idx].x * width)
                ty = int(hand_landmarks[tip_idx].y * height)
                cv2.line(mask, (width - tx, ty), (width - wx, wy), 255, circle_radius * 2)
            for idx in mask_indices:
                cv2.circle(mask, (width - int(hand_landmarks[idx].x * width), int(hand_landmarks[idx].y * height)),
                           circle_radius, 255, -1)

        if rightHand is not None:
            mask_hand(detect_result.hand_landmarks[rightHand])
        if leftHand is not None:
            mask_hand(detect_result.hand_landmarks[leftHand])
        frame_mirrored = cv2.flip(frame, 1)
        hands_only = cv2.cvtColor(frame_mirrored, cv2.COLOR_BGR2RGB)
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = hands_only
        alpha_mask = (mask * 0.8).astype(np.uint8)
        rgba[:, :, 3] = alpha_mask
        if (width, height) != (WIDTH, HEIGHT):
            rgba = cv2.resize(rgba, (WIDTH, HEIGHT))
        surface = pygame.image.frombuffer(rgba.tobytes(), (WIDTH, HEIGHT), 'RGBA')
        self.last_overlay = surface
        self.last_frame = frame
        self.last_detect_result = detect_result

    def process_fingers_and_scoring(self, frame, detect_result):
        global completed, keys_to_press, fingers, total_points
        fingers.clear()
        leftHand = None
        rightHand = None
        handedness = detect_result.handedness
        for i in range(len(handedness)):
            if handedness[i][0].category_name == "Left" and leftHand is None:
                leftHand = i
            elif handedness[i][0].category_name == "Right" and rightHand is None:
                rightHand = i
        Rwrist, Rfinger1, Rfinger2, Rfinger3, Rfinger4, Rfinger5 = None, None, None, None, None, None
        if rightHand is not None:
            Rwrist = detect_result.hand_landmarks[rightHand][0].y
            Rfinger1 = abs(Rwrist - detect_result.hand_landmarks[rightHand][4].y)
            Rfinger2 = abs(Rwrist - detect_result.hand_landmarks[rightHand][8].y)
            Rfinger3 = abs(Rwrist - detect_result.hand_landmarks[rightHand][12].y)
            Rfinger4 = abs(Rwrist - detect_result.hand_landmarks[rightHand][16].y)
            Rfinger5 = abs(Rwrist - detect_result.hand_landmarks[rightHand][20].y)
        Lwrist, Lfinger1, Lfinger2, Lfinger3, Lfinger4, Lfinger5 = None, None, None, None, None, None
        if leftHand is not None:
            Lwrist = detect_result.hand_landmarks[leftHand][0].y
            Lfinger1 = abs(Lwrist - detect_result.hand_landmarks[leftHand][4].y)
            Lfinger2 = abs(Lwrist - detect_result.hand_landmarks[leftHand][8].y)
            Lfinger3 = abs(Lwrist - detect_result.hand_landmarks[leftHand][12].y)
            Lfinger4 = abs(Lwrist - detect_result.hand_landmarks[leftHand][16].y)
            Lfinger5 = abs(Lwrist - detect_result.hand_landmarks[leftHand][20].y)
        if Rfinger2 is not None:
            if Rfinger2 < NON_PINKY_DIST:
                fingers.add("RPo")
            else:
                fingers.discard("RPo")
        if Rfinger3 is not None:
            if Rfinger3 < NON_PINKY_DIST:
                fingers.add("RM")
            else:
                fingers.discard("RM")
        if Rfinger4 is not None:
            if Rfinger4 < NON_PINKY_DIST:
                fingers.add("RR")
            else:
                fingers.discard("RR")
        if Rfinger5 is not None:
            if Rfinger5 < PINKY_DIST:
                fingers.add("RPi")
            else:
                fingers.discard("RPi")
        if Lfinger2 is not None:
            if Lfinger2 < NON_PINKY_DIST:
                fingers.add("LPo")
            else:
                fingers.discard("LPo")
        if Lfinger3 is not None:
            if Lfinger3 < NON_PINKY_DIST:
                fingers.add("LM")
            else:
                fingers.discard("LM")
        if Lfinger4 is not None:
            if Lfinger4 < NON_PINKY_DIST:
                fingers.add("LR")
            else:
                fingers.discard("LR")
        if Lfinger5 is not None:
            if Lfinger5 < PINKY_DIST:
                fingers.add("LPi")
            else:
                fingers.discard("LPi")
        for finger in fingers:
            for key in keys_to_press:
                if finger == key[0]:
                    keys_to_press.discard(key)
                    key[1].set_color((0, 255, 0))
                    total_points += 1
                    break

    def run(self):
        global running, rectangles, total_points
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        self.screen.fill((255, 255, 255))
        pygame.draw.line(self.screen, (255, 0, 0), (MARGIN, 0), (MARGIN, HEIGHT))
        pygame.draw.line(self.screen, (255, 0, 0), (WIDTH - MARGIN, 0), (WIDTH - MARGIN, HEIGHT))
        pygame.draw.line(self.screen, (255, 0, 0), (0, PRESS_THRESH), (WIDTH, PRESS_THRESH), 3)
        for i in range(8):
            pygame.draw.line(self.screen, (0, 255, 0), (MARGIN + i * (WIDTH - 2 * MARGIN) / 8, 0),
                             (MARGIN + i * (WIDTH - 2 * MARGIN) / 8, HEIGHT))
        put_new = False
        cur_time = time.time()
        if self.cur_note < len(self.times):
            if round(cur_time - self.start_time, 1) == self.times[self.cur_note]:
                print(round(cur_time - self.start_time, 1))
                print("NEW TILE")
                put_new = True
                self.cur_note+=3
            if put_new:
                put_new = False
                r = Rectangle()
                rectangles.append(r)
            for rect in rectangles:
                rect.update()
                rect.draw(self.screen)
        else:
            end_text = font.render("GAME OVER!!! RETURN TO START", True, (255, 0, 0))
            end_rect = end_text.get_rect()
            end_rect.center = (700, 50)
            screen.blit(end_text, end_rect)
        text_surface = font.render(f"SCORE: {total_points}", True, (0, 0, 0))
        text_rect = text_surface.get_rect()
        text_rect.center = (60, 50)
        screen.blit(text_surface, text_rect)
        self.update_hand_overlay()
        frame = self.last_frame
        detect_result = self.last_detect_result
        overlay = self.last_overlay
        if frame is not None and detect_result is not None:
            self.process_fingers_and_scoring(frame, detect_result)
        if overlay is not None:
            self.screen.blit(overlay, (0, 0))
        self.back_button.draw(self.screen)
        if self.back_button.is_clicked():
            self.gameStateManager.set_state('start')
            self.back_button.reset()
        pygame.display.flip()
        self.frames += 1


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Piano Tiles")

        self.gameStateManager = GameStateManager('start')
        self.start = Start(self.screen, self.gameStateManager)
        self.piano = Piano(self.screen, self.gameStateManager)
        self.credit = Credits(self.screen, self.gameStateManager)
        self.states = {'piano': self.piano, 'start': self.start, 'credits': self.credit}

    def run(self):
        frames = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.states[self.gameStateManager.get_state()].run()
            self.clock.tick(240)


class Credits:
    def __init__(self, display, gameStateManager):
        self.display = display
        self.gameStateManager = gameStateManager

        self.my_font = pygame.font.SysFont('Arial', 30, bold=True)
        self.back_image = pygame.image.load("back.png").convert_alpha()
        self.back_button = Button(100, 800, pygame.transform.scale(self.back_image,
                                                                   (self.back_image.get_width() / 2,
                                                                    self.back_image.get_height() / 2)))

    def run(self):
        self.display.fill((0, 0, 0))
        text_surface = self.my_font.render("WILLIAM LIU, ZAYD HOSSAIN, YUVRAJ DAR", True, (255, 255, 255))
        self.display.blit(text_surface, (100, 100))

        self.back_button.draw(self.display)
        if self.back_button.is_clicked():
            self.gameStateManager.set_state('start')
            self.back_button.reset()

        pygame.display.update()


class Start:
    def __init__(self, display, gameStateManager):
        self.display = display
        self.gameStateManager = gameStateManager
        self.background = pygame.image.load("piano3.jpg")
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))

        self.start_image = pygame.image.load("play.png").convert_alpha()
        self.start_button = Button(100, 100, pygame.transform.scale(self.start_image,
                                                                    (self.start_image.get_width() / 2,
                                                                     self.start_image.get_height() / 2)))
        self.credits_image = pygame.image.load("credits.png").convert_alpha()
        self.credits_button = Button(100, 400, pygame.transform.scale(self.credits_image,
                                                                      (self.credits_image.get_width() / 2,
                                                                       self.credits_image.get_height() / 2)))
        self.quit_image = pygame.image.load("quit.png").convert_alpha()
        self.quit_button = Button(100, 700, pygame.transform.scale(self.quit_image,
                                                                   (self.quit_image.get_width() / 2,
                                                                    self.quit_image.get_height() / 2)))

        self.logo_image = pygame.image.load("LOGO.png").convert_alpha()
        self.logo_image = pygame.transform.scale(self.logo_image,
                                                 (self.logo_image.get_width() * 2,
                                                  self.logo_image.get_height() * 2))

    def run(self):
        # print("START RUNNING")
        self.display.blit(self.background, (0, 0))
        self.display.blit(self.logo_image, (800, 100))
        self.start_button.draw(self.display)
        self.credits_button.draw(self.display)
        self.quit_button.draw(self.display)

        if self.start_button.is_clicked():
            self.gameStateManager.set_state('piano')
            self.start_button.reset()
        if self.quit_button.is_clicked():
            pygame.quit()
        if self.credits_button.is_clicked():
            self.gameStateManager.set_state('credits')
            self.credits_button.reset()

        pygame.display.update()


class GameStateManager:
    def __init__(self, currentState):
        self.currentState = currentState

    def set_state(self, newState):
        self.currentState = newState

    def get_state(self):
        return self.currentState


if __name__ == '__main__':
    game = Game()
    game.run()

