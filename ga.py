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
import pygame.mixer
import pygame.midi

WIDTH = 1750
HEIGHT = 1000
MARGIN = 150
FPS = 240
TILE_SPEED = 10

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.SCALED)
pygame.display.set_caption("Testing")
pygame.init()
pygame.mixer.init()
pygame.midi.init()
midi_player = pygame.midi.Output(0)
MIDI_VELOCITY = 127
MIDI_DURATION_MS = 200

NON_PINKY_DIST = 0.3
PINKY_DIST = 0.25
POSSIBLE_FINGERS = {"RPo", "RM", "RR", "RPi", "LPo", "LM", "LR", "LPi"}
PRESS_THRESH = 500

fingers = set()
keys_to_press = set()
completed = False
total_points = 0

# Fonts
font = pygame.font.SysFont("Arial", 24)

# Rectangles represent the tiles present on the screen
rectangles = []

clock = pygame.time.Clock()

running = True


class Button:
    def __init__(self, x, y, img, width=None, height=None, bg_color=(220,220,220), border_color=(0,0,0), border_radius=15, padding=20):
        self.top_left = (x, y)
        self.img = img
        self.clicked = False
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_radius = border_radius
        self.padding = padding
        # If width/height not given, use img size + padding
        if width is None:
            width = img.get_width() + 2 * padding
        if height is None:
            height = img.get_height() + 2 * padding
        self.rect = pygame.Rect(x, y, width, height)
        self.img_pos = (x + (width - img.get_width()) // 2, y + (height - img.get_height()) // 2)

    def draw(self, screen):
        pos = pygame.mouse.get_pos()
        mouse_over = self.rect.collidepoint(pos)
        # Change color on hover
        color = (200, 200, 255) if mouse_over else self.bg_color
        pygame.draw.rect(screen, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(screen, self.border_color, self.rect, 3, border_radius=self.border_radius)
        screen.blit(self.img, self.img_pos)
        # Only register click on mouse button down, not while held
        if mouse_over:
            if pygame.mouse.get_pressed()[0] == 1 and not getattr(self, '_was_down', False):
                self.clicked = True
                self._was_down = True
                print("CLICKED")
            elif pygame.mouse.get_pressed()[0] == 0:
                self._was_down = False
        else:
            self._was_down = False

    def is_clicked(self):
        if self.clicked:
            self.clicked = False
            return True
        return False

    def reset(self):
        self.clicked = False


# Class rectangle used to represent tiles on the screen
class Rectangle:
    def __init__(self, lane=None, midi_note=None):
        self.w = (WIDTH - 2 * MARGIN) / 8
        # Make all blocks large and easy to see
        self.h = self.w * 2
        self.color = (0, 0, 0)
        if lane is not None:
            self.lane = lane
        else:
            self.lane = random.randint(1, 8)
        self.x = int((self.lane - 1) * self.w + MARGIN)
        self.y = int(-1 * self.h)
        self.last_update_time = pygame.time.get_ticks()
        self.midi_note = midi_note  # Use pitch from song.json
        self.hit = False  # Track if this tile has been hit

    def update(self):
        self.y += TILE_SPEED
        self.last_update_time = pygame.time.get_ticks()
        # Prevent consecutive tiles in the same lane from being tied together
        # Only add to keys_to_press if there is no other active tile in the same lane at the press line
        if PRESS_THRESH - 5 < self.y < PRESS_THRESH + 5:
            lane_finger = None
            if self.lane == 1:
                lane_finger = "LPi"
            elif self.lane == 2:
                lane_finger = "LR"
            elif self.lane == 3:
                lane_finger = "LM"
            elif self.lane == 4:
                lane_finger = "LPo"
            elif self.lane == 5:
                lane_finger = "RPo"
            elif self.lane == 6:
                lane_finger = "RM"
            elif self.lane == 7:
                lane_finger = "RR"
            elif self.lane == 8:
                lane_finger = "RPi"
            # Check if another tile in the same lane is already active at the press line
            already_active = any(key[0] == lane_finger and not key[1].hit for key in keys_to_press)
            if lane_finger and not already_active:
                keys_to_press.add((lane_finger, self))
        if self.y > HEIGHT:
            # Do not reset hit status when the tile goes off-screen
            pass

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

    def can_be_hit(self):
        # Only allow hit if the tile is below the PRESS_THRESH line
        return self.y >= PRESS_THRESH


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

        # Move camera and hand model initialization here for faster launch
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        modelPath = "hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=modelPath)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Load song notes from selected song file
        song_file = getattr(self.gameStateManager, 'selected_song', 'song1.json')
        try:
            with open(song_file, 'r') as file:
                self.song_data = json.load(file)
        except Exception:
            with open("song1.json", 'r') as file:
                self.song_data = json.load(file)
        pitch_to_lane = {}
        unique_pitches = sorted(set(note["pitch"] for note in self.song_data))
        for idx, pitch in enumerate(unique_pitches):
            pitch_to_lane[pitch] = (idx % 8) + 1
        self.notes = []
        for note in self.song_data:
            lane = pitch_to_lane[note["pitch"]]
            self.notes.append({
                "time": note["time"],
                "lane": lane,
                "pitch": note["pitch"]
            })
        # --- Ensure a minimum gap between notes in the same lane ---
        # Use add_gap_between_consecutive_notes to adjust note times
        note_time_lane = [(n["time"], n["lane"]) for n in self.notes]
        adjusted = add_gap_between_consecutive_notes(note_time_lane, min_gap=0.5)
        # Rebuild self.notes with adjusted times, preserving pitch and lane
        for i, (new_time, lane) in enumerate(adjusted):
            self.notes[i]["time"] = new_time
        print(self.notes)
        self.start_time = time.time()
        self.cur_note = 0

    def update_hand_overlay(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detect_result = self.detector.detect(mp_image)
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
        # Only allow hitting the tile if it is below the line and not already hit
        for finger in fingers:
            for key in list(keys_to_press):
                tile = key[1]
                if finger == key[0] and tile.can_be_hit() and not tile.hit:
                    keys_to_press.discard(key)
                    tile.set_color((0, 255, 0))
                    tile.hit = True
                    total_points += 1
                    # Play MIDI note for this lane
                    midi_player.note_on(tile.midi_note, MIDI_VELOCITY)
                    pygame.time.set_timer(pygame.USEREVENT + tile.midi_note, MIDI_DURATION_MS)
                    break

    def run(self):
        global running, rectangles, total_points
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Turn off MIDI notes after duration
            if event.type >= pygame.USEREVENT + 60 and event.type <= pygame.USEREVENT + 71:
                midi_player.note_off(event.type - pygame.USEREVENT, MIDI_VELOCITY)
                pygame.time.set_timer(event.type, 0)
        self.screen.fill((255, 255, 255))
        pygame.draw.line(self.screen, (255, 0, 0), (MARGIN, 0), (MARGIN, HEIGHT))
        pygame.draw.line(self.screen, (255, 0, 0), (WIDTH - MARGIN, 0), (WIDTH - MARGIN, HEIGHT))
        pygame.draw.line(self.screen, (255, 0, 0), (0, PRESS_THRESH), (WIDTH, PRESS_THRESH), 3)
        for i in range(8):
            line_color = (0, 0, 255) if i == 4 else (0, 255, 0)  # 5th from left (i==4) is blue
            pygame.draw.line(self.screen, line_color, (MARGIN + i * (WIDTH - 2 * MARGIN) / 8, 0),
                             (MARGIN + i * (WIDTH - 2 * MARGIN) / 8, HEIGHT))
        cur_time = time.time() - self.start_time
        if self.cur_note < len(self.notes):
            note = self.notes[self.cur_note]
            note_time = note["time"]
            note_lane = note["lane"]
            note_pitch = note["pitch"]
            if cur_time >= note_time:
                print(f"Spawning note at time {cur_time:.2f} (scheduled: {note_time:.2f}) pitch={note_pitch} lane={note_lane}")
                r = Rectangle(lane=note_lane, midi_note=note_pitch)
                rectangles.append(r)
                self.cur_note += 1
        for rect in rectangles:
            rect.update()
            rect.draw(self.screen)
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

    def reset_song(self):
        global rectangles, keys_to_press, total_points
        # Reload song notes from the currently selected song file
        song_file = getattr(self.gameStateManager, 'selected_song', 'song1.json')
        try:
            with open(song_file, 'r') as file:
                self.song_data = json.load(file)
        except Exception:
            with open("song1.json", 'r') as file:
                self.song_data = json.load(file)
        pitch_to_lane = {}
        unique_pitches = sorted(set(note["pitch"] for note in self.song_data))
        for idx, pitch in enumerate(unique_pitches):
            pitch_to_lane[pitch] = (idx % 8) + 1
        self.notes = []
        for note in self.song_data:
            lane = pitch_to_lane[note["pitch"]]
            self.notes.append({
                "time": note["time"],
                "lane": lane,
                "pitch": note["pitch"]
            })
        # Ensure a minimum gap between notes in the same lane
        note_time_lane = [(n["time"], n["lane"]) for n in self.notes]
        adjusted = add_gap_between_consecutive_notes(note_time_lane, min_gap=0.5)
        for i, (new_time, lane) in enumerate(adjusted):
            self.notes[i]["time"] = new_time
        self.start_time = time.time()
        self.cur_note = 0
        rectangles.clear()
        keys_to_press.clear()
        total_points = 0


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
        self.songpicker = SongPicker(self.screen, self.gameStateManager)
        self.states = {'piano': self.piano, 'start': self.start, 'credits': self.credit, 'songpicker': self.songpicker}
        # Make piano accessible from gameStateManager for reset
        self.gameStateManager.piano = self.piano

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
        # SONGS button in the center, moved 100px down and 50px less to the right
        self.songs_button = Button(900, 750, font.render("SONGS", True, (0,0,0)))

    def run(self):
        self.display.blit(self.background, (0, 0))
        self.display.blit(self.logo_image, (800, 100))
        self.start_button.draw(self.display)
        self.credits_button.draw(self.display)
        self.quit_button.draw(self.display)
        self.songs_button.draw(self.display)
        if self.songs_button.is_clicked():
            self.gameStateManager.set_state('songpicker')
            self.songs_button.reset()
        if self.start_button.is_clicked():
            # Reset the piano song state before switching to piano
            if hasattr(self.gameStateManager, 'piano'):
                self.gameStateManager.piano.reset_song()
            self.gameStateManager.set_state('piano')
            self.start_button.reset()
        if self.quit_button.is_clicked():
            pygame.quit()
            sys.exit()
        if self.credits_button.is_clicked():
            self.gameStateManager.set_state('credits')
            self.credits_button.reset()
        pygame.display.update()


class SongPicker:
    def __init__(self, display, gameStateManager):
        self.display = display
        self.gameStateManager = gameStateManager
        self.background = pygame.image.load("piano3.jpg")
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))
        # Explicitly create each button with a custom name and label, styled as nice buttons
        self.song1_button = Button(WIDTH//2 - 200, 200 + 0*80, font.render("song1", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song2_button = Button(WIDTH//2 - 200, 200 + 1*80, font.render("song2", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song3_button = Button(WIDTH//2 - 200, 200 + 2*80, font.render("song3", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song4_button = Button(WIDTH//2 - 200, 200 + 3*80, font.render("song4", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song5_button = Button(WIDTH//2 - 200, 200 + 4*80, font.render("song5", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song6_button = Button(WIDTH//2 - 200, 200 + 5*80, font.render("song6", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song7_button = Button(WIDTH//2 - 200, 200 + 6*80, font.render("song7", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song8_button = Button(WIDTH//2 - 200, 200 + 7*80, font.render("song8", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song9_button = Button(WIDTH//2 - 200, 200 + 8*80, font.render("song9", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song10_button = Button(WIDTH//2 - 200, 200 + 9*80, font.render("song10", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song_buttons = [
            self.song1_button,
            self.song2_button,
            self.song3_button,
            self.song4_button,
            self.song5_button,
            self.song6_button,
            self.song7_button,
            self.song8_button,
            self.song9_button,
            self.song10_button
        ]
        self.back_image = pygame.image.load("back.png").convert_alpha()
        self.back_button = Button(50, 900, pygame.transform.scale(self.back_image, (self.back_image.get_width()//4, self.back_image.get_height()//4)))

    def run(self):
        self.display.blit(self.background, (0, 0))
        title = font.render("Select a Song", True, (0,0,0))
        self.display.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        for idx, btn in enumerate(self.song_buttons):
            btn.draw(self.display)
            if btn.is_clicked():
                self.gameStateManager.selected_song = f"song{idx+1}.json"
                btn.reset()
                self.gameStateManager.set_state('start')
        self.back_button.draw(self.display)
        if self.back_button.is_clicked():
            self.gameStateManager.set_state('start')
            self.back_button.reset()
        pygame.display.update()


class GameStateManager:
    def __init__(self, currentState):
        self.currentState = currentState
        self.selected_song = "song1.json"  # Always default to song1.json

    def set_state(self, newState):
        self.currentState = newState

    def get_state(self):
        return self.currentState

    def get_selected_song(self):
        # Return the currently selected song, defaulting to song1.json only if not set
        return self.selected_song if hasattr(self, 'selected_song') and self.selected_song else 'song1.json'


# --- Add this function to preprocess notes and add a gap between consecutive notes in the same lane ---
def add_gap_between_consecutive_notes(notes, min_gap=0.5):
    # notes: list of (time, lane)
    notes_sorted = sorted(notes, key=lambda x: x[0])
    last_time_per_lane = {}
    adjusted_notes = []
    for time, lane in notes_sorted:
        if lane in last_time_per_lane:
            prev_time = last_time_per_lane[lane]
            if time - prev_time < min_gap:
                time = prev_time + min_gap
        last_time_per_lane[lane] = time
        adjusted_notes.append((time, lane))
    return adjusted_notes


if __name__ == '__main__':
    game = Game()
    game.run()
