import time
import os
import pygame
import sys
import random
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json

WIDTH = 1750
HEIGHT = 1000
MARGIN = 150
FPS = 240
TILE_SPEED = 10

screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.SCALED)
pygame.init()
pygame.mixer.init()
MIDI_VELOCITY = 127
MIDI_DURATION_MS = 200

NON_PINKY_DIST = 0.3
PINKY_DIST = 0.25
PRESS_THRESH = 500

fingers = set()
keys_to_press = set()
total_points = 0

font = pygame.font.SysFont("Arial", 24)

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
        if width is None:
            width = img.get_width() + 2 * padding
        if height is None:
            height = img.get_height() + 2 * padding
        self.rect = pygame.Rect(x, y, width, height)
        self.img_pos = (x + (width - img.get_width()) // 2, y + (height - img.get_height()) // 2)

    def draw(self, screen):
        pos = pygame.mouse.get_pos()
        mouse_over = self.rect.collidepoint(pos)
        color = (200, 200, 255) if mouse_over else self.bg_color
        pygame.draw.rect(screen, color, self.rect, border_radius=self.border_radius)
        pygame.draw.rect(screen, self.border_color, self.rect, 3, border_radius=self.border_radius)
        screen.blit(self.img, self.img_pos)
        if mouse_over:
            if pygame.mouse.get_pressed()[0] == 1 and not getattr(self, '_was_down', False):
                self.clicked = True
                self._was_down = True
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

class Rectangle:
    def __init__(self, lane=None, note_file=None, note_time=None):
        self.w = (WIDTH - 2 * MARGIN) / 8
        self.h = self.w * 2
        self.color = (0, 0, 0)
        if lane is not None:
            self.lane = lane
        else:
            self.lane = random.randint(1, 8)
        self.x = int((self.lane - 1) * self.w + MARGIN)
        self.y = int(-1 * self.h)
        self.last_update_time = pygame.time.get_ticks()
        self.note_file = note_file
        self.note_time = note_time  # Store the timing for this note instance
        self.hit = False

    def update(self):
        self.y += TILE_SPEED
        self.last_update_time = pygame.time.get_ticks()
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
            already_active = any(key[0] == lane_finger and not key[1].hit for key in keys_to_press)
            if lane_finger and not already_active:
                keys_to_press.add((lane_finger, self))
        if self.y > HEIGHT:
            pass

    def draw(self, surface):
        r = pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x), int(self.y)), (int(self.x + self.w), int(self.y)), 3)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x), int(self.y)), (int(self.x), int(self.y + self.h)), 3)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x + self.w), int(self.y)),
                         (int(self.x + self.w), int(self.y + self.h)), 3)
        pygame.draw.line(surface, (255, 0, 0), (int(self.x), int(self.y + self.h)),
                         (int(self.x + self.w), int(self.y + self.h)), 3)

    def get_point(self):
        return self.x, self.y

    def set_color(self, color):
        self.color = color

    def can_be_hit(self):
        return self.y >= PRESS_THRESH

class Piano:
    def __init__(self, display, gameStateManager):
        self.screen = display
        self.gameStateManager = gameStateManager
        self.frames = 0
        self.back_image = pygame.image.load(os.path.join("images", "back.png")).convert_alpha()
        back_btn_width = int(self.back_image.get_width() * 0.4)
        back_btn_height = int(self.back_image.get_height() * 0.4)
        self.back_button = Button(5, 800, pygame.transform.scale(self.back_image,
                                                                 (back_btn_width,
                                                                  back_btn_height)))
        self.last_overlay = None
        self.last_overlay_frame = -1
        self.last_frame = None
        self.last_detect_result = None
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        modelPath = os.path.join("dep", "hand_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=modelPath)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.note_sounds = {}
        self.reset_song()
        self.last_hit_info = None  # (time, note)
        self.last_hit_time = None

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
        global keys_to_press, fingers, total_points
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
            for key in list(keys_to_press):
                tile = key[1]
                if finger == key[0] and tile.can_be_hit() and not tile.hit:
                    keys_to_press.discard(key)
                    tile.set_color((0, 255, 0))
                    tile.hit = True
                    total_points += 1
                    note_name = tile.note_file
                    if note_name in self.note_sounds:
                        self.note_sounds[note_name].play()
                    # Track last hit info
                    self.last_hit_info = (time.time() - self.start_time, note_name)
                    self.last_hit_time = time.time()
                    break

    def run(self):
        global running, rectangles, total_points
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        self.screen.fill((255, 255, 255))
        # Draw left margin line in red
        pygame.draw.line(self.screen, (255, 0, 0), (MARGIN, 0), (MARGIN, HEIGHT), 4)
        # Draw right margin line (keep as is)
        pygame.draw.line(self.screen, (255, 0, 0), (WIDTH - MARGIN, 0), (WIDTH - MARGIN, HEIGHT))
        pygame.draw.line(self.screen, (255, 0, 0), (0, PRESS_THRESH), (WIDTH, PRESS_THRESH), 3)
        for i in range(8):
            line_color = (0, 0, 255) if i == 4 else (0, 255, 0)
            pygame.draw.line(self.screen, line_color, (MARGIN + i * (WIDTH - 2 * MARGIN) / 8, 0),
                             (MARGIN + i * (WIDTH - 2 * MARGIN) / 8, HEIGHT))
        cur_time = time.time() - self.start_time
        if self.cur_note < len(self.notes):
            note = self.notes[self.cur_note]
            note_time = note["time"]
            note_lane = note["lane"]
            note_pitch = note["note"]
            if cur_time >= note_time:
                r = Rectangle(lane=note_lane, note_file=note_pitch, note_time=note_time)
                rectangles.append(r)
                self.cur_note += 1
        for rect in rectangles:
            rect.update()
            rect.draw(self.screen)
        # Draw finger names at the top of each lane (overlay on tiles)
        lane_finger_names = [
            "Left Pinky", "Left Ring", "Left Middle", "Left Pointer",
            "Right Pointer", "Right Middle", "Right Ring", "Right Pinky"
        ]
        label_font = pygame.font.SysFont("Arial", 20, bold=True)
        for i in range(8):
            lane_x = int(MARGIN + i * (WIDTH - 2 * MARGIN) / 8)
            label = lane_finger_names[i]
            label_surface = label_font.render(label, True, (0, 40, 225))
            label_rect = label_surface.get_rect()
            label_rect.center = (lane_x + ((WIDTH - 2 * MARGIN) / 16), 30)
            self.screen.blit(label_surface, label_rect)
        score_percent = (self.hit_notes / self.total_notes) * 100 if self.total_notes > 0 else 0
        text_surface = font.render(f"SCORE: {score_percent:.1f}%", True, (0, 0, 0))
        text_rect = text_surface.get_rect()
        text_rect.center = (60, 50)
        screen.blit(text_surface, text_rect)
        self.update_hand_overlay()
        frame = self.last_frame
        detect_result = self.last_detect_result
        overlay = self.last_overlay
        first_unhit = next((rect for rect in rectangles if not rect.hit), None)
        if frame is not None and detect_result is not None and first_unhit is not None:
            self.try_hit_note(frame, detect_result, first_unhit)
        if overlay is not None:
            self.screen.blit(overlay, (0, 0))
        self.back_button.draw(self.screen)
        if self.back_button.is_clicked():
            self.gameStateManager.set_state('start')
            self.back_button.reset()
        for rect in rectangles:
            if rect.y > HEIGHT and not rect.hit:
                if hasattr(self.gameStateManager, 'game'):
                    self.hit_notes = sum(1 for r in rectangles if r.hit)
                    score_percent = (self.hit_notes / self.total_notes) * 100 if self.total_notes > 0 else 0
                    if all(r.hit for r in rectangles) and self.cur_note >= self.total_notes:
                        self.gameStateManager.game.set_win(score_percent)
                    else:
                        self.gameStateManager.game.set_game_over(score_percent)
                return
        if self.cur_note >= self.total_notes and all(rect.hit or rect.y > HEIGHT for rect in rectangles):
            if hasattr(self.gameStateManager, 'game'):
                score_percent = (self.hit_notes / self.total_notes) * 100 if self.total_notes > 0 else 0
                if all(r.hit for r in rectangles) and self.cur_note >= self.total_notes:
                    self.gameStateManager.game.set_win(score_percent)
                else:
                    self.gameStateManager.game.set_game_over(score_percent)
        # Show last hit info in top left, always visible, with -- around it and timing from json
        hit_text_str = "-- Last Hit: "
        if self.last_hit_info:
            note_time, hit_note = self.last_hit_info
            if note_time is not None:
                hit_text_str += f"{hit_note} @ {note_time:.2f}s"
            else:
                hit_text_str += f"{hit_note}"
        else:
            hit_text_str += "None"
        hit_text_str += " --"
        small_font = pygame.font.SysFont("Arial", 18)
        hit_text = small_font.render(hit_text_str, True, (80, 80, 80))
        self.screen.blit(hit_text, (10, 10))
        pygame.display.flip()
        self.frames += 1

    def try_hit_note(self, frame, detect_result, rect):
        lane_finger_map = {1: "LPi", 2: "LR", 3: "LM", 4: "LPo", 5: "RPo", 6: "RM", 7: "RR", 8: "RPi"}
        needed_finger = lane_finger_map.get(rect.lane)
        leftHand = rightHand = None
        handedness = detect_result.handedness
        for i in range(len(handedness)):
            if handedness[i][0].category_name == "Left":
                leftHand = i
            elif handedness[i][0].category_name == "Right":
                rightHand = i
        Rwrist = Lwrist = None
        Rfingers = [None]*5
        Lfingers = [None]*5
        if rightHand is not None:
            Rwrist = detect_result.hand_landmarks[rightHand][0].y
            for idx, tip in enumerate([4,8,12,16,20]):
                Rfingers[idx] = abs(Rwrist - detect_result.hand_landmarks[rightHand][tip].y)
        if leftHand is not None:
            Lwrist = detect_result.hand_landmarks[leftHand][0].y
            for idx, tip in enumerate([4,8,12,16,20]):
                Lfingers[idx] = abs(Lwrist - detect_result.hand_landmarks[leftHand][tip].y)
        pressed = False
        if needed_finger == "RPo" and Rfingers[1] is not None and Rfingers[1] < NON_PINKY_DIST:
            pressed = True
        elif needed_finger == "RM" and Rfingers[2] is not None and Rfingers[2] < NON_PINKY_DIST:
            pressed = True
        elif needed_finger == "RR" and Rfingers[3] is not None and Rfingers[3] < NON_PINKY_DIST:
            pressed = True
        elif needed_finger == "RPi" and Rfingers[4] is not None and Rfingers[4] < PINKY_DIST:
            pressed = True
        elif needed_finger == "LPo" and Lfingers[1] is not None and Lfingers[1] < NON_PINKY_DIST:
            pressed = True
        elif needed_finger == "LM" and Lfingers[2] is not None and Lfingers[2] < NON_PINKY_DIST:
            pressed = True
        elif needed_finger == "LR" and Lfingers[3] is not None and Lfingers[3] < NON_PINKY_DIST:
            pressed = True
        elif needed_finger == "LPi" and Lfingers[4] is not None and Lfingers[4] < PINKY_DIST:
            pressed = True
        if pressed and rect.can_be_hit() and not rect.hit:
            rect.set_color((0, 255, 0))
            rect.hit = True
            global total_points
            total_points += 1
            note_name = rect.note_file
            if note_name in self.note_sounds:
                self.note_sounds[note_name].play()
            # Track last hit info with correct timing from the note object
            self.last_hit_info = (rect.note_time, note_name)
            self.last_hit_time = time.time()

    def reset_song(self):
        global rectangles, keys_to_press, total_points
        song_file = self.gameStateManager.get_selected_song()
        try:
            with open(song_file, 'r') as file:
                self.song_data = json.load(file)
        except Exception:
            with open(self.gameStateManager.get_selected_song(), 'r') as file:
                self.song_data = json.load(file)
        valid_notes = [note for note in self.song_data if isinstance(note, dict) and 'note' in note and 'time' in note]
        if not valid_notes:
            print(f"Error: No valid notes found in {song_file}. Check file format.")
            self.notes = []
            self.total_notes = 0
            self.hit_notes = 0
            self.note_sounds = {}
            return
        pitch_to_lane = {}
        unique_notes = sorted(set(note["note"] for note in valid_notes))
        for idx, note_name in enumerate(unique_notes):
            pitch_to_lane[note_name] = (idx % 8) + 1
        self.notes = []
        last_lane = None
        for note in valid_notes:
            lane = pitch_to_lane[note["note"]]
            if last_lane == lane:
                lane = (lane % 8) + 1
                if lane == last_lane:
                    lane = ((lane + 1) % 8) + 1
            self.notes.append({
                "time": note["time"],
                "lane": lane,
                "note": note["note"]
            })
            last_lane = lane
        note_time_lane = [(n["time"], n["lane"]) for n in self.notes]
        adjusted = add_gap_between_consecutive_notes(note_time_lane, min_gap=0.8)
        for i, (new_time, lane) in enumerate(adjusted):
            self.notes[i]["time"] = new_time
            self.notes[i]["lane"] = lane
        self.start_time = time.time()
        self.cur_note = 0
        rectangles.clear()
        keys_to_press.clear()
        total_points = 0
        self.total_notes = len(self.notes)
        self.hit_notes = 0
        self.note_sounds = {}
        for note_name in unique_notes:
            sound_path = os.path.join("pianotes", note_name)
            if not os.path.isfile(sound_path):
                print(f"Missing sound file for note: {note_name} at {sound_path}")
            try:
                self.note_sounds[note_name] = pygame.mixer.Sound(sound_path)
            except Exception as e:
                print(f"Could not load sound for {note_name}: {e}")

class GameOver:
    def __init__(self, display, gameStateManager, score_percent):
        self.display = display
        self.gameStateManager = gameStateManager
        self.score_percent = score_percent
        self.font = pygame.font.SysFont('Arial', 80, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 40)
        self.back_image = pygame.image.load(os.path.join("images", "back.png")).convert_alpha()
        self.back_button = Button(100, 800, pygame.transform.scale(self.back_image, (self.back_image.get_width() // 2, self.back_image.get_height() // 2)))
        self.start_time = None
        self.death_notes = [
            "F3.mp3", "A3.mp3", "Eb4.mp3", "Ab3.mp3", "B2.mp3", "E4.mp3"
        ]
        self.note_index = 0
        self.last_note_time = None
        self.notes_finished = False

    def run(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.display.fill((0, 0, 0))
        text_surface = self.font.render("GAME OVER", True, (255, 0, 0))
        score_surface = self.small_font.render(f"Score: {self.score_percent:.1f}%", True, (255, 255, 255))
        self.display.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, 300))
        self.display.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, 450))
        self.back_button.draw(self.display)
        if self.back_button.is_clicked():
            self.gameStateManager.set_state('start')
            self.back_button.reset()
        pygame.display.update()
        if not self.notes_finished and self.start_time and time.time() - self.start_time > 0.2:
            if self.last_note_time is None:
                self.last_note_time = time.time()
            if self.note_index < len(self.death_notes):
                if time.time() - self.last_note_time > 0.5:
                    note = self.death_notes[self.note_index]
                    try:
                        sound = pygame.mixer.Sound(os.path.join("pianotes", note))
                        sound.play()
                    except Exception as e:
                        print(f"Could not play {note}: {e}")
                    self.note_index += 1
                    self.last_note_time = time.time()
            else:
                self.notes_finished = True

class WinScreen:
    def __init__(self, display, gameStateManager, score_percent):
        self.display = display
        self.gameStateManager = gameStateManager
        self.score_percent = score_percent
        self.font = pygame.font.SysFont('Arial', 80, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 40)
        self.back_image = pygame.image.load(os.path.join("images", "back.png")).convert_alpha()
        self.back_button = Button(100, 800, pygame.transform.scale(self.back_image, (self.back_image.get_width() // 2, self.back_image.get_height() // 2)))
        self.start_time = None
        time.sleep(0.5)
        self.win_notes = [
            "C4.mp3", "E4.mp3", "G4.mp3", "C5.mp3", "B4.mp3", "G4.mp3", "A4.mp3", "F4.mp3"
        ]
        self.notes_played = False
        self.dog_bg = pygame.image.load(os.path.join("images", "dog.png"))
        self.dog_bg = pygame.transform.scale(self.dog_bg, (WIDTH, HEIGHT))

    def play_win_notes(self):
        for note in self.win_notes:
            try:
                sound = pygame.mixer.Sound(os.path.join("pianotes", note))
                sound.play()
            except Exception as e:
                print(f"Could not play {note}: {e}")
            time.sleep(0.4)

    def run(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.display.blit(self.dog_bg, (0, 0))
        big_font = pygame.font.SysFont('Arial', 120, bold=True)
        big_text_surface = big_font.render("YOU WIN!", True, (0, 255, 0))
        self.display.blit(big_text_surface, (WIDTH // 2 - big_text_surface.get_width() // 2, 220))
        score_font = pygame.font.SysFont('Arial', 60, bold=True)
        score_surface = score_font.render("Score: 100%", True, (0, 0, 255))
        self.display.blit(score_surface, (WIDTH // 2 - score_surface.get_width() // 2, 400))
        self.back_button.draw(self.display)
        if self.back_button.is_clicked():
            self.gameStateManager.set_state('start')
            self.back_button.reset()
        pygame.display.update()
        if not self.notes_played and self.start_time and time.time() - self.start_time > 0.2:
            self.play_win_notes()
            self.notes_played = True

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
        self.gameover = None
        self.winscreen = None
        self.states = {'piano': self.piano, 'start': self.start, 'credits': self.credit, 'songpicker': self.songpicker}
        self.gameStateManager.piano = self.piano
        self.gameStateManager.game = self

    def set_game_over(self, score_percent):
        self.gameover = GameOver(self.screen, self.gameStateManager, score_percent)
        self.states['gameover'] = self.gameover
        self.gameStateManager.set_state('gameover')

    def set_win(self, score_percent):
        self.winscreen = WinScreen(self.screen, self.gameStateManager, score_percent)
        self.states['winscreen'] = self.winscreen
        self.gameStateManager.set_state('winscreen')

    def run(self):
        frames = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            state = self.gameStateManager.get_state()
            if state == 'winscreen' and self.winscreen:
                self.winscreen.run()
            elif state == 'gameover' and self.gameover:
                self.gameover.run()
            else:
                self.states[state].run()
            self.clock.tick(240)

class Credits:
    def __init__(self, display, gameStateManager):
        self.display = display
        self.gameStateManager = gameStateManager
        self.my_font = pygame.font.SysFont('Arial', 30, bold=True)
        self.back_image = pygame.image.load(os.path.join("images", "back.png")).convert_alpha()
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
        self.background = pygame.image.load(os.path.join("images", "piano3.jpg"))
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))
        self.start_image = pygame.image.load(os.path.join("images", "play.png")).convert_alpha()
        self.start_button = Button(100, 100, pygame.transform.scale(self.start_image,
                                                                    (self.start_image.get_width() / 2,
                                                                     self.start_image.get_height() / 2)))
        self.credits_image = pygame.image.load(os.path.join("images", "credits.png")).convert_alpha()
        self.credits_button = Button(100, 400, pygame.transform.scale(self.credits_image,
                                                                      (self.credits_image.get_width() / 2,
                                                                       self.credits_image.get_height() / 2)))
        self.quit_image = pygame.image.load(os.path.join("images", "quit.png")).convert_alpha()
        self.quit_button = Button(100, 700, pygame.transform.scale(self.quit_image,
                                                                   (self.quit_image.get_width() / 2,
                                                                    self.quit_image.get_height() / 2)))
        self.logo_image = pygame.image.load(os.path.join("images", "LOGO.png")).convert_alpha()
        self.logo_image = pygame.transform.scale(self.logo_image,
                                                 (self.logo_image.get_width() * 8,
                                                  self.logo_image.get_height() * 8))
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
        self.background = pygame.image.load(os.path.join("images", "piano3.jpg"))
        self.background = pygame.transform.scale(self.background, (WIDTH, HEIGHT))
        self.song1_button = Button(WIDTH//2 - 200, 200 + 0*80, font.render("Happy Birthday", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song2_button = Button(WIDTH//2 - 200, 200 + 1*80, font.render("???", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song3_button = Button(WIDTH//2 - 200, 200 + 2*80, font.render("Tetris", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song4_button = Button(WIDTH//2 - 200, 200 + 3*80, font.render("Megalovania", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song5_button = Button(WIDTH//2 - 200, 200 + 4*80, font.render("Twinkle Twinkle", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song6_button = Button(WIDTH//2 - 200, 200 + 5*80, font.render("Mary Had a Little Lamb", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song7_button = Button(WIDTH//2 - 200, 200 + 6*80, font.render("Jingle Bells", True, (0,0,80)), width=400, height=60, bg_color=(255,255,255), border_color=(0,0,120))
        self.song_buttons = [
            self.song1_button,
            self.song2_button,
            self.song3_button,
            self.song4_button,
            self.song5_button,
            self.song6_button,
            self.song7_button
        ]
        self.back_image = pygame.image.load(os.path.join("images", "back.png")).convert_alpha()
        self.back_button = Button(50, 900, pygame.transform.scale(self.back_image, (self.back_image.get_width()//4, self.back_image.get_height()//4)))

    def run(self):
        self.display.blit(self.background, (0, 0))
        title = font.render("Select a Song", True, (0,0,0))
        self.display.blit(title, (WIDTH//2 - title.get_width()//2, 100))
        for idx, btn in enumerate(self.song_buttons):
            btn.draw(self.display)
            if btn.is_clicked():
                self.gameStateManager.set_selected_song(os.path.join("songs", f"song{idx+1}.json"))
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
        self.selected_song = os.path.join("songs", "song1.json")

    def set_state(self, newState):
        self.currentState = newState
        if newState == 'piano' and hasattr(self, 'piano'):
            self.piano.reset_song()

    def get_state(self):
        return self.currentState

    def get_selected_song(self):
        return self.selected_song if hasattr(self, 'selected_song') and self.selected_song else self.selected_song

    def set_selected_song(self, song_path):
        self.selected_song = song_path

def add_gap_between_consecutive_notes(notes, min_gap=0.5):
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
