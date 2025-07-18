import pygame, random, sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ========== Mediapipe Setup ==========
modelPath = "MediapipeModels/hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=modelPath)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# ========== Constants ==========
pygame.init()
NON_PINKY_DIST = 0.3
PINKY_DIST = 0.25
POSSIBLE_FINGERS = {"RPo", "RM", "RR", "RPi", "LPo", "LM", "LR", "LPi"}
fingers = set()
keys_to_press = random.sample(list(POSSIBLE_FINGERS), random.randint(1, 3))
completed = False
info = pygame.display.Info()
W, H = info.current_w, info.current_h
GAME_H = int(H * 0.65)  # Game takes top 65%
CAM_H = H - GAME_H      # Camera takes bottom 35%
COLS = 8
TILE_W = W // COLS
TILE_H = GAME_H // 6
SPEED = 1
SPAWN_MS = 700
BONUS_LINE_Y = GAME_H - 200
DEATH_LINE_Y = GAME_H

font = pygame.font.SysFont(None, 50)
screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
clock = pygame.time.Clock()

# ========== Camera Setup ==========
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# ========== Tile Class ==========
class Tile:
    def __init__(self, col):
        self.rect = pygame.Rect(col * TILE_W, -TILE_H, TILE_W, TILE_H)

    def move(self): self.rect.y += SPEED
    def draw(self): pygame.draw.rect(screen, (0, 0, 0), self.rect)
    def clicked(self, pos): return self.rect.collidepoint(pos)
    def is_dead(self): return self.rect.bottom >= DEATH_LINE_Y
    def in_bonus_zone(self): return BONUS_LINE_Y <= self.rect.bottom < DEATH_LINE_Y

# ========== Main Game Loop ==========
def main():
    global completed, fingers, keys_to_press
    tiles = []
    score = 0
    last_spawn = pygame.time.get_ticks()
    paused = True

    while True:
        screen.fill((255, 255, 255))

        # Handle events
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                cap.release()
                pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN and not paused:
                for t in tiles[:]:
                    if t.clicked(e.pos):
                        score += 2 if t.in_bonus_zone() else 1
                        tiles.remove(t)
                        break

        # Read and show camera feed
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = detector.detect(mp_image)
            hands = detection_result.hand_landmarks
            paused = len(hands) != 2

            if not paused:
                if completed:
                    keys_to_press = random.sample(list(POSSIBLE_FINGERS), random.randint(1, 3))
                    completed = False
                else:
                    print("KEYS TO PRESS:", keys_to_press)

                fingers.clear()

                def get_finger_diffs(wrist_y, lms):
                    return [abs(wrist_y - lms[i].y) for i in [4, 8, 12, 16, 20]]

                Rwrist = hands[0][0].y
                Rf = get_finger_diffs(Rwrist, hands[0])
                if Rf[1] < NON_PINKY_DIST: fingers.add("RPo")
                if Rf[2] < NON_PINKY_DIST: fingers.add("RM")
                if Rf[3] < NON_PINKY_DIST: fingers.add("RR")
                if Rf[4] < PINKY_DIST: fingers.add("RPi")

                Lwrist = hands[1][0].y
                Lf = get_finger_diffs(Lwrist, hands[1])
                if Lf[1] < NON_PINKY_DIST: fingers.add("LPo")
                if Lf[2] < NON_PINKY_DIST: fingers.add("LM")
                if Lf[3] < NON_PINKY_DIST: fingers.add("LR")
                if Lf[4] < PINKY_DIST:      fingers.add("LPi")

                print("FINGERS PRESSED:", fingers)
                all_pressed = all(k in fingers for k in keys_to_press)
                completed = all_pressed

            frame_annotated = draw_hand_landmarks(frame_rgb.copy(), detection_result)
            h = frame_annotated.shape[0]
            cropped = frame_annotated[:int(h * 0.35), :, :]
            cropped = np.rot90(cropped)
            surface = pygame.surfarray.make_surface(cropped)
            surface = pygame.transform.scale(surface, (W, CAM_H))
            screen.blit(surface, (0, GAME_H))

        # Game logic only runs when not paused
        if not paused:
            if pygame.time.get_ticks() - last_spawn > SPAWN_MS:
                tiles.append(Tile(random.randint(0, COLS - 1)))
                last_spawn = pygame.time.get_ticks()

            for t in tiles[:]:
                t.move()
                t.draw()
                if t.is_dead():
                    print("Game Over! Final Score:", score)
                    cap.release()
                    pygame.quit(); sys.exit()

        # Grid lines
        for i in range(1, COLS):
            pygame.draw.line(screen, (200, 200, 200), (i * TILE_W, 0), (i * TILE_W, GAME_H), 1)

        # Bonus and death lines
        pygame.draw.line(screen, (255, 165, 0), (0, BONUS_LINE_Y), (W, BONUS_LINE_Y), 2)
        pygame.draw.line(screen, (255, 0, 0), (0, DEATH_LINE_Y), (W, DEATH_LINE_Y), 3)

        # Score
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(score_text, score_text.get_rect(center=(W // 2, GAME_H + CAM_H // 2)))

        # Pause message
        if paused:
            pause_msg = font.render("Show 2 hands to continue", True, (255, 0, 0))
            screen.blit(pause_msg, pause_msg.get_rect(center=(W // 2, GAME_H + 50)))

        pygame.display.flip()
        clock.tick(60)

# ========== Landmark Drawing Without Thumb ==========
def draw_hand_landmarks(image, detection_result):
    if not detection_result.hand_landmarks:
        return image

    thumb_indices = {1, 2, 3, 4}
    thumb_connections = {(0, 1), (1, 2), (2, 3), (3, 4)}

    for hand_landmarks in detection_result.hand_landmarks:
        for idx, lm in enumerate(hand_landmarks):
            if idx in thumb_indices:
                continue
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        connections = mp.solutions.hands.HAND_CONNECTIONS
        for connection in connections:
            if connection in thumb_connections:
                continue
            start_idx, end_idx = connection
            if start_idx in thumb_indices or end_idx in thumb_indices:
                continue
            start_lm = hand_landmarks[start_idx]
            end_lm = hand_landmarks[end_idx]
            x1 = int(start_lm.x * image.shape[1])
            y1 = int(start_lm.y * image.shape[0])
            x2 = int(end_lm.x * image.shape[1])
            y2 = int(end_lm.y * image.shape[0])
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return image

if __name__ == "__main__":
    main()