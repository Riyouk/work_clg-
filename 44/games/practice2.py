import pygame
import random
import math
import time

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Escape the Monster")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Clock for controlling game speed
clock = pygame.time.Clock()
FPS = 60

# Font setup
font = pygame.font.SysFont('Arial', 24)
big_font = pygame.font.SysFont('Arial', 48)

# Game variables
score = 0
level = 1
game_over = False
game_won = False
start_time = time.time()

# Player class
class Player:
    def __init__(self):
        self.width = 30
        self.height = 30
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.color = BLUE
        self.stamina = 100
        self.max_stamina = 100
        self.stamina_recovery = 0.5
        self.sprint_depletion = 1.5
        self.is_sprinting = False
        self.sprint_multiplier = 1.7
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    
    def move(self, keys):
        # Handle sprinting
        self.is_sprinting = keys[pygame.K_LSHIFT] and self.stamina > 0
        current_speed = self.speed * self.sprint_multiplier if self.is_sprinting else self.speed
        
        # Update stamina
        if self.is_sprinting:
            self.stamina = max(0, self.stamina - self.sprint_depletion)
        else:
            self.stamina = min(self.max_stamina, self.stamina + self.stamina_recovery)
        
        # Movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.x = max(0, self.x - current_speed)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.x = min(WIDTH - self.width, self.x + current_speed)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.y = max(0, self.y - current_speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.y = min(HEIGHT - self.height, self.y + current_speed)
        
        # Update rectangle position
        self.rect.x = self.x
        self.rect.y = self.y
    
    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)
        # Draw eyes to indicate direction
        pygame.draw.circle(screen, WHITE, (self.x + 10, self.y + 10), 5)
        pygame.draw.circle(screen, WHITE, (self.x + 20, self.y + 10), 5)
        pygame.draw.circle(screen, BLACK, (self.x + 10, self.y + 10), 2)
        pygame.draw.circle(screen, BLACK, (self.x + 20, self.y + 10), 2)
        
        # Draw stamina bar
        stamina_width = 50
        stamina_height = 5
        stamina_x = self.x - 10
        stamina_y = self.y - 10
        
        # Background bar (empty stamina)
        pygame.draw.rect(screen, RED, (stamina_x, stamina_y, stamina_width, stamina_height))
        # Filled stamina
        filled_width = int((self.stamina / self.max_stamina) * stamina_width)
        pygame.draw.rect(screen, GREEN, (stamina_x, stamina_y, filled_width, stamina_height))

# Monster class
class Monster:
    def __init__(self, player):
        self.width = 40
        self.height = 40
        # Start at a random edge of the screen
        edge = random.randint(0, 3)
        if edge == 0:  # Top
            self.x = random.randint(0, WIDTH - self.width)
            self.y = -self.height
        elif edge == 1:  # Right
            self.x = WIDTH
            self.y = random.randint(0, HEIGHT - self.height)
        elif edge == 2:  # Bottom
            self.x = random.randint(0, WIDTH - self.width)
            self.y = HEIGHT
        else:  # Left
            self.x = -self.width
            self.y = random.randint(0, HEIGHT - self.height)
            
        self.base_speed = 2 + (level * 0.5)  # Speed increases with level
        self.speed = self.base_speed
        self.color = RED
        self.player = player
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.last_speed_boost = time.time()
        self.boost_cooldown = 5  # Seconds between speed boosts
        self.boost_duration = 1.5  # Seconds the boost lasts
        self.is_boosting = False
        self.boost_start_time = 0
    
    def move(self):
        # Calculate direction to player
        dx = self.player.x - self.x
        dy = self.player.y - self.y
        distance = max(1, math.sqrt(dx*dx + dy*dy))  # Avoid division by zero
        dx = dx / distance
        dy = dy / distance
        
        # Check if it's time for a speed boost
        current_time = time.time()
        if not self.is_boosting and current_time - self.last_speed_boost > self.boost_cooldown:
            self.is_boosting = True
            self.boost_start_time = current_time
            self.speed = self.base_speed * 2  # Double speed during boost
            self.last_speed_boost = current_time
        
        # Check if boost should end
        if self.is_boosting and current_time - self.boost_start_time > self.boost_duration:
            self.is_boosting = False
            self.speed = self.base_speed
        
        # Move towards player
        self.x += dx * self.speed
        self.y += dy * self.speed
        
        # Update rectangle position
        self.rect.x = self.x
        self.rect.y = self.y
    
    def draw(self):
        # Draw monster body
        pygame.draw.rect(screen, self.color, self.rect)
        
        # Draw eyes (red if boosting, yellow otherwise)
        eye_color = YELLOW if not self.is_boosting else WHITE
        pygame.draw.circle(screen, eye_color, (self.x + 10, self.y + 15), 5)
        pygame.draw.circle(screen, eye_color, (self.x + 30, self.y + 15), 5)
        
        # Draw mouth
        if self.is_boosting:
            # Angry mouth when boosting
            pygame.draw.line(screen, BLACK, (self.x + 10, self.y + 30), (self.x + 30, self.y + 30), 3)
        else:
            # Normal mouth
            pygame.draw.arc(screen, BLACK, (self.x + 10, self.y + 20, 20, 15), 0, math.pi, 3)

# Obstacle class
class Obstacle:
    def __init__(self):
        self.width = random.randint(30, 80)
        self.height = random.randint(30, 80)
        self.x = random.randint(0, WIDTH - self.width)
        self.y = random.randint(0, HEIGHT - self.height)
        self.color = (100, 100, 100)  # Gray
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

# Powerup class
class Powerup:
    def __init__(self):
        self.width = 20
        self.height = 20
        self.x = random.randint(0, WIDTH - self.width)
        self.y = random.randint(0, HEIGHT - self.height)
        self.color = YELLOW
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.type = random.choice(['speed', 'stamina', 'freeze'])
        self.collected = False
        self.spawn_time = time.time()
        self.lifetime = 10  # Powerups disappear after 10 seconds
    
    def draw(self):
        if not self.collected and time.time() - self.spawn_time < self.lifetime:
            pygame.draw.rect(screen, self.color, self.rect)
            # Draw an icon based on powerup type
            if self.type == 'speed':
                # Draw lightning bolt
                pygame.draw.line(screen, BLACK, (self.x + 10, self.y + 5), (self.x + 5, self.y + 10), 2)
                pygame.draw.line(screen, BLACK, (self.x + 5, self.y + 10), (self.x + 15, self.y + 10), 2)
                pygame.draw.line(screen, BLACK, (self.x + 15, self.y + 10), (self.x + 10, self.y + 15), 2)
            elif self.type == 'stamina':
                # Draw heart
                pygame.draw.circle(screen, RED, (self.x + 7, self.y + 7), 5)
                pygame.draw.circle(screen, RED, (self.x + 13, self.y + 7), 5)
                pygame.draw.polygon(screen, RED, [(self.x + 3, self.y + 9), (self.x + 10, self.y + 17), (self.x + 17, self.y + 9)])
            elif self.type == 'freeze':
                # Draw snowflake
                pygame.draw.circle(screen, WHITE, (self.x + 10, self.y + 10), 7)
                pygame.draw.line(screen, BLUE, (self.x + 5, self.y + 5), (self.x + 15, self.y + 15), 2)
                pygame.draw.line(screen, BLUE, (self.x + 15, self.y + 5), (self.x + 5, self.y + 15), 2)

# Create game objects
player = Player()
monster = Monster(player)
obstacles = [Obstacle() for _ in range(5)]
powerups = []
last_powerup_time = time.time()
powerup_spawn_interval = 10  # Seconds between powerup spawns
monster_freeze_end_time = 0

# Game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and (game_over or game_won):
                # Reset game
                player = Player()
                monster = Monster(player)
                obstacles = [Obstacle() for _ in range(5)]
                powerups = []
                score = 0
                level = 1
                game_over = False
                game_won = False
                start_time = time.time()
                last_powerup_time = time.time()
    
    # Clear screen
    screen.fill(BLACK)
    
    # Update game if not over
    if not game_over and not game_won:
        # Get keyboard input
        keys = pygame.key.get_pressed()
        
        # Move player
        player.move(keys)
        
        # Move monster if not frozen
        current_time = time.time()
        if current_time > monster_freeze_end_time:
            monster.move()
        
        # Check for collision with monster
        if player.rect.colliderect(monster.rect):
            game_over = True
        
        # Check for collision with obstacles
        for obstacle in obstacles:
            if player.rect.colliderect(obstacle.rect):
                # Push player back
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    player.x += player.speed
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    player.x -= player.speed
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    player.y += player.speed
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    player.y -= player.speed
                player.rect.x = player.x
                player.rect.y = player.y
        
        # Check for collision with powerups
        for powerup in powerups:
            if not powerup.collected and time.time() - powerup.spawn_time < powerup.lifetime:
                if player.rect.colliderect(powerup.rect):
                    powerup.collected = True
                    if powerup.type == 'speed':
                        player.speed += 1  # Permanent speed boost
                    elif powerup.type == 'stamina':
                        player.stamina = player.max_stamina  # Refill stamina
                        player.max_stamina += 20  # Increase max stamina
                    elif powerup.type == 'freeze':
                        monster_freeze_end_time = time.time() + 3  # Freeze monster for 3 seconds
        
        # Spawn new powerup
        if time.time() - last_powerup_time > powerup_spawn_interval:
            powerups.append(Powerup())
            last_powerup_time = time.time()
        
        # Update score based on survival time
        elapsed_time = time.time() - start_time
        score = int(elapsed_time * 10)
        
        # Level up every 30 seconds
        if int(elapsed_time / 30) + 1 > level:
            level = int(elapsed_time / 30) + 1
            # Add more obstacles
            if level <= 5:  # Cap at 5 levels
                obstacles.append(Obstacle())
                monster.base_speed += 0.5  # Increase monster speed with level
            else:
                game_won = True
    
    # Draw everything
    # Draw obstacles
    for obstacle in obstacles:
        obstacle.draw()
    
    # Draw powerups
    for powerup in powerups:
        powerup.draw()
    
    # Draw player and monster
    player.draw()
    if time.time() > monster_freeze_end_time or int(time.time() * 4) % 2 == 0:  # Blink when frozen
        monster.draw()
    
    # Draw UI
    score_text = font.render(f"Score: {score}", True, WHITE)
    level_text = font.render(f"Level: {level}/5", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(level_text, (10, 40))
    
    # Draw game over screen
    if game_over:
        game_over_text = big_font.render("GAME OVER", True, RED)
        restart_text = font.render("Press R to restart", True, WHITE)
        screen.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 50))
        screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 20))
    
    # Draw game won screen
    if game_won:
        won_text = big_font.render("YOU SURVIVED!", True, GREEN)
        final_score_text = font.render(f"Final Score: {score}", True, WHITE)
        restart_text = font.render("Press R to play again", True, WHITE)
        screen.blit(won_text, (WIDTH//2 - won_text.get_width()//2, HEIGHT//2 - 50))
        screen.blit(final_score_text, (WIDTH//2 - final_score_text.get_width()//2, HEIGHT//2))
        screen.blit(restart_text, (WIDTH//2 - restart_text.get_width()//2, HEIGHT//2 + 40))
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(FPS)

# Quit pygame
pygame.quit()