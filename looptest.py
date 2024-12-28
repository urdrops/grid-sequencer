import pygame
import time

# Initialize Pygame mixer
pygame.mixer.init()

# Load the first music file
music_file1 = "bits/Guitar/g3.mp3"  # Replace with the path to your first music file
pygame.mixer.music.load(music_file1)

# Load the second music file as a Sound object
music_file2 = "bits/Synth/s4.mp3"  # Replace with the path to your second music file
sound2 = pygame.mixer.Sound(music_file2)

# Load the second music file as a Sound object
music_file3 = "bits/Guitar/g1.mp3"  # Replace with the path to your second music file
sound3 = pygame.mixer.Sound(music_file2)

# Play the first music file in a loop
pygame.mixer.music.play(loops=-1)

# Play the second music file in a loop
sound2.play(loops=-1)
sound3.play(loops=-1)

# Display instructions and keep the program running
print("Playing two music files in parallel. Press 'Ctrl+C' to stop.")

try:
    # Keep the script running to let the music play
    while True:
        time.sleep(1)  # Sleep to reduce CPU usage
except KeyboardInterrupt:
    # Stop the music and quit on interruption
    print("\nStopping music.")
    pygame.mixer.music.stop()
    sound2.stop()
    sound3.stop()
