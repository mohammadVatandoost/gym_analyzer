
# ToDo: complete angle conncetion for important points to analyze
angle_connection = [
    (0, 1), (1, 2), (2, 3),  # Left eye
    (0, 4), (4, 5), (5, 6),  # Right eye
    (1, 7), (4, 8),  # Ears to eyes
    (9, 10),  # Mouth
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm + fingers
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm + fingers
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg + foot
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg + foot
    (11, 12),  # Shoulder connection
    (23, 24)  # Hip connection
]