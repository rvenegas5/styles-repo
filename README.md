import cv2
import pygame

# Configuración del juego
pygame.init()
screen_width, screen_height=600, 400
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Cargar recursos del juego (imágenes, sonidos, etc.)
bird_image = pygame.image.load('bird_sprite.png')
bird_width = 50
bird_height = 40

scaled_bird_image = pygame.transform.scale(bird_image, (bird_width, bird_height))

# Posición del pájaro
bird_x = 100
bird_y = screen_height // 2

def detect_face(frame):
    #XML pre-entrenado que contiene información sobre cómo detectar rostros frontales
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Se convierte a escala de grices para facilitar la deteccion y que sea mas preciso
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # devuelve una lista de rectángulos que representan las regiones en las que se han detectado los rostros. Los parámetros 1.3 y 5 son factores de escala y número mínimo de vecinos respectivamente, que influyen en la precisión y sensibilidad de la detección
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_center_x = x + (w // 2)
        face_center_y = y + (h // 2)
        return face_center_x, face_center_y
    return None, None

# Obtener el centro del rostro en cada frame de video
capture = cv2.VideoCapture(0)

# Configurar la ventana para mostrar la imagen del rostro
face_window_name = 'Detección facial'
cv2.namedWindow(face_window_name)

# Bucle principal del juego
running = True
while running:
    screen.fill((255, 255, 255))

    # Dibujar el pájaro
    screen.blit(scaled_bird_image, (bird_x, bird_y))

    pygame.display.flip()
    clock.tick(150)

    # Obtener el centro del rostro
    ret, frame = capture.read()
    if not ret:
        break
    face_x, face_y = detect_face(frame)
    if face_x is not None and face_y is not None:
        bird_y = face_y  # Controlar la posición vertical del pájaro con el centro del rostro

    # Mostrar la imagen del rostro en una ventana separada
    cv2.imshow(face_window_name, frame)

    # Manejar eventos del teclado y la ventana
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
pygame.quit()
