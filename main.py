import cv2
import mediapipe as mp


def contar_dedos(hand_lms):
    """Cuenta cuántos dedos están levantados."""
    dedos_levantados = 0
    dedos = [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]

    for punta_id, nudillo_id in dedos:
        if hand_lms.landmark[punta_id].y < hand_lms.landmark[nudillo_id].y:
            dedos_levantados += 1

    return dedos_levantados


def dedo_levantado(hand_lms, dedo_id):
    """Verifica si un dedo específico está levantado."""
    dedos = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}
    return hand_lms.landmark[dedo_id].y < hand_lms.landmark[dedos[dedo_id]].y


def obtener_coordenadas_dedo(hand_lms, dedo_id, alto, ancho):
    """Obtiene coordenadas (x, y) de un dedo."""
    lm = hand_lms.landmark[dedo_id]
    return int(lm.x * ancho), int(lm.y * alto)


def cargar_imagen():
    """Carga imagen opcional."""
    imagen = cv2.imread("imagen.png")
    if imagen is None:
        print("No se encontró 'imagen.png' en la carpeta del proyecto.")
    return imagen


def main():
    dispositivo_captura = cv2.VideoCapture(0)

    mp_manos = mp.solutions.hands
    manos = mp_manos.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    imagen_especial = cargar_imagen()
    cuadrados = []

    while True:
        success, imagen = dispositivo_captura.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultado = manos.process(img_rgb)

        alto, ancho, _ = imagen.shape

        # Dibujar cuadrados almacenados
        for x, y, tamaño in cuadrados:
            pt1 = (x - tamaño, y - tamaño)
            pt2 = (x + tamaño, y + tamaño)
            cv2.rectangle(imagen, pt1, pt2, (0, 255, 255), 3)

        total_dedos = 0
        manos_list = []

        if resultado.multi_hand_landmarks:
            for hand_lms in resultado.multi_hand_landmarks:
                manos_list.append(hand_lms)

                for id_lm in [4, 8, 12, 16, 20]:
                    cx, cy = obtener_coordenadas_dedo(hand_lms, id_lm, alto, ancho)
                    cv2.circle(imagen, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

                total_dedos += contar_dedos(hand_lms)

            cv2.putText(
                imagen,
                f"Dedos: {total_dedos}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

            # Dibujar cuadrado si ambos índices están levantados
            if len(manos_list) == 2:
                mano1, mano2 = manos_list

                if dedo_levantado(mano1, 8) and dedo_levantado(mano2, 8):
                    x, y = obtener_coordenadas_dedo(mano1, 8, alto, ancho)
                    tamaño = 5

                    if not any(abs(qx - x) < 20 and abs(qy - y) < 20 for qx, qy, _ in cuadrados):
                        cuadrados.append((x, y, tamaño))

            # Mostrar imagen si hay un índice y un meñique levantado
            if len(manos_list) == 2 and imagen_especial is not None:
                mano1, mano2 = manos_list

                hay_menique = dedo_levantado(mano1, 20) or dedo_levantado(mano2, 20)
                hay_indice = dedo_levantado(mano1, 8) or dedo_levantado(mano2, 8)

                if hay_menique and hay_indice:
                    img_resized = cv2.resize(imagen_especial, (200, 200))
                    x_pos = ancho - 220
                    y_pos = 20
                    imagen[y_pos:y_pos+200, x_pos:x_pos+200] = img_resized

        cv2.imshow("Detector de Manos", imagen)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('c'):
            cuadrados.clear()

    dispositivo_captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
