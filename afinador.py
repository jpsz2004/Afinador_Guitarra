import pyaudio
import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.signal import butter, lfilter

# Definir parámetros de audio
FORMAT = pyaudio.paInt16  # Formato de los datos de audio (16-bit PCM)
CHANNELS = 1              # Número de canales (mono)
RATE = 22050           # Tasa de muestreo (samples per second)
CHUNK = 32768            # Tamaño del buffer (número de frames por buffer)

REFERENCE_FREQUENCIES = {
    "E2": 164.82,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63
}

TOLERANCE = 1.0

# Inicializar PyAudio
p = pyaudio.PyAudio()

root = tk.Tk()
root.title("Afinador de Guitarra")
root.geometry("600x400")  # Fijar tamaño de la ventana
root.resizable(False, False)  # Hacer la ventana no redimensionable
root.configure(bg="lightblue")  # Fondo de la ventana

is_tuning = False

# Etiquetas para mostrar la afinación de cada cuerda
labels_frame = ttk.Frame(root)
labels_frame.pack(pady=20)

labels = {}
for string in REFERENCE_FREQUENCIES.keys():
    labels[string] = ttk.Label(labels_frame, text=f"{string}: Desconocido", font=("Helvetica", 16), background="lightblue")
    labels[string].pack(anchor=tk.CENTER, pady=5)

# Botón para iniciar/detener la captura de audio
start_button = ttk.Button(root, text="Iniciar", command=lambda: control_tuning())
start_button.pack()


#Este método se encarga de controlar el inicio y detención de la afinación
def control_tuning():
    global is_tuning
    if is_tuning:
        is_tuning = False
        start_button.configure(text="Iniciar")
    else:
        is_tuning = True
        start_button.configure(text="Detener")
        tune_guitar()

#Método que calcula la trasnsformada rápida de Fourier
def fft(data):
    fft_result = np.fft.rfft(data)
    return fft_result

#Usando la transformada rápida de Fourier se calcula la frecuencia dominante
def dominant_freq(data):
    frequencies = np.fft.rfftfreq(len(data), 1.0 / RATE)
    magnitude = np.abs(fft(data))

    # Frecuencias de interés
    min_freq = 70.0
    max_freq = 350.0
    relevant_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))
    relevant_frequencies = frequencies[relevant_indices]
    relevant_magnitudes = magnitude[relevant_indices]

    # Encontrar la frecuencia dominante
    if len(relevant_frequencies) > 0:
        dominant_index = np.argmax(relevant_magnitudes)
        dominant_freq = relevant_frequencies[dominant_index]
        dominant_magnitude = relevant_magnitudes[dominant_index]

        return dominant_freq, dominant_magnitude
    else:
        return None, None

#Este método se encarga de encontrar la cuerda más cercana a la frecuencia dominante
def find_closest_string(frequency, references):
    closest_string = None
    min_distance = float('inf')

    for string, ref_freq in references.items():
        distance = abs(frequency - ref_freq)
        if distance < min_distance:
            min_distance = distance
            closest_string = string

    return closest_string, min_distance


#Este método se encarga de aplicar un filtro pasa-bajo a los datos de audio
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

#Este método se encarga de aplicar el filtro pasa-bajo a los datos de audio
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

#Función principal para afinar la guitarra
def tune_guitar():
    if not is_tuning:
        return

    # Leer datos del stream
    data = stream.read(CHUNK)

    # Convertir los datos a un array de numpy
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Verificar el tamaño de audio_data
    if len(audio_data) != CHUNK:
        print(f"Tamaño inesperado de audio_data: {len(audio_data)}")
        return

    # Aplicar un filtro pasa-bajo para eliminar frecuencias altas
    filtered_data = lowpass_filter(audio_data, 350.0, RATE)

    # Aplicar la ventana de Hamming
    windowed_data = filtered_data * np.hamming(len(filtered_data))

    # Calcular la frecuencia dominante
    dominant_frequency, _ = dominant_freq(windowed_data)
    if dominant_frequency is not None:
        closest_string, min_distance = find_closest_string(dominant_frequency, REFERENCE_FREQUENCIES)
        
        # Actualizar las etiquetas
        if min_distance <= TOLERANCE:
            if closest_string == "E2":
                dominant_frequency /= 2
                labels[closest_string].config(text=f"{closest_string}: Afinado ({dominant_frequency:.2f} Hz)", foreground="green")
            labels[closest_string].config(text=f"{closest_string}: Afinado ({dominant_frequency:.2f} Hz)", foreground="green")
        elif dominant_frequency > REFERENCE_FREQUENCIES[closest_string]:
            if closest_string == "E2":
                dominant_frequency /= 2
                labels[closest_string].config(text=f"{closest_string} Alta ({dominant_frequency:.2f} Hz)", foreground="red")
            labels[closest_string].config(text=f"{closest_string} Alta ({dominant_frequency:.2f} Hz)", foreground="red")
        else:
            if closest_string == "E2":
                dominant_frequency /= 2
                labels[closest_string].config(text=f"{closest_string} Baja ({dominant_frequency:.2f} Hz)", foreground="blue")
            labels[closest_string].config(text=f"{closest_string} Baja ({dominant_frequency:.2f} Hz)", foreground="blue")

    root.after(100, tune_guitar)

# Abrir un stream de audio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Captura de audio en tiempo real iniciada. Presiona Ctrl+C para detener.")



root.mainloop()

# Cerrar el stream y PyAudio
stream.stop_stream()
stream.close()
p.terminate()
