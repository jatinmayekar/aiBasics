import pyaudio
import wave
import speech_recognition as sr
import threading

# Define the basic parameters for the audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
RATE = 44100              # Sampling rate
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5        # Duration of recording
WAVE_OUTPUT_FILENAME = "output.wav"  # Output filename

# Initialize PyAudio
p = pyaudio.PyAudio()

def get_device_list():
    # Print the list of available devices and their info
    print("Available devices:\n")
    for i in range(p.get_device_count()):
        print(f"Device {i}: {p.get_device_info_by_index(i).get('name')}")

def recording_thread():
    print("Recording...")

    # Open a stream for recording
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []

    # Record the audio in chunks for the specified duration
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def listen_for_wake_word(wake_word):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for wake word...")

        while True:
            try:
                audio = recognizer.listen(source, timeout=10.0)
                try:
                    speech_text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: {speech_text}")

                    if wake_word in speech_text:
                        print(f"Wake word '{wake_word}' detected!")
                        # Call recording function in a separate thread
                        threading.Thread(target=recording_thread).start()

                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")

            except sr.WaitTimeoutError:
                pass  # Timeout occurred, just keep listening

# Start listening for the wake word
listen_for_wake_word("jarvis")
