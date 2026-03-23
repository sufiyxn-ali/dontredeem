import pyttsx3

engine = pyttsx3.init()

engine.setProperty('rate', 250)

engine.setProperty('volume', 1.0)

text = """
Attention! This is an urgent message from the bank security department.
Suspicious activity has been detected on your account.
Immediate action is required to prevent permanent loss of funds.
You must verify and transfer your funds to a secure holding account now.
Failure to act within the next few minutes may result in account suspension.
This is your final warning.
"""


engine.save_to_file(text, "sample_audio.wav")
engine.runAndWait()

print(" Generated sample_audio.wav")