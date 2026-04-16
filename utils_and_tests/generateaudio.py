import pyttsx3

engine = pyttsx3.init()

engine.setProperty('rate', 250)

engine.setProperty('volume', 1.0)

text = """
Hi Caleb, this Alex from John Hopkins University. I am calling to follow up on your application for the research assistant position in our lab. We were impressed with your background and would like to schedule an interview with you. Please let us know your availability for the next week. We look forward to speaking with you soon. Thank you!
"""


engine.save_to_file(text, "sample_1audio.wav")
engine.runAndWait()

print(" Generated sample_audio.wav")