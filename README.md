# Memotron - Gesture Recognition Meme Classifier

## Overview
Memotron is a real-time gesture recognition system that classifies hand gestures into different meme categories using MediaPipe and PyTorch. The system captures hand landmarks through a webcam and triggers corresponding meme images and audio based on the recognized gesture.

## Project Structure
- `LeMemotron.py` - Main application for real-time gesture recognition
- `memotronTrainer.py` - Neural network training script
- `createCSV.py` - Dataset creation from captured landmarks
- `utilities.py` - Helper functions
- `models/` - Pre-trained models (MediaPipe and custom PyTorch models)
- `audios/` - Audio files for each meme category - not included here, but feel free to use your own
- `memes/` - The memes that pop on the screen when detected - not included here, but feel free to use your own

## Meme Categories
- AbsoluteCinema
<img src="Poses/AbsoluteCinema.png" alt="AbsoluteCinema" width="50%" />
- HellYeah
<img src="Poses/HellYeah.png" alt="HellYeah" width="50%" />
- Josh
<img src="Poses/Josh.png" alt="Josh" width="50%" />
- Nerd
<img src="Poses/Nerd.png" alt="Nerd" width="50%" />
- Pouce (Thumbs up)
<img src="Poses/Pouce.png" alt="Pouce" width="50%" />
- rien (Nothing/Neutral)
<img src="Poses/Rien.png" alt="Rien" width="50%" />
- Silence
<img src="Poses/Silence.png" alt="Silence" width="50%" />
- Uwu
<img src="Poses/Uwu.png" alt="Uwu" width="50%" />
- Ellie (Ellie smile meme)
<img src="Poses/Ellie.png" alt="Ellie" width="50%" />
- TheRock (Eyebrow)
<img src="Poses/TheRock.png" alt="TheRock" width="50%" />
- AngryEmoji (Anger + Fist)
<img src="Poses/AngryEmoji.png" alt="AngryEmoji" width="50%" />

You can use your own pictures to train the model, or you can simply use the already existing "memotron_model.pth" or csv file to re-train the model.
