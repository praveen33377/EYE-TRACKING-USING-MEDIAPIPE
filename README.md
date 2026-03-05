# EYE-TRACKING-USING-MEDIAPIPE# Eye Controlled Mouse

This project is a simple **AI-based eye controlled mouse** built using Python.  
It allows you to control your computer cursor using **eye movement and facial gestures** detected through a webcam.

The system tracks facial landmarks using **MediaPipe Face Landmarker** and translates them into mouse actions such as clicking, zooming, and opening the virtual keyboard.

The main motivation behind this project is to explore **hands-free computer interaction** and accessibility technology for people who may have difficulty using a traditional mouse.

---

## Demo

You can add a demo GIF or video here.

![Demo](assets/demo.gif)

---

## Features

- Control mouse cursor using eye movement
- Blink to perform a left click
- Right wink to zoom in
- Left wink to zoom out
- Double eyebrow raise for right click
- Triple eyebrow raise to open the virtual keyboard
- Long blink (1.5 seconds) to open the keyboard
- Opening the mouth for 1.2 seconds also triggers the keyboard
- Smooth cursor movement using position buffering
- Automatic download of the required MediaPipe model
- Initial calibration for better accuracy
- Real-time face and eye tracking

---

## Technologies Used

- Python
- OpenCV
- MediaPipe Face Landmarker
- NumPy
- PyAutoGUI

These libraries are used for **computer vision, facial landmark detection, and mouse control**.

---

## How the System Works

The basic pipeline of the system is shown below:

```
Webcam Input
     │
     ▼
Face Detection (MediaPipe)
     │
     ▼
Facial Landmark Extraction
     │
     ▼
Eye & Gesture Detection
     │
     ▼
Gesture Mapping
     │
     ▼
Mouse Control using PyAutoGUI
```

The webcam captures the user's face, MediaPipe detects facial landmarks, and the program interprets specific gestures to perform mouse actions.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/eye-controlled-mouse.git
cd eye-controlled-mouse
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the program

```bash
python eye_tracker.py
```

The webcam will start and the system will begin tracking your face.

---

## Controls

| Gesture | Action |
|-------|--------|
| Blink | Left Click |
| Right Wink | Zoom In |
| Left Wink | Zoom Out |
| Two Eyebrow Raises | Right Click |
| Three Eyebrow Raises | Open Virtual Keyboard |
| Long Blink | Open Virtual Keyboard |
| Mouth Open | Open Virtual Keyboard |

---

## Calibration

When the program starts:

1. Look at the **center of your screen**
2. The system collects about **10 frames for calibration**
3. After that, cursor control will start automatically

This step helps the system determine your neutral eye position.

---

## Applications

This type of system can be useful for:

- Assistive technology for people with disabilities
- Hands-free computer control
- Human-Computer Interaction research
- Experimental AI interfaces
- Smart accessibility tools

---

## Future Improvements

Some ideas for improving the project:

- Improve gaze tracking accuracy
- Add customizable gestures
- Support multiple users
- Add machine learning based gaze estimation
- Create a mobile or embedded version

---

## Author

**Praveen**

Interested in AI, Computer Vision, and Human-Computer Interaction.

LinkedIn: (add your profile here)

---

## License

This project is licensed under the MIT License.

---

## Support

If you found this project interesting or useful, consider giving the repository a ⭐ on GitHub.

