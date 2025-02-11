Below are three components to help you set up and run the real‑time ray tracer project:

---

## 1. README.md

Create a file named **README.md** with the following content:

```markdown
# Real-Time Ray Tracer

This project implements a basic real‑time ray tracer in Python using Pygame and NumPy. The application demonstrates a simple dynamic scene featuring a sphere and a background gradient. You can control the camera in real time using keyboard input to move and rotate the view.

## Features

- **Interactive Camera Control:**  
  Use the following keys to navigate the scene:  
  - **W/S:** Move forward/backward  
  - **A/D:** Strafe left/right  
  - **Q/E:** Move up/down  
  - **Arrow Keys:** Rotate the camera (yaw and pitch)

- **Real‑Time Rendering:**  
  The scene is rendered at a low resolution for performance and then scaled to the window size for display.

- **Simple Scene:**  
  The scene consists of a single sphere and a background gradient.

## Requirements

- Python 3.8 or higher
- [Pygame](https://www.pygame.org/)
- [NumPy](https://numpy.org/)

## Setup Instructions

### 1. Initialize a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

Once the virtual environment is activated, install the required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Run the main Python script (e.g., `main.py`):

```bash
python main.py
```

A window will appear displaying the ray-traced scene. Use the keyboard controls to move and rotate the camera in real time.

## Project Structure

```
real_time_ray_tracer/
├── main.py           # The main Python script containing the ray tracer code.
├── README.md         # This file.
└── requirements.txt  # Contains the list of dependencies.
```

## License

This project is provided as-is without any warranty.