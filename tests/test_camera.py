from modules.camera import Camera


def test_camera_defaults_closed():
    camera = Camera(index=0, width=640, height=480, target_fps=30)
    assert camera.is_open is False
    assert camera.fps == 0.0
