import dataclasses
import time
from threading import Lock
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import viser
import viser.transforms as vt
from jaxtyping import Float32, UInt8

from ._renderer import Renderer, RenderTask


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: Float32[np.ndarray, "4 4"]

    def get_K(self, img_wh: Tuple[int, int]) -> Float32[np.ndarray, "3 3"]:
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


@dataclasses.dataclass
class ViewerState(object):
    num_train_rays_per_sec: Optional[float] = None
    num_view_rays_per_sec: float = 100000.0
    status: Literal[
        "rendering", "preparing", "training", "paused", "completed"
    ] = "training"


VIEWER_LOCK = Lock()


def with_viewer_lock(fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        with VIEWER_LOCK:
            return fn(*args, **kwargs)

    return wrapper


class Viewer(object):
    """This is the main class for working with nerfview viewer.

    On instantiation, it (a) binds to a viser server and (b) creates a set of
    GUIs depending on its mode. After user connecting to the server, viewer
    renders and servers images in the background based on the camera movement.

    Args:
        server (viser.ViserServer): The viser server object to bind to.
        render_fn (Callable): A function that takes a camera state and image
            resolution as input and returns an image as a uint8 numpy array.
            Optionally, it can return a tuple of two images, where the second image
            is a float32 numpy depth map.
        mode (Literal["training", "rendering"]): The mode of the viewer.
            Support rendering and training. Defaults to "rendering".
    """

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable[
            [CameraState, Tuple[int, int]],
            Union[
                UInt8[np.ndarray, "H W 3"],
                Tuple[UInt8[np.ndarray, "H W 3"], Optional[Float32[np.ndarray, "H W"]]],
            ],
        ],
        vis_options: list = ["RGB"],
        mode: Literal["rendering", "training"] = "rendering",
        min_frame = 0,
        max_frame = 1,
    ):
        # Public states.
        self.server = server
        self.render_fn = render_fn
        self.mode = mode
        self.lock = VIEWER_LOCK
        self.state = ViewerState()
        if self.mode == "rendering":
            self.state.status = "rendering"

        # Private states.
        self._renderers: dict[int, Renderer] = {}
        self._step: int = 0
        self._last_update_step: int = 0
        self._last_move_time: float = 0.0

        server.on_client_disconnect(self._disconnect_client)
        server.on_client_connect(self._connect_client)

        self._frame_start = min_frame
        self._frame_end = max_frame - 1

        self._define_guis(vis_options)

    def _define_guis(self, opts):
        with self.server.gui.add_folder(
            "Stats", visible=self.mode == "training"
        ) as self._stats_folder:
            self._stats_text_fn = (
                lambda: f"""<sub>
                Step: {self._step}\\
                Last Update: {self._last_update_step}
                </sub>"""
            )
            self._stats_text = self.server.gui.add_markdown(self._stats_text_fn())

        with self.server.gui.add_folder(
            "Training", visible=self.mode == "training"
        ) as self._training_folder:
            self._pause_train_button = self.server.gui.add_button("Pause")
            self._pause_train_button.on_click(self._toggle_train_buttons)
            self._pause_train_button.on_click(self._toggle_train_s)
            self._resume_train_button = self.server.gui.add_button("Resume")
            self._resume_train_button.visible = False
            self._resume_train_button.on_click(self._toggle_train_buttons)
            self._resume_train_button.on_click(self._toggle_train_s)

            self._train_util_slider = self.server.gui.add_slider(
                "Train Util", min=0.0, max=1.0, step=0.05, initial_value=0.9
            )
            self._train_util_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Rendering") as self._rendering_folder:
            self._max_img_res_slider = self.server.gui.add_slider(
                "Max Img Res", min=64, max=2048, step=1, initial_value=2048
            )
            self._max_img_res_slider.on_update(self.rerender)

        with self.server.gui.add_folder("Dynamic") as self._camera_folder:
            self._render_frame_slider = self.server.gui.add_slider(
                "Frame", min=self._frame_start, max=self._frame_end, step=1, initial_value=self._frame_start
            )
            self._render_frame_slider.on_update(self.rerender)
            
            def add_frame(_):
                self._render_frame_slider.value += 1
                if self._render_frame_slider.value > self._frame_end:
                    self._render_frame_slider.value = self._frame_start
                self.rerender(None)
            def reduce_frame(_):
                self._render_frame_slider.value -= 1
                if self._render_frame_slider.value < self._frame_start:
                    self._render_frame_slider.value = self._frame_end
                self.rerender(None)

            self._add_frame_button = self.server.gui.add_button("Next Frame")
            self._add_frame_button.on_click(add_frame)
            self._reduce_frame_button = self.server.gui.add_button("Previous Frame")
            self._reduce_frame_button.on_click(reduce_frame)

        with self.server.gui.add_folder("Visualize Mode") as self._camera_folder:
            self._visualize_mode = self.server.gui.add_dropdown(
                "Visualize Mode", opts, initial_value="RGB"
            )
            self._visualize_mode.on_update(self.rerender)
            self._vis_gs_scale= self.server.gui.add_slider(
                "GS Scale", min=0.001, max=0.05, step=0.001, initial_value=0.01
            )
            self._vis_gs_scale.on_update(self.rerender)

        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = vt.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

    def _toggle_train_buttons(self, _):
        self._pause_train_button.visible = not self._pause_train_button.visible
        self._resume_train_button.visible = not self._resume_train_button.visible

    def _toggle_train_s(self, _):
        if self.state.status == "completed":
            return
        self.state.status = "paused" if self.state.status == "training" else "training"

    def rerender(self, _):
        clients = self.server.get_clients()
        for client_id in clients:
            camera_state = self.get_camera_state(clients[client_id])
            assert camera_state is not None
            self._renderers[client_id].submit(RenderTask("rerender", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id].running = False
        self._renderers.pop(client_id)

    def _connect_client(self, client: viser.ClientHandle):
        client_id = client.client_id
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._last_move_time = time.time()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [vt.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    def update(self, step: int, num_train_rays_per_step: int):
        if self.mode == "rendering":
            raise ValueError("`update` method is only available in training mode.")
        # Skip updating the viewer for the first few steps to allow
        # `num_train_rays_per_sec` and `num_view_rays_per_sec` to stabilize.
        if step < 5:
            return
        self._step = step
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = self._stats_text_fn()
        if len(self._renderers) == 0:
            return
        # Stop training while user moves camera to make viewing smoother.
        while time.time() - self._last_move_time < 0.1:
            time.sleep(0.05)
        if self.state.status == "training" and self._train_util_slider.value != 1:
            assert (
                self.state.num_train_rays_per_sec is not None
            ), "User must keep track of `num_train_rays_per_sec` to use `update`."
            train_s = self.state.num_train_rays_per_sec
            view_s = self.state.num_view_rays_per_sec
            train_util = self._train_util_slider.value
            view_n = self._max_img_res_slider.value**2
            train_n = num_train_rays_per_step
            train_time = train_n / train_s
            view_time = view_n / view_s
            update_every = (
                train_util * view_time / (train_time - train_util * train_time)
            )
            if step > self._last_update_step + update_every:
                self._last_update_step = step
                clients = self.server.get_clients()
                for client_id in clients:
                    camera_state = self.get_camera_state(clients[client_id])
                    assert camera_state is not None
                    self._renderers[client_id].submit(
                        RenderTask("update", camera_state)
                    )
                with self.server.atomic(), self._stats_folder:
                    self._stats_text.content = self._stats_text_fn()

    def complete(self):
        self.state.status = "completed"
        self._pause_train_button.disabled = True
        self._resume_train_button.disabled = True
        self._train_util_slider.disabled = True
        with self.server.atomic(), self._stats_folder:
            self._stats_text.content = f"""<sub>
                Step: {self._step}\\
                Training Completed!
                </sub>"""