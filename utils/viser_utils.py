import nerfview
import viser
import time
import torch
from typing import Tuple
from utils.general_utils import visualize_depth


class VisManger:
    def __init__(self, cfg, render_fn):
        self.vis_modes = cfg["vis_modes"]

        self.device = cfg["device"]
        self.render_fn = render_fn
        self.min_frame = cfg["min_frame"]
        self.max_frame = cfg["max_frame"]
        self.cfg = cfg

        print(f"Viser server started at {cfg['port']}")
        self.server = viser.ViserServer(port=cfg["port"], verbose=False)

        self.viewer = nerfview.Viewer(
            server = self.server,
            render_fn = self.render_cb,
            vis_options = self.vis_modes,
            mode="training",
            min_frame=self.min_frame,
            max_frame=self.max_frame,
        )

        # setup init 
        # @server.on_client_connect
        # def _(client: viser.ClientHandle) -> None:
        #     client.camera.position = (1., 1., 1.)
        #     client.camera.look_at = (0., 0., 0.)

        # runtime
        self.tic = None
        self.W = None
        self.H = None


    def checkin(self):
        while self.viewer.state.status == "paused":
            time.sleep(0.01)
        self.viewer.lock.acquire()
        self.tic = time.time()

    def checkout(self, step):
        num_of_pixel = 66*515
        self.viewer.lock.release()
        num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
        self.viewer.state.num_train_rays_per_sec = \
            num_of_pixel * num_train_steps_per_sec
        self.viewer.update(step, num_of_pixel)
    
    @torch.no_grad()
    def render_cb(self,
                  cam: nerfview.CameraState,
                  img_wh: Tuple[int, int],
                  frame = 0,
                  mode = "depth",           # rendering mode
                  init_scale = 0.02):       # init scale for vis Gaussian xyz

        self.W, self.H = img_wh
        c2w = cam.c2w
        K = cam.get_K(img_wh)

        # c2w = torch.from_numpy(c2w).float().to(self.device)
        # K = torch.from_numpy(K).float().to(self.device)

        render_pkg = self.render_fn(
            c2w=c2w,
            K=K,
            width=self.W,
            height=self.H,
            frame=frame,
       )
        depth = render_pkg['depth']
        far = 20.
        # replace zero into far
        depth = visualize_depth(depth, near=1.5, far=far, cmap="gray").permute(1, 2, 0)
        return depth.cpu().numpy()