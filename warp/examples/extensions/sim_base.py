import warp as wp
import warp.sim
import warp.sim.render


class ExampleBase:
    def __init__(self, **sim_cfg):
        self.max_frames = sim_cfg["max_frames"]
        self.num_substeps = sim_cfg["num_substeps"]

        self.headless = sim_cfg["headless"]
        if self.max_frames == -1:
            self.headless = False

        self.sim_time = 0.0
        self.frame_dt = 1.0 / sim_cfg["fps"]
        self.sim_dt = self.frame_dt / self.num_substeps

        # initialize scene
        builder = self._init_scene(**sim_cfg)

        # setup core data for simulation
        self.model = builder.finalize("cuda")
        self.model.ground = sim_cfg["enable_ground"]
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # simulator config
        self.integrator = wp.sim.XPBDIntegrator()

        # setup renderer
        stage_path = sim_cfg["stage_path"]
        if self.headless:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage_path, scaling=1.0)

        # this will take a step
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        return

    def _init_scene(self, **sim_cfg):
        physics_cfg = sim_cfg["physics"]
        geo_cfg = sim_cfg["geometry"]

        builder = wp.sim.ModelBuilder()

        # anchor point (zero mass)
        builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        sep = geo_cfg["chain_length"] / (geo_cfg["num_particles"] - 1)

        # build chain
        for i in range(1, geo_cfg["num_particles"]):
            builder.add_particle(
                (i * sep, 1.0, 0.0),
                (0.0, 0.0, 0.0),
                physics_cfg["particle_mass"],
                radius=geo_cfg["particle_radius"],
            )
            builder.add_spring(i - 1, i, physics_cfg["spring_ke"], physics_cfg["spring_kd"], 0)

        return builder

    def simulate(self):
        """Physics step may consist of multiple substeps"""
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # take time step
            self.integrator.simulate(self.model, self.state_0, self.state_1, dt=self.sim_dt)

            # update state
            self.state_0, self.state_1 = self.state_1, self.state_0

        return

    def step(self):
        """Step physics"""
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt
        return

    def render(self):
        """Step render"""
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            # render the current frame
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

        return

    def run(self):
        if self.max_frames == -1:
            while not self.renderer.has_exit:
                self.step()
                self.render()
        else:
            for _ in range(self.max_frames):
                self.step()
                self.render()

        self.renderer.save()
        return
