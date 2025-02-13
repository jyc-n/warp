import warp as wp
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, **sim_cfg):
        self.max_steps = sim_cfg["max_steps"]
        self.num_substeps = sim_cfg["num_substeps"]

        self.headless = sim_cfg["headless"]
        if self.max_steps == -1:
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

        # setup renderer
        stage_path = sim_cfg["stage_path"]
        if self.headless:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage_path, scaling=1.0)

        self.integrator = wp.sim.SemiImplicitIntegrator()

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
        """Step physics"""
        for _ in range(self.num_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # take time step
            self.integrator.simulate(self.model, self.state_0, self.state_1, dt=self.sim_dt)

            # update state
            self.state_0, self.state_1 = self.state_1, self.state_0

        return

    def render(self):
        """Step render"""
        self.renderer.begin_frame(self.sim_time)
        # render the current frame
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

        return

    def step(self):
        self.simulate()
        self.render()
        self.sim_time += self.frame_dt
        return

    def run(self):
        if self.max_steps == -1:
            while not self.renderer.has_exit:
                self.step()
        else:
            for _ in range(self.max_steps):
                self.step()

        self.renderer.save()

        return


if __name__ == "__main__":
    sim_cfg = {
        "headless": False,
        "enable_ground": False,
        "fps": 60,
        "max_steps": -1,  # use -1 for infinite loop, will disable headless
        "num_substeps": 1,
        "stage_path": "sim_mass_spring.usd",
        "geometry": {
            "num_particles": 11,
            "particle_radius": 0.2,
            "chain_length": 10,
        },
        "physics": {
            "particle_mass": 1.0,
            "spring_ke": 1.0e3,
            "spring_kd": 1.0,
        },
    }

    example = Example(**sim_cfg)
    example.run()
