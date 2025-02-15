import warp as wp
import warp.sim
import warp.sim.render

from sim_base import ExampleBase


class ExampleMassSpring(ExampleBase):
    def __init__(self, **sim_cfg):
        super().__init__(**sim_cfg)
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


if __name__ == "__main__":
    sim_cfg = {
        "headless": True,
        "enable_ground": False,
        "fps": 60,
        "max_frames": 600,  # use -1 for infinite loop, will disable headless
        "num_substeps": 2,
        "stage_path": "sim_mass_spring.usd",
        "geometry": {
            "num_particles": 11,
            "particle_radius": 0.2,
            "chain_length": 10,
        },
        "physics": {
            "particle_mass": 1.0,
            "spring_ke": 1.0e6,
            "spring_kd": 1.0,
        },
    }

    example = ExampleMassSpring(**sim_cfg)
    example.run()
