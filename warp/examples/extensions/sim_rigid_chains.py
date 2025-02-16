import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

from sim_base import ExampleBase


class ExampleRigidChains(ExampleBase):
    def __init__(self, **sim_cfg):
        super().__init__(**sim_cfg)
        return

    def _init_scene(self, **sim_cfg):
        physics_cfg = sim_cfg["physics"]
        geo_cfg = sim_cfg["geometry"]

        self.chain_length = 8
        self.chain_width = 1.0
        # self.chain_types = [
        #     wp.sim.JOINT_REVOLUTE,
        #     wp.sim.JOINT_FIXED,
        #     wp.sim.JOINT_BALL,
        #     wp.sim.JOINT_UNIVERSAL,
        #     wp.sim.JOINT_COMPOUND,
        # ]

        builder = wp.sim.ModelBuilder()

        # start a new articulation
        builder.add_articulation()

        for i in range(self.chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 0.0, 1.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform([self.chain_width, 0.0, 0.0], wp.quat_identity())

            # create body
            b = builder.add_body(origin=wp.transform([i, 0.0, 1.0], wp.quat_identity()), armature=0.1)

            # create shape
            builder.add_shape_box(
                pos=wp.vec3(self.chain_width * 0.5, 0.0, 0.0),
                hx=self.chain_width * 0.5,
                hy=0.1,
                hz=0.1,
                density=10.0,
                body=b,
            )

            # joint_type = wp.sim.JOINT_REVOLUTE

            # joint_limit_lower = -np.deg2rad(60.0)
            # joint_limit_upper = np.deg2rad(60.0)
            joint_target = np.deg2rad(180)

            builder.add_joint_revolute(
                parent=parent,
                child=b,
                axis=(0.0, 0.0, 1.0),
                parent_xform=parent_joint_xform,
                child_xform=wp.transform_identity(),
                # limit_lower=joint_limit_lower,
                # limit_upper=joint_limit_upper,
                target=joint_target,
                target_ke=1e5,
                target_kd=1e2,
                limit_ke=1e5,
                limit_kd=1.0,
            )

        return builder

    def _init_integrator(self, **sim_cfg):
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        return


if __name__ == "__main__":
    sim_cfg = {
        "headless": True,
        "enable_ground": False,
        "fps": 60,
        "max_frames": 600,  # use -1 for infinite loop, will disable headless
        "num_substeps": 10,
        "stage_path": "sim_rigid_chain.usd",
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

    example = ExampleRigidChains(**sim_cfg)
    example.run()
