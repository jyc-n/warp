import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom
from sim_base import ExampleBase

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"

    def __str__(self):
        return self.value


class ExampleCloth(ExampleBase):
    def __init__(self, **sim_cfg):
        super().__init__(**sim_cfg)

        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

    def _init_scene(self, **sim_cfg):
        builder = wp.sim.ModelBuilder()

        geo_cfg = sim_cfg["geometry"]
        height = geo_cfg["cloth_grid_height"]
        width = geo_cfg["cloth_grid_width"]

        solver_cfg = sim_cfg["solver"]

        # add cloth
        if solver_cfg["integrator"] == IntegratorType.EULER:
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=height,
                dim_y=width,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=1.0e1,
            )
        elif solver_cfg["integrator"] == IntegratorType.XPBD:
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=height,
                dim_y=width,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                edge_ke=1.0e2,
                add_springs=True,
                spring_ke=1.0e3,
                spring_kd=0.0,
            )

        # add collider
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

        builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            pos=wp.vec3(1.0, 0.0, 1.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5),
            scale=wp.vec3(2.0, 2.0, 2.0),
            ke=1.0e2,
            kd=1.0e2,
            kf=1.0e1,
        )

        # self.model = builder.finalize()
        # self.model.ground = True
        return builder

    def _init_integrator(self, **sim_cfg):
        solver_cfg = sim_cfg["solver"]

        if solver_cfg["integrator"] == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif solver_cfg["integrator"] == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=1)
        # else:
        #     self.integrator = wp.sim.VBDIntegrator(self.model, iterations=1)

        return


if __name__ == "__main__":
    sim_cfg = {
        "headless": True,
        "enable_ground": True,
        "fps": 60,
        "max_frames": 600,  # use -1 for infinite loop, will disable headless
        "num_substeps": 10,
        "stage_path": "sim_cloth.usd",
        "geometry": {
            "cloth_grid_height": 32,
            "cloth_grid_width": 64,
        },
        "physics": {
            "particle_mass": 1.0,
            "spring_ke": 1.0e6,
            "spring_kd": 1.0,
        },
        "solver": {
            "integrator": IntegratorType.XPBD,
        },
    }

    example = ExampleCloth(**sim_cfg)
    example.run()
