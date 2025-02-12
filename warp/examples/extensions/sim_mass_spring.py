import warp as wp
import warp.sim
import warp.sim.render


def main():
    builder = wp.sim.ModelBuilder()

    # anchor point (zero mass)
    builder.add_particle((0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    # build chain
    for i in range(1, 10):
        builder.add_particle((i, 1.0, 0.0), (0.0, 0.0, 0.0), 1.0)
        builder.add_spring(i - 1, i, 1.0e3, 0.0, 0)

    # create model
    model = builder.finalize("cuda")
    model.ground = False

    # camera_pos = (5.0, 1.0, 5.0)
    # camera_front = (0.0, 0.0, -1.0)
    stage_path = "sim_mass_spring.usd"
    headless = True
    fps = 60
    dt = 1.0 / fps
    if headless:
        renderer = wp.sim.render.SimRenderer(model, stage_path, scaling=40.0)
    else:
        renderer = wp.sim.render.SimRendererOpenGL(model, stage_path)

    state_0 = model.state()
    state_1 = model.state()
    # control = model.control()  # optional, to support time-varying control inputs
    integrator = wp.sim.SemiImplicitIntegrator()
    sim_time = 0.0

    for _ in range(300):
        state_0.clear_forces()
        state_1.clear_forces()
        integrator.simulate(model, state_0, state_1, dt=dt)
        state_0, state_1 = state_1, state_0
        sim_time += dt

        renderer.begin_frame(sim_time)
        renderer.render(state_0)
        renderer.end_frame()
    renderer.save()


if __name__ == "__main__":
    main()
