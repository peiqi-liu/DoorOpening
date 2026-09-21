from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
import omni.physx
physx_interface = omni.physx.get_physx_interface()
print("PHYSX_CHECK_DONE")
simulation_app.close()
