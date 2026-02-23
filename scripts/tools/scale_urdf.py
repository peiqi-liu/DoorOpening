import xml.etree.ElementTree as ET
import os
import glob
import shutil

def backup_file(path):
    bak_path = path + "_scaled.bak"
    shutil.copy(path, bak_path)
    return bak_path

def scale_urdf_meshes(
    input_urdf,
    output_urdf,
    scale_factor=1.1,
    default_scale=(1.0, 1.0, 1.0)
):
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # ---- Scale all meshes ----
    for mesh in root.findall(".//mesh"):
        scale_str = mesh.attrib.get("scale")

        if scale_str is None:
            scale = list(default_scale)
        else:
            scale = [float(x) for x in scale_str.split()]

        scaled = [v * scale_factor for v in scale]
        mesh.attrib["scale"] = " ".join(f"{v:.6f}" for v in scaled)

    # ---- Joint logic ----
    joint_2 = root.find(".//joint[@name='joint_2']")
    joint_3 = root.find(".//joint[@name='joint_3']")

    target_joint = None

    if joint_2 is not None and joint_3 is None:
        target_joint = joint_2
    elif joint_3 is not None:
        target_joint = joint_3

    if target_joint is not None:
        origin = target_joint.find("origin")
        if origin is not None and "xyz" in origin.attrib:
            x, y, z = [float(v) for v in origin.attrib["xyz"].split()]

            x += 0.07

            origin.attrib["xyz"] = f"{x:.6f} {y:.6f} {z:.6f}"

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Saved scaled URDF with conditional joint patch: {output_urdf}")


if __name__ == "__main__":
    # root_path = os.path.dirname(os.path.dirname(__file__))
    asset_base_folder = "source/DoorOpening/assets/door/PartNetv4"
    asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True))
    for asset_path in asset_paths:
        # scale_urdf_meshes(
        #     input_urdf=asset_path,
        #     scale_factor=1.2
        # )
        scale_urdf_meshes(
            input_urdf=asset_path.replace(".urdf", ".urdf_scaled.bak"),
            output_urdf=asset_path,
            scale_factor=1.2,
            default_scale=(0.6, 1.05, 0.8)
        )
