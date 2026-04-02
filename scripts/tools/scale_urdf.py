import xml.etree.ElementTree as ET
import os
import glob
import shutil


def backup_file(path):
    bak_path = path + "_scaled.bak"
    shutil.copy(path, bak_path)
    return bak_path


def shift_link_up(root, link_name, y_shift):
    if y_shift == 0.0:
        return

    for tag in ("visual", "collision"):
        origin = root.find(f".//link[@name='{link_name}']/{tag}/origin")
        if origin is None or "xyz" not in origin.attrib:
            continue

        x, y, z = [float(v) for v in origin.attrib["xyz"].split()]
        y += y_shift
        origin.attrib["xyz"] = f"{x:.6f} {y:.6f} {z:.6f}"


def shift_door_up(root, y_shift):
    if y_shift == 0.0:
        return

    for link_name in ("link_0", "link_1", "link_2", "link_3"):
        shift_link_up(root, link_name, y_shift)


def scale_urdf_meshes(
    input_urdf,
    output_urdf,
    scale_factor=1.1,
    default_scale=(1.0, 1.0, 1.0),
    door_y_shift=0.0,
):
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    # ---- Scale all meshes ----
    for mesh in root.findall(".//mesh"):
        scale_str = mesh.attrib.get("scale")

        if scale_str is None:
            scale = [1.0, 1.0, 1.0]
        else:
            scale = [float(x) for x in scale_str.split()]

        scaled = [scale[i] * default_scale[i] * scale_factor for i in range(3)]
        mesh.attrib["scale"] = " ".join(f"{v:.6f}" for v in scaled)

    # ---- Shift the short door meshes upward together ----
    shift_door_up(root, door_y_shift)

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

            x += 0.03

            origin.attrib["xyz"] = f"{x:.6f} {y:.6f} {z:.6f}"

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Saved scaled URDF with conditional joint patch: {output_urdf}")


if __name__ == "__main__":
    # root_path = os.path.dirname(os.path.dirname(__file__))
    asset_base_folder = "source/DoorOpening/assets/door/PartNetv4"
    asset_paths = sorted(glob.glob(os.path.join(asset_base_folder, "**/mobility.urdf"), recursive=True))
    scale_factors = [1.0] * len(asset_paths)
    door_y_shifts = [0.0] * len(asset_paths)

    # # The 9th door panel is too short.
    # scale_factors[8] = 1.2
    # # Compensate the center scaling by shifting the whole door mesh upward a bit.
    # door_y_shifts[8] = 0.2

    for i, asset_path in enumerate(asset_paths):
        scale_urdf_meshes(
            input_urdf=asset_path.replace(".urdf", ".urdf_scaled.bak"),
            output_urdf=asset_path,
            scale_factor=scale_factors[i],
            default_scale=(1.2, 1.0, 1.0),
            door_y_shift=door_y_shifts[i],
        )
