import xml.etree.ElementTree as ET
import os
import glob
import shutil

def backup_file(path):
    bak_path = path + "_scaled.bak"
    shutil.copy(path, bak_path)
    return bak_path

# def scale_urdf_meshes(
#     input_urdf,
#     scale_factor=1.2,
#     default_scale=(1.0, 1.0, 1.0)
# ):
#     backup_file(input_urdf)
#     tree = ET.parse(input_urdf)
#     root = tree.getroot()

#     for mesh in root.findall(".//mesh"):
#         scale_str = mesh.attrib.get("scale")

#         if scale_str is None:
#             scale = list(default_scale)
#         else:
#             scale = [float(x) for x in scale_str.split()]

#         scaled = [v * scale_factor for v in scale]
#         mesh.attrib["scale"] = " ".join(f"{v:.6f}" for v in scaled)

#     tree.write(input_urdf, encoding="utf-8", xml_declaration=True)
#     print(f"Saved scaled URDF")

def scale_urdf_meshes(
    input_urdf,
    output_urdf,
    scale_factor=1.1,
    default_scale=(1.0, 1.0, 1.0)
):
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    for mesh in root.findall(".//mesh"):
        scale_str = mesh.attrib.get("scale")

        if scale_str is None:
            scale = list(default_scale)
        else:
            scale = [float(x) for x in scale_str.split()]

        scaled = [v * scale_factor for v in scale]
        mesh.attrib["scale"] = " ".join(f"{v:.6f}" for v in scaled)

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Saved scaled URDF")


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
            scale_factor=1.25,
            default_scale=(0.9, 1.0, 0.8)
        )
