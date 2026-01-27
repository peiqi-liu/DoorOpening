#!/usr/bin/env bash
set -e

PARTNET_ROOT="source/DoorOpening/assets/door/PartNetv4"
BLENDER_SCRIPT="scripts/tools/blender_to_obj.py"

# -------------
# 1. Convert DAE -> OBJ
# -------------
echo "=== Converting DAE to OBJ ==="

find "$PARTNET_ROOT" -type d -name "texture_dae" | while read -r TEX_DIR; do
    echo "Processing directory: $TEX_DIR"

    find "$TEX_DIR" -maxdepth 1 -type f -name "*.dae" | while read -r DAE_FILE; do
        OBJ_FILE="${DAE_FILE%.dae}.obj"

        if [[ -f "$OBJ_FILE" ]]; then
            echo "  [skip] $(basename "$OBJ_FILE") already exists"
            continue
        fi

        echo "  [convert] $(basename "$DAE_FILE") -> $(basename "$OBJ_FILE")"
        blender --background --python "$BLENDER_SCRIPT" -- \
            --in_file "$DAE_FILE" \
            --out_file "$OBJ_FILE"
    done
done

# -------------
# 2. Update mobility.urdf
# -------------
echo "=== Updating mobility.urdf ==="

find "$PARTNET_ROOT" -type f -name "mobility.urdf" | while read -r URDF_FILE; do
    echo "Updating: $URDF_FILE"

    # Replace only .dae with .obj (safe, minimal edit)
    sed -i.bak 's/\.dae"/\.obj"/g' "$URDF_FILE"

    # Optional: remove backup if you want
    # rm "${URDF_FILE}.bak"
done

echo "=== Done ==="
