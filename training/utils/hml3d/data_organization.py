def readable_hml263_vec(v, abs=False):
    """Change the tensor to a readable dict."""
    # fmt: off
    body_rotation                   = v[..., 0]     # ⎤
    pelvis_movement                 = v[..., 1 : 3] # ⎥ root(pelvis) data
    pelvis_y                        = v[..., 3]     # ⎦
    rotation_invariant_coordinate   = v[..., 4 : 67]
    bone_rotation_6d_matrix_without = v[..., 67 : 193]
    joints_movement                 = v[..., 193 : 259]
    left_foot_contact               = v[..., 259 : 261]
    right_foot_contact              = v[..., 261 : 263]

    if abs:
        return {
            "000_body_rotation"                         : body_rotation,
            "001_pelvis_movement"                       : pelvis_movement,
            "003_pelvis_y"                              : pelvis_y,
            "004_rotation_invariant_coordinate"         : rotation_invariant_coordinate,
            "067_bone_rotation_6d_matrix_without_pelvis": bone_rotation_6d_matrix_without,
            "193_joints_movement"                       : joints_movement,
            "259_left_foot_contact"                     : left_foot_contact,
            "261_right_foot_contact"                    : right_foot_contact,
            "min / max / mean / std"                    : f"{v.min():6f} / {v.max():6f} / {v.mean():6f} / {v.std():6f}",
        }
    else:
        return {
            "000_body_rotation_per_frame"                   : body_rotation,
            "001_pelvis_movement_about_per_face_coordinates": pelvis_movement,
            "003_pelvis_y"                                  : pelvis_y,
            "004_rotation_invariant_coordinate"             : rotation_invariant_coordinate,
            "067_bone_rotation_6d_matrix_without_pelvis"    : bone_rotation_6d_matrix_without,
            "193_joints_movement_about_per_face_coordinates": joints_movement,
            "259_left_foot_contact"                         : left_foot_contact,
            "261_right_foot_contact"                        : right_foot_contact,
            "min / max / mean / std"                        : f"{v.min():6f} / {v.max():6f} / {v.mean():6f} / {v.std():6f}",
        }
    # fmt: on
