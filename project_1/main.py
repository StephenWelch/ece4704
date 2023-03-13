import sympy as sp
import sympy.plotting as splt
from sympy import sin, cos, pi
from math import radians

def dh_to_transform(rotation, twist, displacement, offset):
    return sp.Matrix([
        [cos(rotation), -sin(rotation)*cos(twist), sin(rotation)*sin(twist), offset*cos(rotation)],
        [sin(rotation), cos(rotation)*cos(twist), -cos(rotation)*sin(twist), offset*sin(rotation)],
        [0, sin(twist), cos(twist), displacement],
    ])

def homogeneous_transform(rotation: sp.Matrix = sp.eye(3), translation: sp.Matrix = sp.zeros(3, 1)) -> sp.Matrix:
    """Return a homogeneous transform matrix given a rotation matrix and a translation vector.

    Args:
        rotation (sp.Matrix): The rotation matrix.
        translation (sp.Matrix): The translation vector.

    Returns:
        sp.Matrix: The homogeneous transform matrix.
    """
    return rotation.row_join(translation).col_join(sp.Matrix([[0, 0, 0, 1]]))

def main():
    turret_yaw, elevator_travel, elbow_pitch, wrist_pitch = sp.symbols('theta_1 d_1_2 theta_3 theta_4')
    
    ARM_LENGTH_M = 0.5

    # Define rotation/displacement matrices for each joint
    turret_rotation = sp.Matrix([
        [cos(turret_yaw), -sin(turret_yaw), 0],
        [sin(turret_yaw), cos(turret_yaw), 0],
        [0, 0, 1]
    ])
    elevator_translation = sp.Matrix([
        [0],
        [0],
        [elevator_travel]
    ])
    elbow_rotation = sp.Matrix([
        [cos(elbow_pitch), 0, sin(elbow_pitch)],
        [sin(elbow_pitch), 0, -cos(elbow_pitch)],
        [0, 1, 0]
    ])
    wrist_rotation = sp.Matrix([
        [cos(wrist_pitch), -sin(wrist_pitch), 0],
        [sin(wrist_pitch), cos(wrist_pitch), 0],
        [0, 0, 1]
    ])
    wrist_translation = sp.Matrix([
        [ARM_LENGTH_M*cos(wrist_pitch)],
        [ARM_LENGTH_M*sin(wrist_pitch)],
        [0]
    ])

    # Define transforms for each joint
    turret_transform = homogeneous_transform(rotation=turret_rotation)
    elevator_transform = homogeneous_transform(translation=elevator_translation)
    elbow_transform = homogeneous_transform(rotation=elbow_rotation)
    wrist_transform = homogeneous_transform(rotation=wrist_rotation, translation=wrist_translation)

    # Define the base to end-effector transform
    base_to_end = turret_transform * elevator_transform * elbow_transform * wrist_transform

    # Print latex for each transform & the base to end-effector transform
    print(f"Turret:\n{sp.latex(turret_transform)}\n")
    print(f"Elevator:\n{sp.latex(elevator_transform)}\n")
    print(f"Elbow:\n{sp.latex(elbow_transform)}\n")
    print(f"Wrist:\n{sp.latex(wrist_transform)}\n")
    print(sp.latex(base_to_end))
    
    # Plot position of end-effector as function of joint angles
    turret_fwd = turret_transform * sp.Matrix([0.5, 0, 0, 0])
    splt.plot_parametric((turret_fwd[0, 0], turret_fwd[1, 0]), (turret_yaw, -pi/2, pi/2))
    
    ...

if __name__ == '__main__':
    main()